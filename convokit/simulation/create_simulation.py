"""Run the agent conversations described by a simulation YAML on a ConvoArena backend.

This is the lower-level API behind :func:`convokit.simulation.simulation.simulate`. :func:`create_simulation`
runs the conversations on a backend that is already running (a local emulator suite via
:class:`convokit.simulation.local_backend.LocalBackend`, or your own ConvoArena deployment via
:class:`FirebaseBackend`) and returns them as a ConvoKit Corpus.

A simulation YAML looks like this::

    description: ...            # shown with the chat, and used as the post description
    blocks:                     # text blocks shown with the chat; the first one fills
      - name: Debate topic      # {topic_name} / {topic_statement} in prompts
        descriptions:           # one conversation is run per description (per combination,
          - Cats or dogs?       # with several blocks)
    max_utterance: 15
    max_time: 4                 # minutes
    pairings:                   # one conversation per pairing and block combination
      - id: ...
        members:
          - participant: mediator:<key in definitions.mediators>
            assistant: null
          - participant: <key in definitions.agents>
            assistant: <key in definitions.assistants, or null>
    definitions:
      agents: {<key>: {name: ..., content: {...}}}
      mediators: {...}
      assistants: {...}

The YAML is translated into ConvoArena experiment templates (ConvoArena is built on
Deliberate Lab, https://github.com/PAIR-code/deliberate-lab) the same way the mediator toolkit creates its experiments. Each pairing
and block combination runs as its own experiment with a single cohort, since mediators join
every cohort of their experiment.

Requirements:

- A Gemini API key for the agents. Without one, the agents are created but every model call
  fails, so no messages are sent. :func:`convokit.simulation.simulation.simulate` stores the key in the local
  backend for you; on your own deployment, save it in the web UI's Settings.
- To start the emulators from source (as :func:`main` does), a ConvoArena checkout set
  up as described in :mod:`convokit.simulation.local_backend`. :func:`convokit.simulation.simulation.simulate` uses the backend
  container instead and needs no checkout.

Running this module as a script runs a simulation on the emulators of a local ConvoArena
checkout and prints the resulting transcripts (see :func:`main`)::

    GEMINI_API_KEY=... python -m convokit.simulation.create_simulation /path/to/convoarena simulation.yaml
"""

from __future__ import annotations

import itertools
import os
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import requests
import yaml
from tqdm.auto import tqdm

from convokit.model import Corpus
from . import deliberate_lab as dl
from .export_to_corpus import export_to_corpus
from .local_backend import LocalBackend


@dataclass
class FirebaseBackend:
    """Your own deployed ConvoArena backend on Firebase.

    :param project_id: Firebase project ID of the deployment
    :param api_key: ConvoArena API key, created in the deployment's web UI under
        Settings -> API Keys
    :param region: region the deployment's Cloud Functions run in
    """

    project_id: str
    api_key: str
    region: str = "us-central1"

    @property
    def base_url(self) -> str:
        """Base URL of the deployment's ConvoArena REST API."""
        return f"https://{self.region}-{self.project_id}.cloudfunctions.net/api/v1"

    def client(self) -> dl.Client:
        """Build a ``deliberate_lab.Client`` connected to the deployment."""
        return dl.Client(base_url=self.base_url, api_key=self.api_key)


PROFILE_STAGE_ID = "profile"
CHAT_STAGE_ID = "chat-round-1"
STAGE_IDS = [PROFILE_STAGE_ID, CHAT_STAGE_ID]

# prompt item types passed through to ConvoArena unchanged
PASS_THROUGH_ITEMS = {
    "PROFILE_INFO",
    "PARTICIPANT_INFO",
    "PARTICIPANT_CHAT_INPUT",
    "PROFILE_CONTEXT",
    "INITIALIZATION_CONTEXT",
}

# r/ChangeMyView rules referenced by RULE prompt items
CMV_RULES = {
    "A": (
        "Rule A - Doesn't Explain View",
        "Explain the reasoning behind your view, not just what that view is (500+ human-generated characters required).",
    ),
    "B": (
        "Rule B - 3rd Party/Devils Advocate/Soapboxing",
        "You must personally hold the view and demonstrate that you are open to it changing. A post cannot be on behalf of others, playing devil's advocate, as any entity other than yourself, or 'soapboxing'. Posts by throwaway accounts must be approved through modmail.",
    ),
    "C": (
        "Rule C - Unclear/Improper Title",
        'Submission titles must adequately sum up your view and include "CMV:" at the beginning. Posts with misleading/overly-simplistic titles or titles that contain spoilers may be removed.',
    ),
    "D": (
        "Rule D - Neutral/Transgender/Harm a specific person/Promo/Meta",
        "Posts cannot express a neutral stance, a stance regarding transgender topics, suggest harm against a specific person, be self-promotional, or discuss this subreddit (visit r/ideasforcmv instead).",
    ),
    "E": (
        "Rule E - No/Minimal Replies from OP in 2 hours",
        "Only post if you are willing to have a conversation with those who reply to you, and are available to do so within 2 hours of your post going live. If you haven't replied during this time, your post will be removed.",
    ),
    "1": (
        "Rule 1 - Doesn't Challenge OP (top-level only)",
        "Direct responses to a CMV post must challenge at least one aspect of OP's stated view (however minor), unless they are asking a clarifying question.",
    ),
    "2": (
        "Rule 2 - Rude/Hostile Comment",
        "Don't be rude or hostile to other users. Your comment will be removed even if the rest of it is solid. 'They started it' is not an excuse. You should report it, not respond to it.",
    ),
    "3": (
        "Rule 3 - Bad Faith Accusation",
        "Refrain from accusing OP or anyone else of being unwilling to change their view, of using AI to generate their post or comment, of lying, or of arguing in bad faith. If you are unsure whether someone is genuine, ask clarifying questions (see: socratic method). If you think they are still exhibiting ill behaviour, please message us.",
    ),
    "4": (
        "Rule 4 - Delta Abuse/Misuse or Should Award Delta",
        "Award a delta if you've acknowledged a change in your view. Do not use deltas for any other purpose. You must include an explanation of the change along with the delta so we know it's genuine. Delta abuse includes sarcastic deltas, joke deltas, super-upvote deltas, etc.",
    ),
    "5": (
        "Rule 5 - Doesn't Contribute Meaningfully",
        'Comments must contain human-generated content and contribute meaningfully to the conversation. Comments that are only links, jokes, or "written upvotes" will be removed. Humor and affirmations of agreement can be contained within more substantial comments.',
    ),
}


# Added to an agent prompt that has no instructions of its own (e.g. only CONTEXT items), which
# would otherwise leave the model with nothing to do, so the agent never speaks.
DEFAULT_AGENT_INSTRUCTIONS = (
    "You are the participant above in a live online conversation. Reply to the conversation "
    "in character with a short, natural 1-2 sentence message."
)

# Older ConvoArena deployments make a "thought" model call on every agent participant turn
# and crash if this prompt is missing. The stub keeps that call cheap (newer backends only
# record the thought).
AGENT_THOUGHT_PROMPT = [
    {"type": "TEXT", "text": 'Return exactly this JSON and nothing else: {"thought": ""}'}
]


@dataclass
class _PromptContext:
    """Values that fill in the simulation YAML's placeholder prompt items."""

    substitutions: dict[str, str]
    post_title: str
    post_description: str
    role: str = ""
    character: str = ""


def load_simulation_config(sim_yaml: str | Path | dict) -> dict:
    """Load a simulation config from a YAML file path, a YAML string, or an already-parsed dict.

    :param sim_yaml: path to the simulation YAML, the YAML itself as a string, or the parsed dict
    :return: the simulation config
    :raises ValueError: if ``sim_yaml`` is None, or the config is not a mapping or defines no
        pairings
    """
    if sim_yaml is None:
        raise ValueError("sim_yaml is required: a path to the simulation YAML, its text, or a dict")
    if isinstance(sim_yaml, dict):
        config = sim_yaml
    elif isinstance(sim_yaml, Path) or os.path.isfile(sim_yaml):
        config = yaml.safe_load(Path(sim_yaml).read_text(encoding="utf-8"))
    else:
        config = yaml.safe_load(sim_yaml)
    if not isinstance(config, dict) or not config.get("pairings"):
        raise ValueError("the simulation YAML must define at least one pairing")
    return config


def _get(d: dict | None, key: str, default=None):
    """Return ``d[key]``, accepting the snake_case key or its camelCase form (YAMLs use both)."""
    if not d:
        return default
    if key in d:
        return d[key]
    head, *rest = key.split("_")
    return d.get(head + "".join(word.capitalize() for word in rest), default)


def _exclude_none(value):
    """Drop None values recursively, like pydantic's model_dump(exclude_none=True)."""
    if isinstance(value, dict):
        return {k: _exclude_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_exclude_none(v) for v in value]
    return value


def _block_variants(config: dict) -> list[list[dict]]:
    """Return every combination of the blocks' descriptions, each a list of {name, description}."""
    options = [
        [
            {"name": block.get("name", ""), "description": description}
            for description in (block.get("descriptions") or [block.get("description", "")])
        ]
        for block in config.get("blocks") or []
    ]
    return [list(combo) for combo in itertools.product(*options)]


def _base_context(config: dict, blocks: list[dict]) -> _PromptContext:
    """Build the prompt placeholder values shared by all members for one block combination."""
    topic = blocks[0] if blocks else {"name": "", "description": ""}
    post_title = config.get("post_title") or topic["description"]
    post_description = config.get("post_description") or config.get("description", "")
    return _PromptContext(
        substitutions={
            "{topic_name}": topic["name"],
            "{topic_statement}": topic["description"],
            "{post_title}": post_title,
            "{post_description}": post_description,
        },
        post_title=post_title,
        post_description=post_description,
    )


def _context_items(context: str) -> list[dict]:
    """Expand a CONTEXT item into STAGE_CONTEXT items for the chat stage (``'current'``),
    the stages before it (``'before'``), or both (``'all'``)."""
    chat_index = STAGE_IDS.index(CHAT_STAGE_ID)
    if context == "all":
        stage_ids = STAGE_IDS[: chat_index + 1]
    elif context == "before":
        stage_ids = STAGE_IDS[:chat_index]
    elif context == "current":
        stage_ids = [CHAT_STAGE_ID]
    else:
        raise ValueError(f"unknown context {context!r}; must be 'all', 'before', or 'current'")
    return [
        {
            "type": "STAGE_CONTEXT",
            "stageId": stage_id,
            "includePrimaryText": True,
            "includeInfoText": False,
            "includeHelpText": False,
            "includeStageDisplay": True,
            "includeParticipantAnswers": True,
        }
        for stage_id in stage_ids
    ]


def _prompt_items(
    items: list[dict] | None, default_context: str | None, ctx: _PromptContext
) -> list[dict]:
    """Translate simulation YAML prompt items into ConvoArena prompt items."""
    out = []
    for item in sorted(items or [], key=lambda i: i.get("id", 0)):
        kind = item["type"]
        if kind == "CONTEXT":
            out += _context_items(item.get("context") or default_context or "current")
        elif kind in PASS_THROUGH_ITEMS:
            out.append({"type": kind})
        elif kind == "PRELOADED_CONTEXT":
            out.append({"type": "INITIALIZATION_CONTEXT"})
        elif kind == "TEXT":
            text = item.get("text") or ""
            for token, value in ctx.substitutions.items():
                text = text.replace(token, value)
            out.append({"type": "TEXT", "text": text})
        elif kind == "POST_TITLE":
            out.append({"type": "TEXT", "text": f"Title: {ctx.post_title}"})
        elif kind == "POST_DESCRIPTION":
            out.append({"type": "TEXT", "text": f"Description: {ctx.post_description}"})
        elif kind == "ARTICLE_PAGE":
            out.append({"type": "TEXT", "text": f"{ctx.post_title}\n{ctx.post_description}"})
        elif kind == "PARTICIPANT_ROLE":
            out.append({"type": "TEXT", "text": f"Role: {ctx.role}"})
        elif kind == "CHARACTER_CONTEXT":
            out.append({"type": "TEXT", "text": ctx.character})
        elif kind == "RULE":
            rule = str(item.get("rule"))
            if rule not in CMV_RULES:
                raise ValueError(f"unknown rule {rule!r}; must be one of {', '.join(CMV_RULES)}")
            title, description = CMV_RULES[rule]
            out.append(
                {"type": "TEXT", "text": f"Rule Title: {title}\nRule Description: {description}"}
            )
        else:
            raise ValueError(f"unknown prompt item type {kind!r}")
    return out


def _persona(content: dict, persona_type: str) -> dict:
    """Build the persona part of a ConvoArena agent template."""
    persona = content["persona"]
    model = content["model"]
    name = persona.get("name", "")
    return {
        "id": persona["id"],
        "name": name,
        "type": persona_type,
        "defaultProfile": {
            "name": name,
            "avatar": persona.get("avatar"),
            "pronouns": persona.get("pronouns"),
        },
        "defaultModelSettings": {"apiType": model["apiType"], "modelName": model["modelName"]},
    }


def _generation(content: dict) -> dict:
    """Build a ConvoArena generation config from a definition's ``generation`` settings."""
    generation = content.get("generation") or {}
    return {
        "temperature": generation.get("temperature"),
        "reasoningLevel": _get(generation, "reasoning_level"),
        "includeReasoning": _get(generation, "include_reasoning"),
    }


def _chat_settings(settings: dict | None) -> dict:
    """Build ConvoArena chat settings, filling in defaults."""
    return {
        "minMessagesBeforeResponding": _get(settings, "min_messages_before_responding", 0),
        "canSelfTriggerCalls": _get(settings, "can_self_trigger_calls", False),
        "initialMessage": _get(settings, "initial_message", ""),
        "wordsPerMinute": _get(settings, "words_per_minute", 0),
    }


def _structured_output(config: dict | None) -> dict | None:
    """Build a ConvoArena structured output config, or None if none is configured."""
    if not config:
        return None
    return {
        "enabled": config.get("enabled", True),
        "type": "JSON_SCHEMA",
        "appendToPrompt": _get(config, "append_to_prompt", False),
        "shouldRespondField": _get(config, "should_respond_field"),
        "messageField": _get(config, "message_field"),
        "explanationField": _get(config, "explanation_field"),
        "readyToEndField": _get(config, "ready_to_end_field"),
        "schema": {
            "type": "OBJECT",
            "properties": [
                {
                    "name": name,
                    "schema": {"type": field["type"], "description": field.get("description", "")},
                }
                for name, field in (config.get("schema") or {}).items()
            ],
        },
    }


def _mediator_template(content: dict, ctx: _PromptContext) -> dict:
    """Build the ConvoArena agent mediator template for a mediator definition."""
    context = content.get("context")
    prompt_config = {
        "id": CHAT_STAGE_ID,
        "type": "chat",
        "includeScaffoldingInPrompt": _get(content, "include_scaffolding_in_prompt"),
        "prompt": _prompt_items(content.get("prompt"), context, ctx),
        "shouldRespondPrompt": _prompt_items(
            _get(content, "should_respond_prompt"),
            _get(content, "should_respond_context") or context,
            ctx,
        ),
        "minParticipantMessagesBeforeResponding": _get(
            content, "min_participant_messages_before_responding"
        ),
        "structuredOutputConfig": _structured_output(_get(content, "structured_output")),
        "generationConfig": _generation(content),
        "chatSettings": _chat_settings(_get(content, "chat_settings")),
        "numRetries": _get(content, "num_retries"),
    }
    init_prompt = _get(content, "initialization_context_prompt") or _get(
        content, "preload_context_prompt"
    )
    if init_prompt:
        prompt_config["initializationContextPrompt"] = _prompt_items(
            init_prompt, _get(content, "initialization_context_context") or context, ctx
        )
    persona = _persona(content, "mediator")
    persona["isDefaultAddToCohort"] = True
    return {"persona": persona, "promptMap": {CHAT_STAGE_ID: prompt_config}}


def _assistant_template(content: dict, ctx: _PromptContext) -> dict:
    """Build the ConvoArena private assistant template for an assistant definition."""
    context = content.get("context")
    persona = _persona(content, "assistant")
    persona["minCallIntervalMs"] = _get(content["persona"], "min_call_interval_ms")
    if ctx.role:
        # PARTICIPANT_ROLE makes the prompt role-specific, so each role needs its own assistant
        persona["id"] = f"{persona['id']}-{ctx.role.lower().replace(' ', '-')}"
    prompt_config = {
        "id": CHAT_STAGE_ID,
        "type": "chat",
        "prompt": _prompt_items(content.get("prompt"), context, ctx),  # a list, see _agent_template
        "order": {},
        "addTo": {},
        "shouldRespondPrompt": _prompt_items(
            _get(content, "should_respond_prompt"),
            _get(content, "should_respond_context") or context,
            ctx,
        ),
        "structuredOutputConfig": _structured_output(_get(content, "structured_output")),
        "generationConfig": _generation(content),
        "numRetries": _get(content, "num_retries"),
    }
    return {"persona": persona, "promptMap": {CHAT_STAGE_ID: prompt_config}}


def _agent_template(
    content: dict, persona_id: str, assistant_id: str | None, ctx: _PromptContext
) -> dict:
    """Build the ConvoArena agent participant template for an agent definition."""
    settings = _get(content, "chat_settings") or {}
    context = settings.get("context") or content.get("context")
    prompt_map = _get(settings, "prompt_map")
    if prompt_map:
        prompts = {
            key: _prompt_items(entry.get("prompt"), context, ctx)
            for key, entry in prompt_map.items()
        }
        order: dict[int, list[str]] = {}
        for key, entry in prompt_map.items():
            order.setdefault(entry.get("order", 1), []).append(key)
    else:
        prompts = {"default": _prompt_items(content.get("prompt"), context, ctx)}
        order = {1: ["default"]}

    persona = _persona(content, "participant")
    persona["id"] = persona_id
    persona["assistantId"] = assistant_id

    has_instructions = any(
        item["type"] == "TEXT" and item["text"].strip()
        for items in prompts.values()
        for item in items
    )
    if not has_instructions:
        # the last pipeline step writes the chat message
        key = order[max(order)][-1]
        prompts[key] = [
            {"type": "PROFILE_INFO"},
            *prompts[key],
            {"type": "TEXT", "text": DEFAULT_AGENT_INSTRUCTIONS},
        ]

    chat_settings = _chat_settings(settings)
    chat_settings["thoughtPrompt"] = AGENT_THOUGHT_PROMPT

    prompt_config = {
        "id": CHAT_STAGE_ID,
        "type": "chat",
        "includeScaffoldingInPrompt": _get(
            settings,
            "include_scaffolding_in_prompt",
            _get(content, "include_scaffolding_in_prompt"),
        ),
        # a single prompt is sent as a plain list, which older ConvoArena deployments
        # (from before keyed prompts) require and newer ones still accept
        "prompt": prompts if prompt_map else prompts["default"],
        "order": order,
        "addTo": {},
        "structuredOutputConfig": _structured_output(
            _get(content, "structured_output") or _get(settings, "structured_output")
        ),
        "generationConfig": _generation(content),
        "chatSettings": chat_settings,
        "numRetries": _get(settings, "num_retries", _get(content, "num_retries")),
    }
    return {"persona": persona, "promptMap": {CHAT_STAGE_ID: prompt_config}}


def _definition(definitions: dict, section: str, key: str) -> dict:
    """Return the content of ``definitions[section][key]``, raising ValueError if it is missing."""
    try:
        entry = definitions[section][key]
    except (KeyError, TypeError):
        raise ValueError(
            f"a pairing references {key!r}, which is not in definitions.{section}"
        ) from None
    return entry.get("content", entry)


def _experiment_template(
    config: dict, pairing: dict, blocks: list[dict]
) -> tuple[dict, dict, list[dict]]:
    """Build the experiment template for one pairing and block combination.

    :return: the template, its cohort participant config, and the agent participant templates
        to add to the cohort
    """
    definitions = config.get("definitions") or {}
    ctx = _base_context(config, blocks)

    mediators, agents = [], []
    assistants: dict[str, dict] = {}
    persona_ids: set[str] = set()
    for member in pairing.get("members") or []:
        kind, _, key = member["participant"].rpartition(":")
        if kind == "mediator":
            mediators.append(_mediator_template(_definition(definitions, "mediators", key), ctx))
            continue
        if kind not in ("", "agent"):
            raise ValueError(f"unknown participant {member['participant']!r}")

        content = _definition(definitions, "agents", key)
        character = content["persona"].get("character")
        member_ctx = replace(
            ctx,
            role=member.get("role") or "",
            character=character if isinstance(character, str) else "",
        )
        assistant_id = None
        if member.get("assistant"):
            assistant = _assistant_template(
                _definition(definitions, "assistants", member["assistant"]), member_ctx
            )
            assistant_id = assistant["persona"]["id"]
            assistants.setdefault(assistant_id, assistant)

        # the same agent can appear twice in a pairing, but persona ids must be unique
        persona_id = base_id = content["persona"]["id"]
        suffix = 2
        while persona_id in persona_ids:
            persona_id = f"{base_id}-{suffix}"
            suffix += 1
        persona_ids.add(persona_id)
        agents.append(_agent_template(content, persona_id, assistant_id, member_ctx))

    description = config.get("description", "")
    cohort_config = {
        "minParticipantsPerCohort": len(agents),
        "maxParticipantsPerCohort": len(agents),
        "includeAllParticipantsInCohortCount": True,
        "botProtection": False,
    }
    stages = [
        {
            "id": PROFILE_STAGE_ID,
            "kind": "profile",
            "name": "Profile Setup",
            "descriptions": {"primaryText": "Set up your profile", "infoText": "", "helpText": ""},
            "progress": {
                "minParticipants": 1,
                "waitForAllParticipants": False,
                "showParticipantProgress": False,
            },
            "profileType": "ANONYMOUS_ANIMAL",
        },
        {
            "id": CHAT_STAGE_ID,
            "kind": "chat",
            "name": "Conversation",
            "descriptions": {
                "primaryText": description,
                "infoText": "",
                "helpText": "",
                "blocks": blocks,
            },
            "progress": {
                "minParticipants": len(agents),
                "waitForAllParticipants": False,
                "showParticipantProgress": True,
            },
            "discussions": [],
            "timeLimitInMinutes": config.get("max_time"),
            "requireFullTime": False,
            "numUtterances": config.get("max_utterance"),
        },
    ]
    template = {
        "id": f"template-{uuid.uuid4().hex[:16]}",
        "experiment": {
            "id": f"exp-{uuid.uuid4().hex[:16]}",
            "versionId": 0,
            "metadata": {
                "name": f"[agent-agent] {config.get('name', 'simulation')}",
                "publicName": "Simulation",
                "description": description,
                "tags": ["convokitai-simulation"],
            },
            "permissions": {"visibility": "public", "readers": []},
            "defaultCohortConfig": cohort_config,
            "prolificConfig": {
                "enableProlificIntegration": False,
                "defaultRedirectCode": "",
                "attentionFailRedirectCode": "",
                "bootedRedirectCode": "",
            },
            "stageIds": STAGE_IDS,
            "cohortLockMap": {},
            "cohortDefinitions": [],
        },
        "stageConfigs": stages,
        "agentMediators": mediators,
        "agentParticipants": agents,
        "agentAssistants": list(assistants.values()),
    }
    return _exclude_none(template), cohort_config, agents


def _seed_gemini_api_key(backend: LocalBackend, gemini_api_key: str) -> None:
    """Store the Gemini API key in ``experimenterData/{email}`` in the Firestore emulator.

    This does what ConvoArena's ``scripts/seed-api-key.mjs`` does (which its
    ``run_locally.sh`` calls, but :class:`convokit.simulation.local_backend.LocalBackend` does not). The backend
    reads the key from there when generating agent messages, and without it silently does
    nothing: there is no error and no chat message, so the agents never speak.
    """
    url = (
        f"http://127.0.0.1:{backend.firestore_port}/v1/projects/"
        f"{backend.project_id}/databases/(default)/documents/experimenterData"
    )
    document = {
        "fields": {
            "id": {"stringValue": backend.experimenter_email},
            "email": {"stringValue": backend.experimenter_email},
            "apiKeys": {
                "mapValue": {
                    "fields": {
                        "geminiApiKey": {"stringValue": gemini_api_key},
                        "ollamaApiKey": {
                            "mapValue": {
                                "fields": {
                                    "url": {"stringValue": ""},
                                    "apiKey": {"stringValue": ""},
                                }
                            }
                        },
                    }
                }
            },
            "viewedExperiments": {"arrayValue": {"values": []}},
            "showAlphaFeatures": {"booleanValue": False},
        }
    }
    resp = requests.post(
        url,
        params={"documentId": backend.experimenter_email},
        headers={"Authorization": "Bearer owner", "Content-Type": "application/json"},
        json=document,
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"seeding Gemini API key failed: {resp.status_code} {resp.text}")


def _add_agent_to_cohort(
    backend: LocalBackend | FirebaseBackend,
    experiment_id: str,
    cohort_id: str,
    agent: dict,
) -> dict:
    """Add a participant created from an agent template to a cohort.

    The templates passed to ``client.create_simulation`` only register the agents' personas
    and prompts on the experiment; an agent joins a cohort only when ``createParticipant`` is
    called. ConvoArena requires ``agentConfig.modelSettings`` there (see its
    ``utils/src/participant.validation.ts``) even though the template already has
    ``defaultModelSettings``.
    """
    # both backends serve the REST API at <functions URL>/api/v1, and
    # createParticipant sits alongside it
    url = backend.base_url.removesuffix("/api/v1") + "/createParticipant"
    persona = agent["persona"]
    agent_config = {
        "agentId": persona["id"],
        "promptContext": "",
        "modelSettings": persona["defaultModelSettings"],
    }
    if persona.get("assistantId"):
        agent_config["assistantId"] = persona["assistantId"]
    resp = requests.post(
        url,
        json={
            "data": {
                "experimentId": experiment_id,
                "cohortId": cohort_id,
                "isAnonymous": True,
                "agentConfig": agent_config,
            }
        },
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"createParticipant failed: {resp.status_code} {resp.text}")
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"createParticipant error: {body['error']}")
    return body.get("result", body)


def _chat_message_count(export: dict) -> int:
    """Return how many non-system messages the experiment's chat has."""
    return sum(
        m.get("type") != "system"
        for cohort in export.get("cohortMap", {}).values()
        for m in cohort.get("chatMap", {}).get(CHAT_STAGE_ID, [])
    )


def _failed_model_calls(client: dl.Client, experiment_id: str) -> list[str]:
    """Return one line per model call of the experiment that didn't return OK.

    The backend doesn't retry a failed opening message, so one failure there leaves the chat
    silent."""
    try:
        logs = client.export_experiment_logs(experiment_id)
    except Exception as exc:  # the logs are only for diagnosis
        return [f"    (could not load model logs: {exc})"]
    lines = []
    for log in logs or []:
        response = log.get("response") or {}
        if response.get("status", "ok") == "ok":
            continue
        who = (log.get("userProfile") or {}).get("name") or log.get("publicId")
        what = log.get("description") or "chat message"
        error = response.get("errorMessage") or ""
        lines.append(
            f"    model call failed: {who} ({what}): {response.get('status')} {error}".rstrip()
        )
    return lines


def _stall_report(export: dict, failed_calls: list[str] = ()) -> str:
    """Describe where an unfinished experiment is stuck: each participant's status and
    stage, how far its chat got, and which model calls failed."""
    lines = []
    for participant in export.get("participantMap", {}).values():
        profile = participant.get("profile", {})
        lines.append(
            f"    {profile.get('name')} ({profile.get('publicId')}): "
            f"{profile.get('currentStatus')} in stage {profile.get('currentStageId')}"
        )
    for cohort in export.get("cohortMap", {}).values():
        chat_data = cohort.get("dataMap", {}).get(CHAT_STAGE_ID, {})
        started = chat_data.get("discussionStartTimestamp") is not None
        lines.append(
            f"    chat: {_chat_message_count(export)} message(s), "
            f"discussion {'started' if started else 'never started (so its time limit never runs)'}"
        )
    lines += failed_calls
    return "\n".join(lines)


def _wait_for_agents_in_chat(
    client: dl.Client, experiment_id: str, count: int, timeout: float = 15
) -> bool:
    """Poll until ``count`` participants of the experiment have reached the chat stage, or until
    ``timeout`` seconds have passed; return whether they did.

    The backend unlocks the chat when the last participant enters it, but only counts the
    others whose entry has already been saved. If two agents enter at the same moment, each
    sees the other as not ready and the chat never unlocks, so agents are added one at a time.
    """
    poll_client = dl.Client(base_url=client.base_url, api_key=client.api_key, timeout=10)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            participants = poll_client.export_experiment(experiment_id).get("participantMap", {})
            in_chat = sum(
                (p.get("profile") or {}).get("currentStageId") == CHAT_STAGE_ID
                for p in participants.values()
            )
            if in_chat >= count:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def _wait_for_first_message(client: dl.Client, experiment_id: str, timeout: float = 120) -> bool:
    """Poll until the experiment's chat has a message or ``timeout`` seconds have passed; return
    whether it has one.

    The backend sends each chat's opening messages once, when its agents enter the chat, and
    never retries them; if they fail (e.g. because many model calls at once hit the Gemini rate
    limit) the chat stays silent. :class:`_Stagger` spaces out conversation starts to keep
    those calls apart.
    """
    # exports get 3x the client timeout; keep each poll short so a slow backend can't stall this
    poll_client = dl.Client(base_url=client.base_url, api_key=client.api_key, timeout=10)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if _chat_message_count(poll_client.export_experiment(experiment_id)) > 0:
                return True
        except requests.RequestException:  # includes dl.APIError and timeouts
            pass
        time.sleep(5)
    return False


def _format_duration(seconds: float) -> str:
    """Format seconds as e.g. ``4m05s``."""
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s"


def _wait_for_exports(
    client: dl.Client, experiment_ids: list[str], timeout: float | None = None
) -> tuple[list[dict], set[str]]:
    """Poll until every experiment's participants have finished, or until ``timeout`` seconds
    have passed, then return all exports (as they stand) and the IDs that did not finish."""
    exports: dict[str, dict] = {}
    deadline = time.monotonic() + timeout if timeout is not None else None
    with tqdm(
        total=len(experiment_ids),
        desc="experiments completed",
        unit="exp",
        # tqdm only redraws on updates, so the postfix is reset every poll to keep {elapsed} current
        bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| waited {elapsed}{postfix}",
    ) as progress:
        while len(exports) < len(experiment_ids):
            if deadline is not None:
                left = _format_duration(max(0.0, deadline - time.monotonic()))
                progress.set_postfix_str(f"{left} left before timeout")
            else:
                progress.refresh()
            for experiment_id in experiment_ids:
                if experiment_id in exports:
                    continue
                try:
                    exports[experiment_id] = client.get_completed_experiment_data(experiment_id)
                    progress.update(1)
                except RuntimeError:
                    pass
            if len(exports) == len(experiment_ids):
                break
            if deadline is not None and time.monotonic() > deadline:
                break
            time.sleep(5)

    unfinished = {experiment_id for experiment_id in experiment_ids if experiment_id not in exports}
    for experiment_id in experiment_ids:
        if experiment_id in unfinished:
            exports[experiment_id] = client.export_experiment(experiment_id)
            print(
                f"WARNING: experiment {experiment_id} did not finish within {timeout:.0f}s; "
                f"keeping what it has so far:\n"
                f"{_stall_report(exports[experiment_id], _failed_model_calls(client, experiment_id))}"
            )
    return [exports[experiment_id] for experiment_id in experiment_ids], unfinished


def _heartbeat(log_path: Path, stop: threading.Event, interval: float = 5.0) -> None:
    """Print a progress line every ``interval`` seconds until ``stop`` is set."""
    start = time.monotonic()
    while not stop.wait(interval):
        elapsed = int(time.monotonic() - start)
        size = log_path.stat().st_size if log_path.exists() else 0
        print(f"  ... still waiting on emulators ({elapsed}s elapsed, log is {size} bytes)")


class _Stagger:
    """Spaces out calls to :meth:`wait` across threads so that consecutive calls return at least
    ``interval`` seconds apart."""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        time.sleep(start - now)


def _start_conversation(
    backend: LocalBackend | FirebaseBackend,
    template: dict,
    cohort_config: dict,
    agents: list[dict],
    pairing_id: str | None,
    topic: str,
    stagger: _Stagger,
) -> tuple[str, str]:
    """Create one conversation's experiment and cohort, add its agents, and wait for its first
    message. Runs in a worker thread of :func:`create_simulation`.

    :return: the experiment ID and cohort ID
    """
    stagger.wait()
    client = backend.client()  # requests sessions aren't thread-safe, so one client per thread
    result = client.create_simulation(
        template=template,
        num_cohorts=1,
        cohort_names=[f"[convokitai-sim] {pairing_id or 'pairing'}"],
        cohort_participant_config=[dl.CohortParticipantConfig(**cohort_config)],
    )
    experiment_id = result["experiment"]["experiment"]["id"]
    cohort = result["cohorts"][0]
    cohort_id = cohort["cohort"]["id"] if "cohort" in cohort else cohort["id"]
    # each conversation's lines are printed together, since several run at once
    lines = [
        f"created experiment {experiment_id}, cohort {cohort_id} (pairing {pairing_id}: {topic})"
    ]

    for joined, agent in enumerate(agents, start=1):
        _add_agent_to_cohort(backend, experiment_id, cohort_id, agent)
        lines.append(f"  {agent['persona']['id']} joined")
        if joined < len(agents) and not _wait_for_agents_in_chat(client, experiment_id, joined):
            lines.append(
                f"  WARNING: {agent['persona']['id']} hasn't reached the chat yet; adding the next agent anyway"
            )

    if _wait_for_first_message(client, experiment_id):
        lines.append("  conversation started")
    else:
        lines.append("  WARNING: no chat message yet; it may never start:")
        lines += _failed_model_calls(client, experiment_id)
    print("\n".join(lines))
    return experiment_id, cohort_id


def create_simulation(
    backend: LocalBackend | FirebaseBackend,
    sim_yaml: str | Path | dict,
    wait_timeout: float | None = None,
    max_concurrent: int = 4,
    start_interval: float = 5.0,
) -> Corpus:
    """Run every conversation described by the simulation YAML on a running backend and wait
    until they all finish.

    Each pairing and block combination is created as its own experiment with one cohort.
    Conversations are set up in parallel, up to ``max_concurrent`` at a time, with their starts
    spaced ``start_interval`` seconds apart so that their opening model calls don't all hit the
    Gemini rate limit at once. Within a conversation, the agents are added to the cohort one at
    a time. Progress and warnings are printed.

    Example::

        from convokit.simulation import FirebaseBackend, create_simulation

        backend = FirebaseBackend(project_id="my-project", api_key="dlb_live_...")
        corpus = create_simulation(backend, "simulation.yaml")

    :param backend: the backend to run on: a :class:`convokit.simulation.local_backend.LocalBackend` inside its
        ``with`` block, or a :class:`FirebaseBackend`
    :param sim_yaml: path to the simulation YAML, the YAML itself as a string, or the parsed dict
    :param wait_timeout: seconds to wait for the conversations to finish before keeping them as
        they stand. Defaults to ``max_time`` plus 5 minutes (or no limit if ``max_time`` isn't
        set).
    :param max_concurrent: how many conversations to set up at the same time. Set to 1 to set
        them up one after another.
    :param start_interval: minimum seconds between the starts of two conversations' setup.
        Raise it if conversations fail to start because of Gemini rate limits.
    :return: a ConvoKit Corpus with one Conversation per pairing and block combination
        that has chat messages. Each Conversation's meta has the pairing ID, experiment ID,
        blocks shown, and whether it completed; the corpus's
        ``ai_meta["simulation_config"]`` holds the parsed simulation YAML.
    :raises ValueError: if the simulation YAML is invalid, or no conversation has any chat
        messages
    :raises RuntimeError: if no conversation could be set up
    """
    config = load_simulation_config(sim_yaml)
    client = backend.client()
    if wait_timeout is None and config.get("max_time"):
        # time for the agents to join and for the last turns to wrap up
        wait_timeout = config["max_time"] * 60 + 300

    # build every template first, so an invalid YAML fails before any experiment is created
    jobs = [
        (*_experiment_template(config, pairing, blocks), pairing.get("id"), blocks)
        for pairing in config["pairings"]
        for blocks in _block_variants(config)
    ]
    print(
        f"setting up {len(jobs)} conversation(s), up to {max_concurrent} at a time "
        f"(each takes up to a few minutes to start)..."
    )
    stagger = _Stagger(start_interval)
    with ThreadPoolExecutor(max_workers=max(1, max_concurrent)) as pool:
        futures = [
            pool.submit(
                _start_conversation,
                backend,
                template,
                cohort_config,
                agents,
                pairing_id,
                "; ".join(block["description"] for block in blocks),
                stagger,
            )
            for template, cohort_config, agents, pairing_id, blocks in jobs
        ]
    runs = []
    for future, (_, _, _, pairing_id, blocks) in zip(futures, jobs):
        try:
            experiment_id, cohort_id = future.result()
        except Exception as exc:  # keep the conversations that did start
            topic = "; ".join(block["description"] for block in blocks)
            print(f"WARNING: setting up pairing {pairing_id} ({topic}) failed: {exc}")
            continue
        runs.append((experiment_id, cohort_id, pairing_id))
    if not runs:
        raise RuntimeError("no conversation could be set up; see the warnings above")

    print(f"{len(runs)} conversation(s) running, waiting for them to finish...")
    exports, unfinished = _wait_for_exports(client, [run[0] for run in runs], wait_timeout)
    corpus = export_to_corpus(exports)

    for experiment_id, cohort_id, pairing_id in runs:
        if corpus.has_conversation(cohort_id):
            convo = corpus.get_conversation(cohort_id)
            convo.meta["pairing_id"] = pairing_id
            convo.meta["completed"] = experiment_id not in unfinished
        else:
            print(
                f"WARNING: experiment {experiment_id} has no chat messages, so it is not in the corpus"
            )
    corpus.ai_meta = {"simulation_config": config}
    return corpus


def main(repo_root: str, sim_yaml: str, gemini_api_key: str | None = None) -> None:
    """Run a simulation on emulators started from a ConvoArena checkout and print the results.

    Starts the emulators of the checkout at ``repo_root`` with
    :class:`convokit.simulation.local_backend.LocalBackend` (printing progress, with the emulator log in the temp
    directory), stores ``gemini_api_key`` in the backend, runs :func:`create_simulation`,
    prints the corpus's summary statistics and each conversation's transcript, and stops the
    emulators. This is what running the module as a script does, with the key taken from the
    ``GEMINI_API_KEY`` environment variable.

    :param repo_root: path to a ConvoArena checkout, set up as described in
        :mod:`convokit.simulation.local_backend`
    :param sim_yaml: path to the simulation YAML, or the YAML itself as a string
    :param gemini_api_key: Gemini API key for the agents. Without one, the agents send no
        messages.
    """
    log_path = Path(tempfile.gettempdir()) / f"dl-run-conversation-{int(time.time())}.log"
    print("starting emulators (first boot can take a couple minutes) — tail progress with:")
    print(f"  tail -f {log_path}")

    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat, args=(log_path, stop_heartbeat), daemon=True)
    heartbeat.start()
    backend_cm = LocalBackend(repo_root, log_path=log_path)
    try:
        backend = backend_cm.__enter__()
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=1)

    try:
        print(f"backend up at {backend.base_url}")

        if gemini_api_key:
            _seed_gemini_api_key(backend, gemini_api_key)
            print("seeded Gemini API key into experimenterData")
        else:
            print(
                "WARNING: no GEMINI_API_KEY found — agents will join but "
                "every chat turn will silently fail with no messages sent"
            )

        corpus = create_simulation(backend, sim_yaml)

        print("done.")
        corpus.print_summary_stats()
        for convo in corpus.iter_conversations():
            print(f"\n=== {convo.id} (pairing {convo.meta.get('pairing_id')}) ===")
            print(convo.get_transcript(supports=True))
    finally:
        backend_cm.__exit__(*sys.exc_info())
        print("backend stopped")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(
            "usage: python -m convokit.simulation.create_simulation /path/to/convoarena simulation.yaml"
        )
    main(sys.argv[1], sys.argv[2], os.environ.get("GEMINI_API_KEY"))
