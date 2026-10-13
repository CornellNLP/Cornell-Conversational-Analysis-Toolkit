"""
Helpers for generating Utterances (for AI Speakers) and Supports (for PrivateAssistants) with
``convokit.genai``.

A generation config is a dict with the keys:

- ``"prompt"``: the prompt that generates the message (required)
- ``"model"``: the model name, e.g. ``"gemini-2.5-flash"`` or ``"gpt-4o-mini"`` (required unless a
  client is given)
- ``"provider"``: the ``convokit.genai`` provider, ``"gemini"`` / ``"gpt"`` / ``"local"`` (inferred
  from ``"model"`` if missing)
- ``"temperature"``: sampling temperature (optional; the client's default is used if missing)

Generation requires the GenAI extras: ``pip install "convokit[genai]"``.
"""

import time
import uuid
from typing import Dict, Optional

from .aiUtil import transcript_sort_key


def infer_provider(model: str) -> str:
    """
    Infer the ``convokit.genai`` provider from a model name.

    :param model: the model name
    :return: ``"gemini"`` for Gemini models, ``"gpt"`` for OpenAI models (``gpt*``, ``o1*``,
        ``o3*``, ``o4*``)
    :raises ValueError: if the provider cannot be inferred
    """
    model = model.lower()
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return "gpt"
    raise ValueError(
        "Cannot infer the provider for model {!r}; set config['provider'] "
        "to one of 'gemini', 'gpt', 'local'.".format(model)
    )


def get_client(config: Dict, client=None, config_manager=None):
    """
    Get the ``convokit.genai`` LLMClient to generate with.

    :param config: generation config; ``config["model"]`` and ``config["provider"]`` are used to
        build the client
    :param client: an LLMClient; if given, it is returned as is
    :param config_manager: GenAIConfigManager used to build the client (defaults to a new
        ``GenAIConfigManager()``)
    :return: the LLMClient
    :raises ImportError: if the ``convokit.genai`` dependencies are not installed
    :raises ValueError: if no client is given and ``config`` has no ``"model"``
    """
    if client is not None:
        return client
    try:
        from convokit.genai import GenAIConfigManager, get_llm_client
    except ImportError as e:
        raise ImportError(
            "GenAI dependencies not available. Please install via `pip install convokit[genai]`."
        ) from e
    model = config.get("model")
    if not model:
        raise ValueError("config must have a 'model' to generate with, or pass a client.")
    provider = config.get("provider") or infer_provider(model)
    return get_llm_client(provider, config_manager or GenAIConfigManager(), model=model)


def call_llm(client, config: Dict, prompt: str) -> str:
    """
    Generate a response to ``prompt`` with ``client``, using ``config["temperature"]`` if set.

    :param client: the LLMClient
    :param config: generation config
    :param prompt: the full prompt
    :return: the generated text, with surrounding whitespace stripped
    """
    kwargs = {"temperature": config["temperature"]} if "temperature" in config else {}
    return client.generate(prompt, **kwargs).text.strip()


def build_prompt(prompt: str, transcript: Optional[str], instruction: str) -> str:
    """
    Join the config prompt, the conversation transcript (if any), and the final instruction.

    :param prompt: the prompt from the generation config
    :param transcript: the conversation transcript, or None to omit it
    :param instruction: the instruction appended at the end
    :return: the full prompt
    """
    parts = [prompt]
    if transcript is not None:
        parts.append("Conversation so far:\n" + transcript)
    parts.append(instruction)
    return "\n\n".join(parts)


def last_utterance_id(conversation) -> Optional[str]:
    """
    Get the id of the latest Utterance in the conversation (by timestamp, then id).

    :param conversation: the Conversation
    :return: the utterance id, or None if the conversation has no utterances
    """
    utterances = list(conversation.iter_utterances())
    return max(utterances, key=transcript_sort_key).id if utterances else None


def last_support(conversation, reply_to: str):
    """
    Get the latest Support in the conversation replying to the utterance ``reply_to``.

    :param conversation: the Conversation
    :param reply_to: id of the utterance
    :return: the Support, or None if there is none
    """
    supports = [s for s in conversation.supports if s.reply_to == reply_to]
    return max(supports, key=transcript_sort_key) if supports else None


def next_timestamp(conversation) -> int:
    """
    Get a timestamp that orders after every utterance and support in the conversation.

    :param conversation: the Conversation, or None
    :return: one more than the largest numeric timestamp in the conversation, or the current Unix
        time if there is none
    """
    if conversation is not None:
        timestamps = [u.timestamp for u in conversation.iter_utterances()]
        timestamps += [s.timestamp for s in conversation.supports]
        numeric = [t for t in timestamps if isinstance(t, (int, float))]
        if numeric:
            return int(max(numeric)) + 1
    return int(time.time())


def new_id() -> str:
    """
    Generate a random id for a new Utterance or Support.

    :return: a random hex string
    """
    return uuid.uuid4().hex
