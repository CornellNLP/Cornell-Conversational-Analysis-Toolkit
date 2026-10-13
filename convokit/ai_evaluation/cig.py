"""
Conversational Information Gain (CIG): a 1-4 score for each utterance indicating how much it
added to what the group already knew (1 = nothing new, 4 = a new idea or reframing).

An LLM reads the whole conversation once and rates every utterance against what came before it.
Two kinds of utterances are left unrated (score ``None``):

- the opening exchange, up to and including the first utterance by which every speaker has
  spoken at least once: there is nothing before it to add to, so it is shown to the LLM as
  context only;
- fragments: 3 words or fewer, or 4-5 words without a final ``.``, ``!`` or ``?`` (e.g.
  "ok!", "yeah that makes sense").

Reference: CIG: Measuring Conversational Information Gain in Deliberative Dialogues with
Semantic Memory Dynamics, https://aclanthology.org/2026.acl-long.2203/
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from convokit import Transformer
from convokit.genai import GenAIConfigManager
from convokit.model import Conversation, Corpus, Speaker, Utterance

from .llm import get_eval_llm_client
from .util import (
    default_is_mediator,
    default_speaker_name,
    default_topic,
    display_names,
)

MEDIATOR_LABEL = "Mediator(mod)"


def is_fragment(text: str) -> bool:
    """
    Check whether an utterance is too short to rate: at most 3 words, or at most 5 words without
    terminal punctuation (``.``, ``!`` or ``?``).

    :param text: the utterance text
    :return: True if the utterance is a fragment
    """
    words = str(text).split()
    return len(words) <= 3 or (len(words) <= 5 and not str(text).rstrip().endswith((".", "!", "?")))


class CIG(Transformer):
    """
    ConvoKit Transformer that rates every utterance of a conversation for conversational
    information gain (CIG) with an LLM, and stores the rating in ``utt.meta["cig"]`` (``None``
    where not rated).

    Each utterance gets a score from 1 (adds nothing new) to 4 (a new idea or reframing). The
    opening exchange (up to and including the first utterance by which every speaker has
    spoken) is shown to the LLM as context only, and fragments (see ``is_fragment``) are not
    rated.

    Usage::

        cig = CIG(model="gpt-5")
        cig.transform(corpus)
        cig.summarize(corpus)   # mean score per conversation, mediator vs participants

    The corpus must identify the mediator and the name each speaker goes by in the chat. By
    default these come from the ConvoKit AI fields (``speaker.ai_meta["role"]`` and
    ``conversation.alias``); they can be replaced with ``is_mediator`` and ``name_func``.

    Reference: CIG: Measuring Conversational Information Gain in Deliberative Dialogues with
    Semantic Memory Dynamics, https://aclanthology.org/2026.acl-long.2203/

    :param llm_client: client with a ``generate_json(prompt)`` method; by default one for
        ``model`` built with :func:`get_eval_llm_client <convokit.ai_evaluation.llm.get_eval_llm_client>`
    :param model: model name for the default client: an OpenAI model (e.g. ``"gpt-5"``) or a
        Gemini model (e.g. ``"gemini-2.5-flash"``)
    :param provider: ``"gpt"`` or ``"gemini"`` for the default client; inferred from ``model``
        if not given
    :param config_manager: GenAIConfigManager holding the API key for the default client
    :param reasoning_effort: OpenAI reasoning effort for the default client (not used for Gemini)
    :param custom_prompt: prompt template replacing ``prompts/cig_prompt.txt``; must contain the
        ``{topic}``, ``{context}``, ``{target}``, ``{start}``, ``{end}`` and ``{total}`` placeholders
    :param cig_attribute_name: name of the utterance metadata attribute to store the rating in
    :param is_mediator: function from Speaker to whether it is the mediator (shown to the LLM as
        ``Mediator(mod)``)
    :param name_func: function from (Conversation, Speaker) to the name shown for a participant
    :param topic_func: function from Conversation to the topic line shown to the model
    :param n_workers: number of conversations to rate concurrently
    :param verbosity: print progress every ``verbosity`` conversations (0 to disable)
    """

    PROMPT_TEMPLATE = None

    @classmethod
    def _load_prompt(cls):
        if cls.PROMPT_TEMPLATE is None:
            path = os.path.join(os.path.dirname(__file__), "prompts", "cig_prompt.txt")
            with open(path, "r", encoding="utf-8") as f:
                cls.PROMPT_TEMPLATE = f.read()

    def __init__(
        self,
        llm_client=None,
        model: str = "gpt-5",
        provider: Optional[str] = None,
        config_manager: Optional[GenAIConfigManager] = None,
        reasoning_effort: str = "minimal",
        custom_prompt: Optional[str] = None,
        cig_attribute_name: str = "cig",
        is_mediator: Callable[[Speaker], bool] = default_is_mediator,
        name_func: Callable[[Conversation, Speaker], str] = default_speaker_name,
        topic_func: Callable[[Conversation], str] = default_topic,
        n_workers: int = 1,
        verbosity: int = 10,
    ):
        self._load_prompt()
        self.llm_client = llm_client or get_eval_llm_client(
            model,
            provider=provider,
            config_manager=config_manager,
            reasoning_effort=reasoning_effort,
        )
        self.prompt = custom_prompt or self.PROMPT_TEMPLATE
        self.cig_attribute_name = cig_attribute_name
        self.is_mediator = is_mediator
        self.name_func = name_func
        self.topic_func = topic_func
        self.n_workers = n_workers
        self.verbosity = verbosity

    def rating_targets(self, utts: List[Utterance]) -> Tuple[List[bool], List[bool]]:
        """
        Split a conversation's utterances into the opening context batch and the rating targets.

        :param utts: the conversation's utterances, in chronological order
        :return: two lists of booleans parallel to ``utts``: whether each utterance is in the
            opening context batch, and whether it is to be rated (both empty if there are no
            utterances)
        """
        if not utts:
            return [], []
        seen, n_seen = set(), []
        for utt in utts:
            seen.add(utt.speaker.id)
            n_seen.append(len(seen))
        cut = n_seen.index(len(seen))
        in_context = [i <= cut for i in range(len(utts))]
        is_target = [not c and not is_fragment(utt.text) for c, utt in zip(in_context, utts)]
        return in_context, is_target

    def build_prompt(self, conversation: Conversation) -> Tuple[Optional[str], List[int]]:
        """
        Build the rating prompt for a conversation.

        :param conversation: the Conversation to rate
        :return: the prompt (None if the conversation has nothing to rate) and the positions of
            the utterances to rate, in chronological order
        """
        utts = conversation.get_chronological_utterance_list()
        names = display_names(conversation, self.is_mediator, self.name_func)
        in_context, is_target = self.rating_targets(utts)

        context, target, targets, skipped = [], [], [], []
        for pos, utt in enumerate(utts):
            label = MEDIATOR_LABEL if self.is_mediator(utt.speaker) else names[utt.speaker.id]
            line = f"{pos}. {label}: {utt.text}"
            (context if in_context[pos] else target).append(line)
            if is_target[pos]:
                targets.append(pos)
            elif not in_context[pos]:
                skipped.append(pos)
        if not targets:
            return None, []

        prompt = self.prompt.format(
            topic=self.topic_func(conversation),
            context="\n\n".join(context),
            target="\n\n".join(target),
            start=min(targets),
            end=max(targets),
            total=len(targets),
        )
        if skipped:
            prompt += (
                "\nDo NOT rate these utterance indices (too short, context only): "
                f"{', '.join(map(str, skipped))}\n"
            )
        return prompt, targets

    def _rate(self, conversation: Conversation) -> Tuple[str, Dict[int, int]]:
        """Rate one conversation; returns its id and a dict of utterance position -> rating."""
        prompt, targets = self.build_prompt(conversation)
        if prompt is None:
            return conversation.id, {}
        ratings = self.llm_client.generate_json(prompt)
        return conversation.id, {
            int(r["utterance_index"]): int(r["informativeness"]) for r in ratings
        }

    def transform(self, corpus: Corpus, selector: Callable[[Conversation], bool] = lambda c: True):
        """
        Rate the utterances of the selected conversations and store each rating in
        ``utt.meta[cig_attribute_name]`` (``None`` for utterances that are not rated).

        Conversations whose rating fails are reported and left with ``None`` ratings.

        :param corpus: the Corpus to transform
        :param selector: function from Conversation to whether it should be rated
        :return: the Corpus, with ratings added
        """
        convos = [convo for convo in corpus.iter_conversations() if selector(convo)]
        results = {}
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = [ex.submit(self._rate, convo) for convo in convos]
            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    convo_id, ratings = fut.result()
                    results[convo_id] = ratings
                except Exception as exc:
                    print("failed:", exc)
                if self.verbosity and i % self.verbosity == 0:
                    print(i, "/", len(convos))

        for convo in convos:
            ratings = results.get(convo.id, {})
            for pos, utt in enumerate(convo.get_chronological_utterance_list()):
                utt.meta[self.cig_attribute_name] = ratings.get(pos)
        return corpus

    def summarize(
        self,
        corpus: Corpus,
        selector: Callable[[Conversation], bool] = lambda c: True,
        speaker_selector: Callable[[Speaker], bool] = lambda s: True,
        min_mediator_utts: int = 2,
    ) -> pd.DataFrame:
        """
        Compute the mean rating of the mediator's utterances and of the participants' utterances,
        per conversation.

        :param corpus: a Corpus that has been transformed
        :param selector: function from Conversation to whether it should be included
        :param speaker_selector: function from Speaker to whether its utterances count
        :param min_mediator_utts: skip conversations where the mediator spoke fewer times than
            this (0 to keep all; conversations without a mediator are kept)
        :return: DataFrame with one row per (conversation, is_mediator) and columns
            ``conversation_id``, ``is_mediator``, ``n_rated`` and ``cig_mean``
        """
        rows = []
        for convo in corpus.iter_conversations():
            if not selector(convo):
                continue
            utts = list(convo.iter_utterances())
            n_mediator = sum(self.is_mediator(utt.speaker) for utt in utts)
            if 0 < n_mediator < min_mediator_utts:
                continue
            ratings = {}
            for utt in utts:
                score = utt.meta.get(self.cig_attribute_name)
                if score is not None and speaker_selector(utt.speaker):
                    ratings.setdefault(self.is_mediator(utt.speaker), []).append(score)
            for is_mediator, scores in sorted(ratings.items()):
                rows.append(
                    {
                        "conversation_id": convo.id,
                        "is_mediator": is_mediator,
                        "n_rated": len(scores),
                        "cig_mean": sum(scores) / len(scores),
                    }
                )
        return pd.DataFrame(rows)
