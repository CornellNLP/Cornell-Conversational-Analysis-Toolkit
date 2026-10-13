"""
Mediator redirection: for each mediator turn, did it change what the participants said next?

The score comes from a language model's probabilities. Take a mediator turn T_k and the
participant reply that followed it. Ask the model how likely that reply was given T_k, and how
likely it was given a stand-in instead (the mediator's previous turn, or nothing). The
redirection score is the log-odds difference: positive means the reply was more expected after
T_k than after the stand-in, i.e. the turn steered the conversation; near zero means the reply
would have come anyway.

This is ``convokit.redirection`` (Nguyen et al., 2024) with context selectors for one mediator
and several participants instead of two alternating speakers::

    actual     [participant turn before T_k, T_k]      -> reply
    reference  [participant turn before T_k, T_(k-1)]  -> same reply   (counterfactual "prev")
               [participant turn before T_k]           -> same reply   (counterfactual "delete")

Each participant line is prefixed with the participant's name in the chat (e.g. "Goose: ..."),
the same name the mediator uses to address them; mediator lines are prefixed with
"mediator: ".

Reference: Taking a turn for the better: Conversation redirection throughout the course of
mental-health therapy, https://aclanthology.org/2024.findings-emnlp.555/
"""

from typing import Callable, Dict, List, Tuple

import pandas as pd

from convokit import Transformer
from convokit.model import Conversation, Corpus, Speaker
from convokit.redirection.redirection import Redirection
from .util import default_is_mediator, default_speaker_name

MEDIATOR_LABEL = "mediator"
ROLE_ATTRIBUTE_NAME = "role"  # the key convokit.redirection.preprocessing reads
COUNTERFACTUALS = ("prev", "delete")


def mediator_speaker_prefixes(roles: List[str]) -> Dict[str, str]:
    """
    Build speaker prefixes that use each role name itself (e.g. "Goose: "), so that a
    participant's prefix matches the name the mediator addresses them by.

    :param roles: the role names in a conversation
    :return: dict of role name -> prefix
    """
    return {role: role + ": " for role in roles}


def mediator_previous_context_selector(
    convo: Conversation,
    counterfactual: str = "prev",
    role_attribute_name: str = "role",
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute the actual and reference contexts of the mediator turns, for use as the
    ``previous_context_selector`` of ``convokit.redirection.Redirection``.

    A context is computed for every mediator turn that has a participant turn before it and,
    with the ``"prev"`` counterfactual, an earlier mediator turn to stand in for it. Mediator
    turns directly following another mediator turn are skipped.

    :param convo: Conversation whose utterances carry ``meta[role_attribute_name]``, with the
        value ``"mediator"`` for the mediator
    :param counterfactual: ``"prev"`` replaces the mediator turn with the mediator's previous
        turn in the reference context; ``"delete"`` removes it
    :param role_attribute_name: utterance metadata attribute holding the role
    :return: dicts of utterance id -> actual context and utterance id -> reference context
    :raises ValueError: if ``counterfactual`` is not ``"prev"`` or ``"delete"``
    """
    if counterfactual not in COUNTERFACTUALS:
        raise ValueError(f"counterfactual must be one of {COUNTERFACTUALS}, got {counterfactual!r}")
    actual_contexts, reference_contexts = {}, {}
    utts = list(convo.iter_utterances())
    roles = sorted({utt.meta[role_attribute_name] for utt in utts})
    role_to_prefix = mediator_speaker_prefixes(roles)

    prev_participant = None
    prev_mediator = None
    for i, utt in enumerate(utts):
        cur_role = utt.meta[role_attribute_name]
        if cur_role != MEDIATOR_LABEL:
            prev_participant = utt
            continue

        adjacent = i > 0 and utts[i - 1].meta[role_attribute_name] == MEDIATOR_LABEL
        # only the "prev" counterfactual needs an earlier mediator turn to stand in for this one
        has_reference = counterfactual == "delete" or prev_mediator is not None
        if prev_participant is not None and has_reference and not adjacent:
            prev_data = role_to_prefix[prev_participant.meta[role_attribute_name]]
            prev_data += prev_participant.text
            cur_data = role_to_prefix[MEDIATOR_LABEL] + utt.text

            actual_contexts[utt.id] = [prev_data, cur_data]
            if counterfactual == "prev":
                ref_data = role_to_prefix[MEDIATOR_LABEL] + prev_mediator.text
                reference_contexts[utt.id] = [prev_data, ref_data]
            else:
                reference_contexts[utt.id] = [prev_data]

        prev_mediator = utt

    return actual_contexts, reference_contexts


def mediator_future_context_selector(
    convo: Conversation, role_attribute_name: str = "role"
) -> Dict[str, List[str]]:
    """
    Compute the future context of every mediator turn: the participant turn right after it, for
    use as the ``future_context_selector`` of ``convokit.redirection.Redirection``.

    Mediator turns followed by another mediator turn, by an empty utterance, or by nothing get
    no future context.

    :param convo: Conversation whose utterances carry ``meta[role_attribute_name]``
    :param role_attribute_name: utterance metadata attribute holding the role
    :return: dict of utterance id -> future context
    """
    future_contexts = {}
    utts = list(convo.iter_utterances())
    roles = sorted({utt.meta[role_attribute_name] for utt in utts})
    role_to_prefix = mediator_speaker_prefixes(roles)

    for i, utt in enumerate(utts[:-1]):
        if utt.meta[role_attribute_name] != MEDIATOR_LABEL:
            continue
        nxt = utts[i + 1]
        if nxt.meta[role_attribute_name] == MEDIATOR_LABEL or not nxt.text.strip():
            continue
        future_contexts[utt.id] = [role_to_prefix[nxt.meta[role_attribute_name]] + nxt.text]

    return future_contexts


class MediatorRedirection(Transformer):
    """
    ConvoKit Transformer that scores each mediator turn by how much it redirected the
    participant reply that followed it, and stores the score in ``utt.meta["redirection"]``.

    The score compares how likely a language model finds the participant reply after the
    mediator turn versus after a counterfactual context (the mediator's previous turn, or no
    mediator turn), as a log-odds difference: positive values mean the turn made the reply more
    expected. A mediator turn is scored only if it is followed by a participant reply, has a
    participant turn before it, does not directly follow another mediator turn, and (with the
    ``"prev"`` counterfactual) has an earlier mediator turn to compare against (see
    ``mediator_previous_context_selector``).

    Usage::

        likelihood_model = GemmaLikelihoodModel(...)          # from convokit.redirection
        redirection = MediatorRedirection(likelihood_model)
        redirection.transform(corpus)
        redirection.summarize(corpus)   # mean score per conversation

    It also writes ``utt.meta["role"]`` (``"mediator"``, or the participant's name), the
    attribute ``convokit.redirection`` reads. ``fit`` fine-tunes the likelihood model (with
    ``convokit.redirection``'s own "Speaker A/B" formatting); scoring works without it.

    The corpus must identify the mediator and the name each speaker goes by in the chat. By
    default these come from the ConvoKit AI fields (``speaker.ai_meta["role"]`` and
    ``conversation.alias``); they can be replaced with ``is_mediator`` and ``name_func``.

    Reference: Taking a turn for the better: Conversation redirection throughout the course of
    mental-health therapy, https://aclanthology.org/2024.findings-emnlp.555/

    :param likelihood_model: a ``convokit.redirection`` LikelihoodModel (e.g.
        GemmaLikelihoodModel); scores can be computed with the off-the-shelf model, and ``fit``
        fine-tunes it
    :param counterfactual: ``"prev"`` or ``"delete"`` (see ``mediator_previous_context_selector``)
    :param is_mediator: function from Speaker to whether it is the mediator
    :param name_func: function from (Conversation, Speaker) to a participant's speaker prefix
    :param redirection_attribute_name: name of the utterance metadata attribute to store scores in
    """

    def __init__(
        self,
        likelihood_model,
        counterfactual: str = "prev",
        is_mediator: Callable[[Speaker], bool] = default_is_mediator,
        name_func: Callable[[Conversation, Speaker], str] = default_speaker_name,
        redirection_attribute_name: str = "redirection",
    ):
        self.is_mediator = is_mediator
        self.name_func = name_func
        self.redirection_attribute_name = redirection_attribute_name
        self.redirection = Redirection(
            likelihood_model=likelihood_model,
            previous_context_selector=lambda convo: mediator_previous_context_selector(
                convo,
                counterfactual=counterfactual,
            ),
            future_context_selector=mediator_future_context_selector,
            redirection_attribute_name=redirection_attribute_name,
        )

    def label_roles(self, corpus: Corpus, selector: Callable[[Conversation], bool]):
        """
        Write each utterance's role into ``utt.meta["role"]`` in the selected conversations:
        ``"mediator"`` for the mediator, otherwise the participant's name from ``name_func``
        (with " (participant)" appended if that name is itself ``"mediator"``).

        :param corpus: the Corpus to label
        :param selector: function from Conversation to whether it should be labeled
        """
        for convo in corpus.iter_conversations():
            if not selector(convo):
                continue
            for utt in convo.iter_utterances():
                if self.is_mediator(utt.speaker):
                    role = MEDIATOR_LABEL
                else:
                    # roles tell the mediator apart by its label, so a participant can't share it
                    role = self.name_func(convo, utt.speaker)
                    if role == MEDIATOR_LABEL:
                        role += " (participant)"
                utt.meta[ROLE_ATTRIBUTE_NAME] = role

    def fit(
        self,
        corpus: Corpus,
        y=None,
        train_selector: Callable[[Conversation], bool] = lambda convo: True,
        val_selector: Callable[[Conversation], bool] = lambda convo: True,
    ):
        """
        Fine-tune the likelihood model on the selected conversations (see
        ``convokit.redirection.Redirection.fit``).

        :param corpus: the Corpus to train on
        :param y: unused; accepted for compatibility with the Transformer interface
        :param train_selector: function from Conversation to whether it is used for training
        :param val_selector: function from Conversation to whether it is used for validation
        :return: this MediatorRedirection
        """
        self.label_roles(corpus, train_selector)
        self.label_roles(corpus, val_selector)
        self.redirection.fit(corpus, train_selector=train_selector, val_selector=val_selector)
        return self

    def transform(
        self,
        corpus: Corpus,
        selector: Callable[[Conversation], bool] = lambda convo: True,
        verbosity: int = 5,
    ):
        """
        Label roles and store the redirection score of each scorable mediator turn in
        ``utt.meta[redirection_attribute_name]``.

        :param corpus: the Corpus to transform
        :param selector: function from Conversation to whether it should be scored
        :param verbosity: print progress every ``verbosity`` conversations
        :return: the Corpus, with scores added
        """
        self.label_roles(corpus, selector)
        self.redirection.transform(corpus, selector=selector, verbosity=verbosity)
        return corpus

    def summarize(
        self,
        corpus: Corpus,
        selector: Callable[[Conversation], bool] = lambda convo: True,
    ) -> pd.DataFrame:
        """
        Compute the mean redirection score per conversation.

        :param corpus: a Corpus that has been transformed
        :param selector: function from Conversation to whether it should be included
        :return: DataFrame with one row per conversation that has at least one score, and
            columns ``conversation_id``, ``n_scored`` and ``redirection_mean``
        """
        rows = []
        for convo in corpus.iter_conversations():
            if not selector(convo):
                continue
            scores = [
                utt.meta[self.redirection_attribute_name]
                for utt in convo.iter_utterances()
                if utt.meta.get(self.redirection_attribute_name) is not None
            ]
            if scores:
                rows.append(
                    {
                        "conversation_id": convo.id,
                        "n_scored": len(scores),
                        "redirection_mean": sum(scores) / len(scores),
                    }
                )
        return pd.DataFrame(rows)
