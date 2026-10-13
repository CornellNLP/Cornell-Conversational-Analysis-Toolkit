"""
Helpers shared by the evaluation Transformers.

They read the ConvoKit AI data format (see convokit/model/AI_FORMAT.md): the mediator is the
speaker whose ``ai_meta["role"]`` is ``"public assistant"``, and ``conversation.alias`` maps
speaker ids to the names used in the chat.
"""

from typing import Callable, Dict, Optional

from convokit.model import Conversation, Speaker

MEDIATOR_ROLE = "public assistant"


def default_is_mediator(speaker: Speaker) -> bool:
    """
    Check whether a speaker is the mediator, i.e. its ConvoKit AI role
    (``speaker.ai_meta["role"]``) is ``"public assistant"``.

    :param speaker: the Speaker
    :return: True if the speaker is the mediator
    """
    return getattr(speaker, "ai_meta", {}).get("role") == MEDIATOR_ROLE


def default_speaker_name(conversation: Conversation, speaker: Speaker) -> str:
    """
    Get the name a speaker goes by in a conversation, from ``conversation.alias``.

    :param conversation: the Conversation
    :param speaker: the Speaker
    :return: the speaker's alias in the conversation, or its id if it has none
    """
    return getattr(conversation, "alias", {}).get(speaker.id, speaker.id)


def default_topic(conversation: Conversation) -> str:
    """
    Get the topic line shown to the LLM: ``conversation.meta["topic"]``, followed by the debate
    statement ``conversation.meta["statement"]`` if there is one.

    :param conversation: the Conversation
    :return: the topic line (an empty string if neither field is set)
    """
    topic = conversation.meta.get("topic") or ""
    statement = conversation.meta.get("statement")
    return f"{topic} — debate statement: {statement}" if statement else topic


def pre_survey_text(conversation: Conversation, speaker: Speaker) -> Optional[str]:
    """
    Get the reason a participant gave for their stance before the conversation.

    :param conversation: the Conversation; its ``meta["pre_survey"]`` maps speaker ids to text
    :param speaker: the Speaker
    :return: the speaker's pre-survey text, or None if there is none
    """
    return (conversation.meta.get("pre_survey") or {}).get(speaker.id)


def display_names(
    conversation: Conversation,
    is_mediator: Callable[[Speaker], bool],
    name_func: Callable[[Conversation, Speaker], str],
) -> Dict[str, str]:
    """
    Map each non-mediator speaker id in a conversation to its name, in order of first utterance.

    :param conversation: the Conversation
    :param is_mediator: function from Speaker to whether it is the mediator
    :param name_func: function from (Conversation, Speaker) to the speaker's name
    :return: dict of speaker id -> name
    """
    names = {}
    for utt in conversation.get_chronological_utterance_list():
        if utt.speaker.id not in names and not is_mediator(utt.speaker):
            names[utt.speaker.id] = name_func(conversation, utt.speaker)
    return names
