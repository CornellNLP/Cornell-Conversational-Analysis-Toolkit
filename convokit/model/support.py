"""
The Support class: a private message from a PrivateAssistant to the speakers it assists.
"""

from typing import Dict, Iterable, List, Optional, Union


class Support:
    """
    Represents a single message from a PrivateAssistant to the speaker(s) it assists, outside the
    main Conversation.

    The speakers that can see a Support are given by its PrivateAssistant's ``speakers``.

    :param id: the unique id of the support
    :param text: the text of the support message
    :param reply_to: id of the utterance the assisted speaker is replying to
    :param draft: the draft of the reply the assisted speaker has written so far (``""`` means the
        draft is empty)
    :param private_assistant_id: id of the PrivateAssistant that produced the support
    :param timestamp: the time the support was sent (an int, or a string)

    :ivar id: the unique id of the support
    :ivar text: the text of the support message
    :ivar reply_to: id of the utterance the assisted speaker is replying to
    :ivar draft: the assisted speaker's draft at the time of the support
    :ivar private_assistant_id: id of the PrivateAssistant that produced the support
    :ivar timestamp: the time the support was sent
    """

    def __init__(
        self,
        id: str,
        text: str,
        reply_to: Optional[str] = None,
        draft: str = "",
        private_assistant_id: Optional[str] = None,
        timestamp: Union[str, int, None] = None,
    ):
        self.id = id
        self.text = text
        self.reply_to = reply_to
        self.draft = draft if draft is not None else ""
        self.private_assistant_id = private_assistant_id
        self.timestamp = timestamp

    def to_dict(self) -> Dict:
        """
        Convert this support into a JSON-serializable dict.

        :return: dict with keys ``"id"``, ``"text"``, ``"reply_to"``, ``"draft"``,
            ``"private_assistant_id"``, and ``"timestamp"``
        """
        return {
            "id": self.id,
            "text": self.text,
            "reply_to": self.reply_to,
            "draft": self.draft,
            "private_assistant_id": self.private_assistant_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Optional["Support"]:
        """
        Build a Support from its dict form (as produced by :meth:`to_dict`). Keys outside the
        Support fields are ignored.

        :param data: the dict form of the support
        :return: the Support, or None if ``data`` is not a dict
        """
        if not isinstance(data, dict):
            return None
        return cls(
            id=data.get("id"),
            text=data.get("text", ""),
            reply_to=data.get("reply_to"),
            draft=data.get("draft", ""),
            # legacy key: older versions of the format used "assistant_id"
            private_assistant_id=data.get("private_assistant_id", data.get("assistant_id")),
            timestamp=data.get("timestamp"),
        )

    @staticmethod
    def normalize_list(value: Optional[Iterable]) -> List["Support"]:
        """
        Convert a list of Supports and/or dicts into a list of Supports. Entries that are neither
        are dropped.

        :param value: an iterable of Supports and/or their dict forms (or None)
        :return: a list of Supports
        """
        normalized = []
        for item in value or []:
            if isinstance(item, Support):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(Support.from_dict(item))
        return normalized

    def get_transcript(self, conversation, supports: bool = False) -> str:
        """
        Get a plain-text transcript of the given Conversation, from its beginning up to and
        including this support. See :meth:`Conversation.get_transcript
        <convokit.Conversation.get_transcript>`.

        :param conversation: the Conversation this support belongs to (Supports don't keep a
            reference to it)
        :param supports: whether to include the other Supports before this one (this support is
            always shown)
        :return: the transcript as a single string
        :raises ValueError: if this support is not in ``conversation``
        """
        return conversation.get_transcript(supports=supports, until=self)

    def __repr__(self):
        return "Support({})".format(self.to_dict())
