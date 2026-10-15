from typing import Dict, List, Optional

from convokit.util import warn
from .aiUtil import split_ai_fields
from .corpusComponent import CorpusComponent
from .speaker import Speaker
from .support import Support


class Utterance(CorpusComponent):
    """Represents a single utterance in the dataset.

    :param id: the unique id of the utterance.
    :param speaker: the speaker giving the utterance.
    :param conversation_id: the id of the root utterance of the conversation.
    :param reply_to: id of the utterance this was a reply to.
    :param timestamp: timestamp of the utterance. Can be any
        comparable type.
    :param text: text of the utterance.
    :param ai_meta: additional outputs from the speaker's LLM call for this message (empty if there
        are none or the speaker is not an AI; see ``AI_FORMAT.md``). Recognized keys:

        - ``"config"``: the generation config used to generate the message
        - ``"supports"``: list of Supports attached to this utterance

        An ``"ai_meta"`` entry in ``meta`` (as in a dumped corpus) is moved into this attribute,
        unless this argument is given.

    :ivar id: the unique id of the utterance.
    :ivar speaker: the speaker giving the utterance.
    :ivar conversation_id: the id of the root utterance of the conversation.
    :ivar reply_to: id of the utterance this was a reply to.
    :ivar timestamp: timestamp of the utterance.
    :ivar text: text of the utterance.
    :ivar meta: A dictionary-like view object providing read-write access to
        utterance-level metadata.
    :ivar ai_meta: AI metadata of the utterance. Unlike the other classes, assigning replaces the
        existing ``ai_meta`` instead of merging into it.
    :ivar supports: the Supports attached to this utterance
    """

    AI_FIELDS = ("ai_meta",)

    def __init__(
        self,
        owner=None,
        id: Optional[str] = None,
        speaker: Optional[Speaker] = None,
        conversation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        timestamp: Optional[int] = None,
        text: str = "",
        meta: Optional[Dict] = None,
        ai_meta: Optional[Dict] = None,
    ):
        # check arguments that have alternate naming due to backwards compatibility
        if speaker is None:
            raise ValueError("No Speaker found: Utterance must be initialized with a Speaker.")

        if conversation_id is not None and not isinstance(conversation_id, str):
            warn(
                "Utterance conversation_id must be a string: conversation_id of utterance with ID: {} "
                "has been casted to a string.".format(id)
            )
            conversation_id = str(conversation_id)
        if not isinstance(text, str):
            warn(
                "Utterance text must be a string: text of utterance with ID: {} "
                "has been casted to a string.".format(id)
            )
            text = "" if text is None else str(text)

        props = {
            "speaker_id": speaker.id,
            "conversation_id": conversation_id,
            "reply_to": reply_to,
            "timestamp": timestamp,
            "text": text,
        }
        meta, ai_fields = split_ai_fields(meta, "utterance")
        super().__init__(obj_type="utterance", owner=owner, id=id, initial_data=props, meta=meta)
        self.speaker_ = speaker
        if ai_meta is not None or "ai_meta" in ai_fields:
            self.ai_meta = ai_meta if ai_meta is not None else ai_fields["ai_meta"]

    ############################################################################
    ## directly-accessible class properties (roughly equivalent to keys in the
    ## JSON, plus aliases for compatibility)
    ############################################################################

    def _get_speaker(self):
        return self.speaker_

    def _set_speaker(self, val):
        self.speaker_ = val
        self.set_data("speaker_id", self.speaker.id)

    speaker = property(_get_speaker, _set_speaker)

    def _get_conversation_id(self):
        return self.get_data("conversation_id")

    def _set_conversation_id(self, val):
        self.set_data("conversation_id", val)

    conversation_id = property(_get_conversation_id, _set_conversation_id)

    def _get_reply_to(self):
        return self.get_data("reply_to")

    def _set_reply_to(self, val):
        self.set_data("reply_to", val)

    reply_to = property(_get_reply_to, _set_reply_to)

    def _get_timestamp(self):
        return self.get_data("timestamp")

    def _set_timestamp(self, val):
        self.set_data("timestamp", val)

    timestamp = property(_get_timestamp, _set_timestamp)

    def _get_text(self):
        return self.get_data("text")

    def _set_text(self, val):
        self.set_data("text", val)

    text = property(_get_text, _set_text)

    @property
    def ai_meta(self) -> Dict:
        """The AI metadata of this utterance."""
        if "_ai_meta" not in self.__dict__:
            self._ai_meta = {}
        return self._ai_meta

    @ai_meta.setter
    def ai_meta(self, value):
        value = dict(value or {})
        if "supports" in value:
            value["supports"] = Support.normalize_list(value["supports"])
        self._ai_meta = value
        self._persist_ai_field("ai_meta")

    @property
    def supports(self) -> List[Support]:
        """The Supports attached to this utterance (a new list; assign to modify)."""
        return Support.normalize_list(self.__dict__.get("_ai_meta", {}).get("supports", []))

    @supports.setter
    def supports(self, value):
        self.ai_meta = {**self.ai_meta, "supports": value}

    ############################################################################
    ## end properties
    ############################################################################

    def _apply_ai_fields(self, stored: Dict, legacy: bool = False) -> None:
        """
        Set the AI attributes from the AI fields stored in the metadata (the on-disk format).

        :param stored: the utterance's metadata entries for the AI keys that it has
        :param legacy: unused; accepted for consistency with Speaker
        """
        if "ai_meta" in stored:
            self.ai_meta = stored["ai_meta"]

    def get_transcript(self, supports: bool = False) -> str:
        """
        Get a plain-text transcript of this utterance's Conversation, from its beginning up to and
        including this utterance. See :meth:`Conversation.get_transcript
        <convokit.Conversation.get_transcript>`.

        :param supports: whether to include Supports (Supports replying to this utterance come after
            it, so they are not included)
        :return: the transcript as a single string
        :raises ValueError: if the utterance is not part of a Corpus
        """
        if self.owner is None:
            raise ValueError("Utterance {!r} is not part of a Corpus".format(self.id))
        return self.get_conversation().get_transcript(supports=supports, until=self)

    def get_conversation(self):
        """
        Get the Conversation (identified by Utterance.conversation_id) this Utterance belongs to

        :return: a Conversation object
        """
        return self.owner.get_conversation(self.conversation_id)

    def get_speaker(self):
        """
        Get the Speaker that made this Utterance.

        :return: a Speaker object
        """

        return self.speaker

    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "reply_to": self.reply_to,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "text": self.text,
            "vectors": self.vectors,
            "meta": self.meta if type(self.meta) == dict else self.meta.to_dict(),
        }

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Utterance):
            return False
        try:
            return (
                self.id == other.id
                and (
                    self.conversation_id is None
                    or other.conversation_id is None
                    or self.conversation_id == other.conversation_id
                )
                and self.reply_to == other.reply_to
                and self.speaker == other.speaker
                and self.timestamp == other.timestamp
                and self.text == other.text
            )
        except AttributeError:  # for backwards compatibility with wikiconv
            return self.__dict__ == other.__dict__

    def __str__(self):
        return (
            "Utterance(id: {}, conversation_id: {}, reply-to: {}, "
            "speaker: {}, timestamp: {}, text: {}, vectors: {}, meta: {})".format(
                repr(self.id),
                self.conversation_id,
                self.reply_to,
                self.speaker,
                self.timestamp,
                repr(self.text),
                self.vectors,
                self.meta,
            )
        )
