from functools import total_ordering
from typing import Dict, List, Optional

from .aiUtil import as_dict, split_ai_fields
from .corpusComponent import CorpusComponent
from .corpusUtil import *


@total_ordering
class Speaker(CorpusComponent):
    """
    Represents a single speaker in a dataset, which may be a human or an AI agent.

    :param id: id of the speaker.
    :type id: str
    :param utts: dictionary of utterances by the speaker, where key is utterance id
    :param convos: dictionary of conversations started by the speaker, where key is conversation id
    :param meta: arbitrary dictionary of attributes associated
        with the speaker. ``"is_ai"`` / ``"ai_meta"`` entries (as in a dumped corpus) are moved
        out of ``meta`` into the corresponding attributes, unless those arguments are given.
    :type meta: dict
    :param is_ai: whether this speaker is known to be an AI agent (default False)
    :param ai_meta: AI metadata of the speaker (see ``AI_FORMAT.md``). Recognized keys:

        - ``"config"``: generation config for this speaker's messages, used by :meth:`generate`
          (see :mod:`convokit.model.aiGeneration`)
        - ``"role"``: the speaker's role in the conversation, e.g. ``"participant"`` or
          ``"public assistant"``

    :ivar id: id of the speaker.
    :ivar meta: A dictionary-like view object providing read-write access to
        speaker-level metadata.
    :ivar is_ai: whether this speaker is known to be an AI agent
    :ivar ai_meta: AI metadata of the speaker. Assigning a dict merges it into the existing
        ``ai_meta``.
    """

    AI_FIELDS = ("is_ai", "ai_meta")

    def __init__(
        self,
        owner=None,
        id: str = None,
        utts=None,
        convos=None,
        meta: Optional[Dict] = None,
        is_ai: Optional[bool] = None,
        ai_meta: Optional[Dict] = None,
    ):
        meta, ai_fields = split_ai_fields(meta, "speaker")
        super().__init__(obj_type="speaker", owner=owner, id=id, meta=meta)
        self.utterances = utts if utts is not None else dict()
        self.conversations = convos if convos is not None else dict()
        if is_ai is not None or "is_ai" in ai_fields:
            self.is_ai = is_ai if is_ai is not None else ai_fields["is_ai"]
        if ai_meta is not None or "ai_meta" in ai_fields:
            self.ai_meta = ai_meta if ai_meta is not None else ai_fields["ai_meta"]
        # self._split_attribs = set()
        # self._update_uid()

    # def identify_by_attribs(self, attribs: Collection) -> None:
    #     """Identify a speaker by a list of attributes. Sets which speaker info
    #     attributes should distinguish speakers of the same name in equality tests.
    #     For example, in the Supreme Court dataset, speakers are labeled with the
    #     current case id. Call this method with attribs = ["case"] to count
    #     the same person across different cases as different speakers.
    #
    #     By default, if this function is not called, speakers are identified by name only.
    #
    #     :param attribs: Collection of attribute names.
    #     :type attribs: Collection
    #     """
    #
    #     self._split_attribs = set(attribs)
    #     # self._update_uid()

    ############################################################################
    ## AI fields
    ############################################################################

    @property
    def is_ai(self) -> bool:
        """Whether this speaker is known to be an AI agent."""
        return self.__dict__.get("_is_ai", False)

    @is_ai.setter
    def is_ai(self, value):
        self._is_ai = bool(value)
        self._persist_ai_field("is_ai")

    @property
    def ai_meta(self) -> Dict:
        """The AI metadata of this speaker (a copy; assign to modify)."""
        return as_dict(self.__dict__.get("_ai_meta"))

    @ai_meta.setter
    def ai_meta(self, value):
        if isinstance(value, dict):
            self._ai_meta = {**as_dict(self.__dict__.get("_ai_meta")), **value}
        else:
            self._ai_meta = {}
        self._persist_ai_field("ai_meta")

    def _apply_ai_fields(self, stored: Dict, legacy: bool = False) -> None:
        """
        Set the AI attributes from the AI fields stored in the metadata (the on-disk format).

        :param stored: the speaker's metadata entries for the keys in ``AI_META_KEYS`` (and, if
            ``legacy`` is True, ``LEGACY_AI_KEYS``) that it has
        :param legacy: whether to also move a ``"role"`` from older versions of the format into
            ``ai_meta["role"]``
        """
        if "is_ai" in stored:
            self.is_ai = stored["is_ai"]
        if "ai_meta" in stored:
            self.ai_meta = as_dict(stored["ai_meta"])
        if legacy and stored.get("role") is not None and "role" not in self.ai_meta:
            self.ai_meta = {"role": stored["role"]}

    def generate(
        self,
        conversation=None,
        reply_to: Optional[str] = None,
        append: bool = False,
        client=None,
        config_manager=None,
        id: Optional[str] = None,
        timestamp: Optional[int] = None,
    ):
        """
        Generate an Utterance for this speaker with ``convokit.genai``, using the prompt in
        ``ai_meta["config"]`` (see :mod:`convokit.model.aiGeneration` for the recognized config
        keys). Requires ``is_ai`` to be True.

        Example::

            speaker = corpus.get_speaker("bot")
            utt = speaker.generate(conversation, append=True)

        :param conversation: if given, the conversation transcript (up to ``reply_to``) is included
            in the prompt
        :param reply_to: id of the utterance to reply to (defaults to the last utterance of
            ``conversation``)
        :param append: whether to add the generated utterance to ``conversation`` (and its Corpus)
        :param client: a ``convokit.genai`` LLMClient to use instead of building one from the config
        :param config_manager: GenAIConfigManager used to build the client (defaults to one reading
            ``~/.convokit/config.yml``)
        :param id: id of the generated utterance (random if not given)
        :param timestamp: timestamp of the generated utterance (defaults to after the
            conversation's last message)
        :return: the generated Utterance (if ``append`` is True, the Utterance as stored in the
            Corpus)
        :raises ValueError: if the speaker is not an AI speaker, has no ``"prompt"`` in
            ``ai_meta["config"]``, or if ``append`` is True and no ``conversation`` is given
        """
        from . import aiGeneration as generation
        from .utterance import Utterance

        config = self.ai_meta.get("config") or {}
        if not self.is_ai:
            raise ValueError("Speaker {!r} is not an AI speaker (is_ai is False)".format(self.id))
        if not config.get("prompt"):
            raise ValueError("Speaker {!r} has no prompt in ai_meta['config']".format(self.id))
        if append and conversation is None:
            raise ValueError("A conversation is required to append the generated utterance")

        transcript = None
        name = self.id
        if conversation is not None:
            name = conversation.alias.get(self.id, self.id)
            if reply_to is None:
                reply_to = generation.last_utterance_id(conversation)
            if reply_to is not None:
                transcript = conversation.get_transcript(until=conversation.get_utterance(reply_to))
        prompt = generation.build_prompt(
            config["prompt"],
            transcript,
            "Write the next message in the conversation as {}. "
            "Respond with only the text of the message.".format(name),
        )
        text = generation.call_llm(
            generation.get_client(config, client, config_manager), config, prompt
        )

        utterance = Utterance(
            id=id or generation.new_id(),
            speaker=self,
            conversation_id=conversation.id if conversation is not None else None,
            reply_to=reply_to,
            timestamp=(
                timestamp if timestamp is not None else generation.next_timestamp(conversation)
            ),
            text=text,
            ai_meta={"config": dict(config)},
        )
        if append:
            conversation.owner.add_utterances([utterance])
            utterance = conversation.owner.get_utterance(utterance.id)
        return utterance

    ############################################################################
    ## end AI fields
    ############################################################################

    def _add_utterance(self, utt):
        self.utterances[utt.id] = utt

    def _add_conversation(self, convo):
        self.conversations[convo.id] = convo

    def get_utterance(self, ut_id: str):  # -> Utterance:
        """
        Get the Utterance with the specified Utterance id

        :param ut_id: The id of the Utterance
        :return: An Utterance object
        """
        return self.utterances[ut_id]

    def iter_utterances(self, selector=lambda utt: True):  # -> Generator[Utterance, None, None]:
        """
        Get utterances made by the Speaker, with an optional selector that selects for Utterances that
        should be included.

                :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
                        By default, the selector includes all Utterances in the Corpus.
        :return: An iterator of the Utterances made by the speaker
        """
        for v in self.utterances.values():
            if selector(v):
                yield v

    def get_utterances_dataframe(self, selector=lambda utt: True, exclude_meta: bool = False):
        """
        Get a DataFrame of the Utterances made by the Speaker with fields and metadata attributes.
        Set an optional selector that filters for Utterances that should be included.
        Edits to the DataFrame do not change the corpus in any way.

        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a Utterance and returns True or False (i.e. include / exclude).
                By default, the selector includes all Utterances in the Corpus.
        :return: a pandas DataFrame
        """
        return get_utterances_dataframe(self, selector, exclude_meta)

    def get_utterance_ids(self, selector=lambda utt: True) -> List[str]:
        """

        :return: a List of the ids of Utterances made by the speaker
        """
        return list([utt.id for utt in self.iter_utterances(selector)])

    def get_conversation(self, cid: str):  # -> Conversation:
        """
        Get the Conversation with the specified Conversation id

        :param cid: The id of the Conversation
        :return: A Conversation object
        """
        return self.conversations[cid]

    def iter_conversations(
        self, selector=lambda convo: True
    ):  # -> Generator[Conversation, None, None]:
        """

        :return: An iterator of the Conversations that the speaker has participated in
        """
        for v in self.conversations.values():
            if selector(v):
                yield v

    def get_conversations_dataframe(self, selector=lambda convo: True, exclude_meta: bool = False):
        """
        Get a DataFrame of the Conversations the Speaker has participated in, with fields and metadata attributes.
        Set an optional selector that filters for Conversations that should be included. Edits to the DataFrame do not
        change the corpus in any way.

        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
            By default, the selector includes all Conversations in the Corpus.
        :return: a pandas DataFrame
        """
        return get_conversations_dataframe(self, selector, exclude_meta)

    def get_conversation_ids(self, selector=lambda convo: True) -> List[str]:
        """

        :return: a List of the ids of Conversations started by the speaker
        """
        return [convo.id for convo in self.iter_conversations(selector)]

    def print_speaker_stats(self):
        """
        Helper function for printing the number of Utterances made and Conversations participated in by the Speaker.

        :return: None (prints output)
        """
        print("Number of Utterances: {}".format(len(list(self.iter_utterances()))))
        print("Number of Conversations: {}".format(len(list(self.iter_conversations()))))

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Speaker):
            return False
        try:
            return self.id == other.id
        except AttributeError:
            return self.__dict__["_name"] == other.__dict__["_name"]

    def __str__(self):
        if self.is_ai or self.__dict__.get("_ai_meta"):
            return "Speaker(id: {}, is_ai: {}, vectors: {}, meta: {}, ai_meta: {})".format(
                repr(self.id), self.is_ai, self.vectors, self.meta, self.ai_meta
            )
        return "Speaker(id: {}, vectors: {}, meta: {})".format(
            repr(self.id), self.vectors, self.meta
        )
