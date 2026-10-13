"""
Helper functions for loading and dumping the AI fields of a Corpus (see AI_FORMAT.md).

On disk, the AI fields are stored in the regular metadata; in memory, they are attributes of the
Speakers, Utterances, Conversations, and the Corpus. The functions here move them between the two.
"""

import json
import os
import pickle
import re
from typing import Dict, List, Optional

from .aiUtil import AI_META_KEYS, LEGACY_AI_KEYS, LEGACY_CORPUS_KEYS, as_dict, normalize_for_dump
from .privateAssistant import PrivateAssistant
from .support import Support

# corpus-level supports file written by older versions of the format
LEGACY_SUPPORTS_FILENAME = "supports.json"

COMPONENT_TYPES = ("speaker", "utterance", "conversation")


def _unlocked_meta_deletion(corpus, obj_type, fn):
    """Call ``fn()`` with metadata deletion temporarily unlocked for ``obj_type``."""
    corpus.meta_index.lock_metadata_deletion[obj_type] = False
    try:
        return fn()
    finally:
        corpus.meta_index.lock_metadata_deletion[obj_type] = True


def _index_keys(corpus, obj_type: str) -> set:
    """Metadata keys listed in the corpus index for ``obj_type``."""
    return set(corpus.meta_index.get_index(obj_type).keys())


def is_ai_corpus_on_disk(corpus, filename: Optional[str] = None) -> bool:
    """
    Whether the corpus metadata (as loaded from disk) holds AI fields in any form, including the
    legacy ones. Only then are legacy AI keys (e.g. a speaker's ``meta["role"]``) treated as AI
    fields, so regular corpora that happen to use these metadata keys are left untouched.

    :param corpus: the Corpus, before its AI fields are extracted
    :param filename: the corpus directory, if it was loaded from one
    :return: True if the corpus has AI fields
    """
    if any(key in corpus.meta for key in AI_META_KEYS["corpus"]):
        return True
    if any(_index_keys(corpus, t) & set(AI_META_KEYS[t]) for t in COMPONENT_TYPES):
        return True
    return filename is not None and os.path.isfile(os.path.join(filename, LEGACY_SUPPORTS_FILENAME))


def extract_component_ai_fields(corpus, legacy: bool = False) -> None:
    """
    Move the AI fields of every Speaker, Utterance, and Conversation out of their metadata into
    their attributes, and remove the AI keys from the metadata index.

    Object types whose index lists no AI keys are skipped, so this costs nothing for a corpus
    without AI fields.

    :param corpus: the Corpus to modify
    :param legacy: whether to also handle the AI keys used by older versions of the format
    :return: None
    """
    if not hasattr(corpus, "utterances"):
        # empty corpus
        return

    for obj_type in COMPONENT_TYPES:
        keys = AI_META_KEYS[obj_type] + (LEGACY_AI_KEYS[obj_type] if legacy else ())
        indexed = _index_keys(corpus, obj_type) & set(keys)
        if not indexed:
            continue
        if corpus.backend == "db":
            _extract_ai_fields_db(corpus, obj_type, list(keys), legacy)
        else:
            _extract_ai_fields_mem(corpus, obj_type, list(keys), legacy)
        for key in indexed:
            corpus.meta_index.del_from_index(obj_type, key)


def _objects_by_id(corpus, obj_type: str) -> Dict:
    """The corpus components of ``obj_type``, by id."""
    if obj_type == "speaker":
        # include speakers only reachable through an utterance
        speakers = {u.speaker.id: u.speaker for u in corpus.iter_utterances()}
        speakers.update(corpus.speakers)
        return speakers
    if obj_type == "utterance":
        return corpus.utterances
    return getattr(corpus, "conversations", {})


def _extract_ai_fields_mem(corpus, obj_type: str, keys: List[str], legacy: bool) -> None:
    """Move the AI fields of each component, one at a time (for the mem backend)."""

    def extract():
        for obj in _objects_by_id(corpus, obj_type).values():
            stored = {key: obj.meta[key] for key in keys if key in obj.meta}
            if stored:
                obj._apply_ai_fields(stored, legacy=legacy)
                for key in stored:
                    del obj.meta[key]

    _unlocked_meta_deletion(corpus, obj_type, extract)


def _extract_ai_fields_db(corpus, obj_type: str, keys: List[str], legacy: bool) -> None:
    """
    Move the AI fields of all components with bulk database operations (for the db backend):
    one query to read them from the metadata, one bulk write to store them with the components,
    and one update to remove them from the metadata.
    """
    from pymongo import UpdateOne

    mapper = corpus.backend_mapper
    meta_collection = mapper.get_collection("meta")
    prefix = "^" + re.escape(obj_type + "_")
    query = {"_id": {"$regex": prefix}, "$or": [{key: {"$exists": True}} for key in keys]}
    index = corpus.meta_index.get_index(obj_type)
    objs = _objects_by_id(corpus, obj_type)

    updated = []
    corpus._ai_persist_suspended = True
    try:
        for doc in meta_collection.find(query, projection=keys):
            obj = objs.get(doc.pop("_id")[len(obj_type) + 1 :])
            if obj is None:
                continue
            for key, value in doc.items():
                if index.get(key) == ["bin"]:
                    doc[key] = pickle.loads(value)
            obj._apply_ai_fields(doc, legacy=legacy)
            updated.append(obj)
    finally:
        corpus._ai_persist_suspended = False

    writes = []
    for obj in updated:
        fields = {
            name: normalize_for_dump(obj.__dict__["_" + name])
            for name in obj.AI_FIELDS
            if "_" + name in obj.__dict__
        }
        if fields:
            writes.append(UpdateOne({"_id": obj.id}, {"$set": fields}))
    if writes:
        mapper.get_collection(obj_type).bulk_write(writes, ordered=False)
    meta_collection.update_many({"_id": {"$regex": prefix}}, {"$unset": {key: "" for key in keys}})


def extract_corpus_ai_fields(corpus, filename: Optional[str] = None, legacy: bool = False) -> Dict:
    """
    Remove the corpus-level AI fields from ``corpus.meta`` and return them.

    If ``legacy`` is True, also returns any corpus-level assistants / supports stored by older
    versions of the format (including a ``supports.json`` file in the corpus directory), so they
    can be migrated onto conversations.

    :param corpus: the Corpus to read from (its metadata is modified)
    :param filename: the corpus directory to look for a legacy ``supports.json`` in, if any
    :param legacy: whether to also handle the keys used by older versions of the format
    :return: dict with keys ``"ai_meta"``, ``"has_ai"`` (None if not stored),
        ``"legacy_assistants"``, and ``"legacy_supports"``
    """
    ai_meta = as_dict(corpus.meta.get("ai_meta"))
    fields = {
        "ai_meta": {k: v for k, v in ai_meta.items() if k != "assistants"},
        "has_ai": corpus.meta.get("has_ai"),
        "legacy_assistants": [],
        "legacy_supports": [],
    }
    keys = set(AI_META_KEYS["corpus"])

    if legacy:
        fields["legacy_assistants"] = (
            ai_meta.get("assistants") or corpus.meta.get("assistants") or []
        )
        fields["legacy_supports"] = corpus.meta.get("supports", []) or []
        keys |= set(LEGACY_AI_KEYS["corpus"]) | set(LEGACY_CORPUS_KEYS)
        if filename is not None and os.path.isdir(filename):
            support_path = os.path.join(filename, LEGACY_SUPPORTS_FILENAME)
            if os.path.exists(support_path):
                with open(support_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                fields["legacy_supports"] = (
                    payload.get("supports", []) if isinstance(payload, dict) else payload
                )

    for key in keys:
        if key in corpus.meta:
            del corpus.meta[key]

    return fields


def migrate_legacy_assistants_and_supports(corpus, assistants, supports: List[Dict]) -> None:
    """
    Move corpus-level assistants and supports (from older versions of the format) onto the
    Conversations they belong to.

    Each assistant is added to the Conversation matching its ``conversation_id``, and each support
    to the PrivateAssistant matching its ``private_assistant_id``. Supports with no matching
    assistant are added directly to the Conversation matching their ``conversation_id``. Entries
    with no matching Conversation are dropped.

    :param corpus: the Corpus to modify
    :param assistants: PrivateAssistants (or their dict forms)
    :param supports: Supports in dict form
    :return: None
    """
    assistants = PrivateAssistant.normalize_list(assistants)
    if not assistants and not supports:
        return

    assistants_by_id = {assistant.id: assistant for assistant in assistants}
    for record in supports or []:
        support = Support.normalize_list([record])
        if not support:
            continue
        support = support[0]
        assistant = assistants_by_id.get(support.private_assistant_id)
        if assistant is not None:
            if support.id not in {s.id for s in assistant.supports}:
                assistant.supports.append(support)
            continue
        convo_id = record.get("conversation_id") if isinstance(record, dict) else None
        if convo_id is not None and corpus.has_conversation(convo_id):
            convo = corpus.get_conversation(convo_id)
            if support.id not in {s.id for s in convo.supports}:
                convo.supports = convo.supports + [support]

    for assistant in assistants:
        if assistant.conversation_id is not None and corpus.has_conversation(
            assistant.conversation_id
        ):
            convo = corpus.get_conversation(assistant.conversation_id)
            if assistant.id not in {a.id for a in convo.private_assistants}:
                convo.private_assistants = convo.private_assistants + [assistant]


def stash_ai_fields_in_meta(corpus) -> None:
    """
    Write AI fields into the metadata of every component and the corpus (the on-disk format), so
    the dump writes them out. Undo with :func:`unstash_ai_fields_from_meta`.

    Nothing is written for a corpus without AI fields (``has_ai`` is False and no component has
    ``ai_meta``), so dumping a regular corpus is unchanged.

    :param corpus: the Corpus to modify
    :return: bool, whether any AI fields were written
    """
    if not corpus._has_ai_fields():
        return False

    if corpus.backend == "db":
        _stash_ai_fields_db(corpus)
        return True

    type_check = corpus.meta_index.type_check
    # keys missing from the index are silently skipped by the dump, so make sure they get indexed
    corpus.meta_index.enable_type_check()
    try:
        for speaker in corpus.iter_speakers():
            speaker.meta["ai_meta"] = normalize_for_dump(speaker.ai_meta)
            speaker.meta["is_ai"] = speaker.is_ai

        for utt in corpus.iter_utterances():
            utt.meta["ai_meta"] = normalize_for_dump(utt.ai_meta)

        for convo in corpus.iter_conversations():
            convo.meta["ai_meta"] = normalize_for_dump(convo.ai_meta)

        corpus.meta["ai_meta"] = normalize_for_dump(corpus.ai_meta)
        corpus.meta["has_ai"] = corpus.has_ai
    finally:
        if not type_check:
            corpus.meta_index.disable_type_check()
    return True


def _stash_ai_fields_db(corpus) -> None:
    """:func:`stash_ai_fields_in_meta` with bulk database writes (for the db backend)."""
    from pymongo import UpdateOne

    meta_collection = corpus.backend_mapper.get_collection("meta")
    for obj_type in COMPONENT_TYPES:
        writes = []
        for obj in corpus.iter_objs(obj_type):
            fields = {"ai_meta": normalize_for_dump(obj.ai_meta)}
            if obj_type == "speaker":
                fields["is_ai"] = obj.is_ai
            writes.append(UpdateOne({"_id": obj.meta.backend_key}, {"$set": fields}))
        if writes:
            meta_collection.bulk_write(writes, ordered=False)
        # keys missing from the index are skipped by the dump
        corpus.meta_index.set_index(obj_type, "ai_meta", str(dict))
        if obj_type == "speaker":
            corpus.meta_index.set_index(obj_type, "is_ai", str(bool))
    corpus.meta["ai_meta"] = normalize_for_dump(corpus.ai_meta)
    corpus.meta["has_ai"] = corpus.has_ai


def unstash_ai_fields_from_meta(corpus) -> None:
    """
    Remove the AI fields written by :func:`stash_ai_fields_in_meta` from all metadata and from the
    metadata index.

    :param corpus: the Corpus to modify
    :return: None
    """
    for obj_type in COMPONENT_TYPES:
        if corpus.backend == "db":
            corpus.backend_mapper.get_collection("meta").update_many(
                {"_id": {"$regex": "^" + re.escape(obj_type + "_")}},
                {"$unset": {key: "" for key in AI_META_KEYS[obj_type]}},
            )
        else:

            def remove():
                for obj in corpus.iter_objs(obj_type):
                    for key in AI_META_KEYS[obj_type]:
                        if key in obj.meta:
                            del obj.meta[key]

            _unlocked_meta_deletion(corpus, obj_type, remove)
        for key in AI_META_KEYS[obj_type]:
            corpus.meta_index.del_from_index(obj_type, key)

    for key in AI_META_KEYS["corpus"]:
        if key in corpus.meta:
            del corpus.meta[key]
