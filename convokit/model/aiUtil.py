"""
Shared helpers for the AI fields of the ConvoKit data model (see AI_FORMAT.md).

On disk, AI fields live inside the regular ConvoKit metadata, so a corpus with AI fields is
still laid out like any other ConvoKit corpus:

- speaker meta: ``"is_ai"``, ``"ai_meta"`` (keys ``"config"``, ``"role"``)
- utterance meta: ``"ai_meta"`` (keys ``"config"``, ``"supports"``)
- conversation meta: ``"ai_meta"`` (keys ``"alias"``, ``"private_assistants"``, ``"supports"``)
- corpus meta: ``"has_ai"``, ``"ai_meta"``

In memory, these fields are pulled out of ``.meta`` and exposed as attributes.
"""

from typing import Dict, Iterable, Optional, Tuple

# metadata keys holding AI fields, per object type
AI_META_KEYS = {
    "speaker": ("ai_meta", "is_ai"),
    "utterance": ("ai_meta",),
    "conversation": ("ai_meta",),
    "corpus": ("ai_meta", "has_ai"),
}

# corpus meta keys used by older versions of the format; migrated onto conversations on load
LEGACY_CORPUS_KEYS = ("supports", "assistants")

# metadata keys used for AI fields by older versions of the format, per object type; removed from
# the regular metadata when a corpus with AI fields is loaded (a speaker's "role" is moved into
# its ai_meta)
LEGACY_AI_KEYS = {
    "speaker": ("role", "sender_name", "agent_id"),
    "utterance": ("sender_name", "agent_id"),
    "conversation": ("sender_name", "agent_id"),
    "corpus": ("sender_name", "agent_id"),
}


def strip_keys(meta: Optional[Dict], keys: Iterable[str]) -> Dict:
    """
    Return a copy of ``meta`` with the given keys removed.

    :param meta: metadata dict (or ``None``, treated as empty)
    :param keys: keys to remove; keys not present are ignored
    :return: a new dict without ``keys``
    """
    cleaned = dict(meta or {})
    for key in keys:
        cleaned.pop(key, None)
    return cleaned


def split_ai_fields(meta: Optional[Dict], obj_type: str) -> Tuple[Dict, Dict]:
    """
    Split a metadata dict into regular metadata and AI fields for the given object type.

    ``meta`` is returned as is (not copied) if it holds no AI fields, so this is cheap to call
    for every component of a corpus without AI fields. Other mappings (e.g. ``ConvoKitMeta``)
    are also returned as is.

    :param meta: metadata dict (or None)
    :param obj_type: ``"speaker"``, ``"utterance"``, ``"conversation"``, or ``"corpus"``
    :return: tuple of the metadata without AI fields and a dict of the AI fields that were present
    """
    if not isinstance(meta, dict):
        return meta, {}
    keys = AI_META_KEYS[obj_type]
    if not any(key in meta for key in keys):
        return meta, {}
    ai_fields = {key: meta[key] for key in AI_META_KEYS[obj_type] if key in meta}
    return strip_keys(meta, keys), ai_fields


def as_dict(value) -> Dict:
    """
    Return a shallow copy of ``value`` if it is a dict, otherwise an empty dict.

    :param value: any value
    :return: a dict
    """
    return dict(value) if isinstance(value, dict) else {}


def normalize_for_dump(value):
    """
    Recursively convert Support / PrivateAssistant objects into JSON-serializable dicts.

    Objects with a ``to_dict()`` method are replaced by its result; dicts, lists and tuples are
    converted recursively (tuples become lists, dict keys become strings).

    :param value: the value to convert
    :return: the converted value
    """
    if hasattr(value, "to_dict") and not isinstance(value, dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): normalize_for_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_dump(item) for item in value]
    return value


def transcript_sort_key(obj) -> Tuple:
    """
    Sort key ordering Utterances / Supports by timestamp, then id.

    Numeric timestamps sort before string timestamps, and a missing timestamp sorts as 0.

    :param obj: an Utterance or Support
    :return: a tuple usable as a sort key
    """
    # support timestamps may be strings while utterance timestamps are ints
    timestamp = obj.timestamp
    if timestamp is None:
        key = (0, 0)
    elif isinstance(timestamp, (int, float)):
        key = (0, timestamp)
    else:
        key = (1, str(timestamp))
    return key, str(obj.id)
