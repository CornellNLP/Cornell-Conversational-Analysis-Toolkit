"""
Claim-level consensus: the propositions each participant holds before and after a mediated
conversation, and the extent to which they come to be shared.

Pipeline
--------
Each speaker, including the mediator, is assigned a memory of atomic claims (short,
self-contained propositions, e.g. "Data centers use a lot of water."). Utterances are processed
in chronological order:

1. Extraction. An LLM decomposes the utterance into atomic claims (``claim_extraction_prompt``),
   given the preceding ``context_window`` utterances for pronoun and reference resolution. Each
   claim is typed ``own`` (the speaker's position), ``reported`` (attributed to someone else) or
   ``procedural`` (about the conversation itself). Claims extracted from the same utterance that
   mutually entail each other under a DeBERTa NLI cross-encoder (probability >= ``sibling_entail``
   in both directions) are collapsed into one.
2. Retrieval. Candidate entries are the speaker's own active claims (all of them, or the nearest
   ``own_cap``) and the ``top_k`` nearest active claims of every other speaker, by cosine
   similarity of sentence embeddings.
3. Consolidation. An LLM (``claim_consolidation_prompt``) returns, for each new claim, its logical
   relation to one candidate (``equivalent``, ``forward_entail``, ``backward_entail``,
   ``contradiction`` or ``neutral``), the candidate's id, a ``merged_claim`` when the new claim
   refines the candidate, and ``supersedes`` when it contradicts the speaker's own earlier claim.
4. Memory update. The relation is mapped deterministically to an operation; the model never
   chooses the operation, and no entry is ever deleted:

   ================================  ==========================================================
   relation (to the chosen target)   operation
   ================================  ==========================================================
   neutral, or no valid target       ADD a new entry
   any relation, other speaker       ADD a new entry with a link to the other speaker's claim
   equivalent / backward_entail      RESTATE: log the restatement on the existing entry
   forward_entail                    REFINE: replace the text with ``merged_claim``, keep the
                                     previous text in the entry's ``history``
   contradiction, supersedes=true    SUPERSEDE: mark the old entry ``retracted`` and link the
                                     new entry to it
   contradiction, supersedes=false   ADD a new entry with a contradiction link (both held)
   ================================  ==========================================================

   Rows are checked from top to bottom; the other rows apply when the target is one of the
   speaker's own claims. Every entry records ``first_turn``, ``last_turn``, ``history``,
   ``restated``, ``links`` and ``status``; a speaker's held claims are their active ``own``
   entries.

Participants' memories may be seeded from a pre-survey statement (``seed_text_func``) to
establish their initial position.

Measurement
-----------
After the last utterance, for each pair of participants, an LLM (``claim_agreement_prompt``)
labels pairs of claims across the two participants' lists as ``agree`` or ``contradiction``,
once on the seed lists and once on the final held lists. A pair's agreement is the mean, over
its two participants, of the share of their claims with an agreeing counterpart; the
conversation's agreement is the mean over participant pairs. Per conversation:

- ``agreement_gain`` = (final - initial) / (1 - initial): the fraction of the remaining
  disagreement closed during the conversation (negative if agreement decreased; undefined if
  the initial agreement is already 1).

Per participant, using the links recorded at consolidation time to find which speaker stated a
claim first (``origin_of``):

- ``share_from_mediator``: share of the participant's final claims first stated by the mediator;
- ``share_from_partner``: share first stated by another participant;
- ``share_new_common_ground``: of the claims newly shared with a partner at the end (not shared
  at the start), the share first stated by the mediator.

Reference: CIG: Measuring Conversational Information Gain in Deliberative Dialogues with
Semantic Memory Dynamics, https://aclanthology.org/2026.acl-long.2203/
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from convokit import Transformer
from convokit.genai import GenAIConfigManager
from convokit.model import Conversation, Corpus, Speaker

from .llm import get_eval_llm_client
from .util import (
    default_is_mediator,
    default_speaker_name,
    default_topic,
    display_names,
    pre_survey_text,
)

MEDIATOR = "Mediator"
MEDIATOR_LABEL = "Mediator(mod)"
PRE_SURVEY = "pre-survey"
RELATIONS = (
    "equivalent",
    "forward_entail",
    "backward_entail",
    "contradiction",
    "neutral",
)
KINDS = ("own", "reported", "procedural")
# relations through which a claim counts as taken over from another speaker's claim
ADOPT_RELS = ("equivalent", "forward_entail")


def _turn(turn) -> int:
    """Turn id as a number, with the pre-survey as -1."""
    return -1 if turn == PRE_SURVEY else int(turn)


def _entry_index(entry_id: str) -> int:
    """Position of a memory entry from its id ("c_<n>" -> n)."""
    return int(entry_id.split("_")[1])


def _relation(label) -> Optional[str]:
    """Normalize the LLM's label: "agree..." -> "agree", "contradict..." -> "contradiction", else None."""
    label = str(label or "").strip().lower()
    if label.startswith("agree"):
        return "agree"
    if label.startswith("contradict"):
        return "contradiction"
    return None


class _VectorIndex:
    """Per-speaker store of unit-length claim embeddings, for nearest-neighbour retrieval."""

    def __init__(self):
        self.vectors = {}

    def upsert(self, speaker: str, entry_id: str, vector):
        self.vectors.setdefault(speaker, {})[entry_id] = np.asarray(vector)

    def nearest(self, speaker: str, query, k: int, among=None) -> List[str]:
        """The ids of the ``k`` entries of ``speaker`` nearest to ``query``, only considering
        the ids in ``among`` if it is given."""
        entries = self.vectors.get(speaker, {})
        ids = [i for i in entries if among is None or i in among]
        if not ids:
            return []
        scores = np.stack([entries[i] for i in ids]) @ np.asarray(query)
        return [ids[j] for j in np.argsort(-scores)[:k]]


class ClaimTracker(Transformer):
    """
    ConvoKit Transformer that tracks the claims of every speaker in each conversation and
    measures the consensus the participants reach.

    Each utterance is decomposed by an LLM into atomic claims, which are consolidated into a
    per-speaker claim memory (added, restated, refined or superseded). At the end, an LLM judges
    which claims the participants agree on before and after the conversation, and the claim
    links are used to find whether a participant's claims first came from the mediator or from
    another participant. See the module docstring for the full method.

    Usage::

        tracker = ClaimTracker(model="gpt-5-mini")
        tracker.transform(corpus)
        tracker.summarize(corpus)   # one row per participant per conversation

    What gets written to the corpus:

    - ``convo.meta["claims"]``: everything the tracker produced for the conversation; the keys
      are ``initial`` and ``final`` (each participant's claim list before and after), ``memory``
      (every claim entry with its history and links), ``steps`` (what each utterance extracted
      and how it was recorded) and ``agreement`` (the judged pairs and the gain).
    - per participant, via ``corpus.get_speaker_convo_info(speaker_id, convo_id)``:
      ``claims_initial``, ``claims_final``, ``claims_share_from_mediator``,
      ``claims_share_from_partner``, ``claims_share_new_common_ground``.

    The corpus must identify the mediator and the name each speaker goes by in the chat. By
    default these come from the ConvoKit AI fields (``speaker.ai_meta["role"]`` and
    ``conversation.alias``); they can be replaced with ``is_mediator`` and ``name_func``.

    Reference: CIG: Measuring Conversational Information Gain in Deliberative Dialogues with
    Semantic Memory Dynamics, https://aclanthology.org/2026.acl-long.2203/

    :param llm_client: client with a ``generate_json(prompt)`` method; by default one for
        ``model`` in JSON mode with a fixed ``seed``, built with
        :func:`get_eval_llm_client <convokit.ai_evaluation.llm.get_eval_llm_client>`
    :param model: model name for the default client: an OpenAI model (e.g. ``"gpt-5-mini"``) or
        a Gemini model (e.g. ``"gemini-2.5-flash"``)
    :param provider: ``"gpt"`` or ``"gemini"`` for the default client; inferred from ``model``
        if not given
    :param config_manager: GenAIConfigManager holding the API key for the default client
    :param reasoning_effort: OpenAI reasoning effort for the default client (not used for Gemini)
    :param seed: seed for the default client
    :param embedder: a SentenceTransformer used for retrieval (default: ``embedding_model``)
    :param embedding_model: name of the embedding model to load if ``embedder`` is not given
        (Qwen3 needs transformers >= 4.51)
    :param nli: a sentence_transformers CrossEncoder whose label 1 is entailment, used to collapse
        near-paraphrases extracted from one utterance (default: ``nli_model``)
    :param nli_model: name of the NLI model to load if ``nli`` is not given
    :param device: torch device for the default embedder and NLI model (default: ``"mps"`` if
        available, else ``"cpu"``)
    :param top_k: number of nearest claims of each other speaker shown to the consolidation LLM
    :param own_cap: if the speaker has more active claims than this, only roughly this many of
        the nearest ones are shown to the consolidation LLM
    :param context_window: number of preceding utterances shown to the extraction LLM
    :param sibling_entail: mutual entailment probability from which two claims extracted from
        one utterance count as the same claim
    :param is_mediator: function from Speaker to whether it is the mediator
    :param name_func: function from (Conversation, Speaker) to the name used for a participant
    :param topic_func: function from Conversation to the topic line shown to the model
    :param seed_text_func: function from (Conversation, Speaker) to the text a participant's
        memory is seeded from (by default ``conversation.meta["pre_survey"][speaker.id]``), or
        None for no seeding
    :param custom_prompts: dict replacing any of the prompt files, with keys ``"extraction"``,
        ``"consolidation"`` and ``"agreement"``
    :param claims_attribute_name: name of the conversation metadata attribute to store results in
    :param n_workers: number of conversations to track concurrently
    :param verbosity: print progress every ``verbosity`` conversations (0 to disable)
    """

    PROMPTS = None

    @classmethod
    def _load_prompts(cls):
        if cls.PROMPTS is None:
            base_path = os.path.join(os.path.dirname(__file__), "prompts")
            prompts = {}
            for name in ("extraction", "consolidation", "agreement"):
                with open(
                    os.path.join(base_path, f"claim_{name}_prompt.txt"),
                    encoding="utf-8",
                ) as f:
                    prompts[name] = f.read()
            cls.PROMPTS = prompts

    def __init__(
        self,
        llm_client=None,
        model: str = "gpt-5-mini",
        provider: Optional[str] = None,
        config_manager: Optional[GenAIConfigManager] = None,
        reasoning_effort: str = "minimal",
        seed: int = 0,
        embedder=None,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        nli=None,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None,
        top_k: int = 5,
        own_cap: int = 40,
        context_window: int = 3,
        sibling_entail: float = 0.5,
        is_mediator: Callable[[Speaker], bool] = default_is_mediator,
        name_func: Callable[[Conversation, Speaker], str] = default_speaker_name,
        topic_func: Callable[[Conversation], str] = default_topic,
        seed_text_func: Optional[
            Callable[[Conversation, Speaker], Optional[str]]
        ] = pre_survey_text,
        custom_prompts: Optional[Dict[str, str]] = None,
        claims_attribute_name: str = "claims",
        n_workers: int = 1,
        verbosity: int = 10,
    ):
        self._load_prompts()
        self.prompts = {**self.PROMPTS, **(custom_prompts or {})}
        self.llm_client = llm_client or get_eval_llm_client(
            model,
            provider=provider,
            config_manager=config_manager,
            reasoning_effort=reasoning_effort,
            seed=seed,
            json_mode=True,
        )
        if embedder is None or nli is None:
            from sentence_transformers import CrossEncoder, SentenceTransformer

            if device is None:
                import torch

                device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.embedder = embedder or SentenceTransformer(embedding_model, device=device)
        self.nli = nli or CrossEncoder(nli_model, device=device)
        self._model_lock = (
            threading.Lock()
        )  # one torch model call at a time (MPS is not thread-safe)

        self.top_k = top_k
        self.own_cap = own_cap
        self.context_window = context_window
        self.sibling_entail = sibling_entail
        self.is_mediator = is_mediator
        self.name_func = name_func
        self.topic_func = topic_func
        self.seed_text_func = seed_text_func
        self.claims_attribute_name = claims_attribute_name
        self.n_workers = n_workers
        self.verbosity = verbosity

    # -- model calls -------------------------------------------------------

    def _chat_json(self, prompt: str):
        """Send a prompt to the LLM and return its parsed JSON reply."""
        return self.llm_client.generate_json(prompt)

    def _embed(self, texts: List[str]):
        """Unit-length embeddings of the texts, as lists."""
        with self._model_lock:
            return self.embedder.encode(list(texts), normalize_embeddings=True).tolist()

    def _same_claim(self, a: str, b: str) -> bool:
        """Whether two claims entail each other in both directions under the NLI model."""
        with self._model_lock:
            p = self.nli.predict([(a, b), (b, a)], apply_softmax=True)[:, 1]
        return float(min(p)) >= self.sibling_entail

    # -- extraction --------------------------------------------------------

    def extract_claims(self, context: str, target: str, speaker: str, turn) -> List[dict]:
        """
        Extract the atomic claims of one utterance with the LLM, with near-paraphrases collapsed
        into one claim (the others are kept as its ``siblings``). The speaker is taken from the
        transcript, not from the LLM.

        :param context: the preceding utterances (or the topic line), as text
        :param target: the utterance to extract claims from, as a transcript line
        :param speaker: the name of the utterance's speaker
        :param turn: the utterance's position in the conversation, or ``"pre-survey"``
        :return: list of claim dicts with ``speaker``, ``target_speaker``, ``claim``, ``kind``,
            ``about``, ``turn_id`` and ``siblings``
        """
        prompt = (
            f"{self.prompts['extraction']}\n\n### Task\n\n**Context**\n{context}\n\n"
            f"**Target utterance**\n{target}\n**Output** (List of extracted memory json objects)\n"
        )
        claims = []
        for m in self._chat_json(prompt).get("memories", []):
            if not isinstance(m, dict) or not str(m.get("claim", "")).strip():
                continue
            claims.append(
                {
                    "speaker": speaker,
                    "target_speaker": m.get("target_speaker") or "Everyone",
                    "claim": str(m["claim"]).strip(),
                    "kind": m.get("kind") if m.get("kind") in KINDS else "own",
                    "about": m.get("about") if m.get("kind") == "reported" else None,
                    "turn_id": str(turn),
                }
            )
        return self.collapse_siblings(claims)

    def collapse_siblings(self, claims: List[dict]) -> List[dict]:
        """
        Collapse claims of the same kind that entail each other (see ``sibling_entail``) into
        the first of them, keeping the texts of the others in its ``siblings``.

        :param claims: claims extracted from one utterance
        :return: the remaining claims, each with a ``siblings`` list
        """
        kept = []
        for c in claims:
            twin = next(
                (
                    k
                    for k in kept
                    if k["kind"] == c["kind"] and self._same_claim(k["claim"], c["claim"])
                ),
                None,
            )
            if twin is None:
                kept.append({**c, "siblings": []})
            else:
                twin["siblings"].append(c["claim"])
        return kept

    # -- memory ------------------------------------------------------------

    @staticmethod
    def new_entry(memory: List[dict], claim: dict, turn) -> dict:
        """
        Create a memory entry for a claim (without adding it to the memory).

        :param memory: the current memory, used to number the entry
        :param claim: the claim dict
        :param turn: the turn at which the claim was made
        :return: the new entry
        """
        turn = str(turn)
        return {
            "id": f"c_{len(memory)}",
            "speaker": claim["speaker"],
            "role": "mediator" if claim["speaker"] == MEDIATOR else "participant",
            "target_speaker": claim.get("target_speaker"),
            "claim": claim["claim"],
            "kind": claim.get("kind", "own"),
            "about": claim.get("about"),
            "first_turn": turn,
            "last_turn": turn,
            "status": "active",
            "history": [
                {
                    "turn": turn,
                    "op": "ADD",
                    "claim": claim["claim"],
                    "relation": None,
                    "target": None,
                }
            ],
            "restated": [
                {"turn": turn, "claim": t, "relation": "sibling"} for t in claim.get("siblings", [])
            ],
            "links": [],
            "supersedes": None,
            "superseded_by": None,
            "retracted_at": None,
        }

    @staticmethod
    def active(memory: List[dict], speaker: Optional[str] = None) -> List[dict]:
        """
        Get the active (not retracted) entries of the memory.

        :param memory: the memory
        :param speaker: if given, only this speaker's entries are returned
        :return: list of active entries
        """
        return [
            m
            for m in memory
            if m["status"] == "active" and (speaker is None or m["speaker"] == speaker)
        ]

    @classmethod
    def held_entries(cls, memory: List[dict], speaker: str, kinds=("own",)) -> List[dict]:
        """
        Get the claims a speaker holds: their active entries of the given kinds.

        :param memory: the memory
        :param speaker: the speaker's name
        :param kinds: the claim kinds to include
        :return: list of entries
        """
        return [m for m in cls.active(memory, speaker) if m["kind"] in kinds]

    def _sync_vectors(self, index: _VectorIndex, entries: List[dict]):
        """(Re-)embed the given entries in the index."""
        if not entries:
            return
        for m, v in zip(entries, self._embed([m["claim"] for m in entries])):
            index.upsert(m["speaker"], m["id"], v)

    def retrieve_candidates(
        self,
        index: _VectorIndex,
        memory: List[dict],
        speaker: str,
        new_claims: List[dict],
    ) -> List[dict]:
        """
        Retrieve the memory entries shown to the consolidation LLM: the speaker's own active
        claims (all of them, or about ``own_cap`` of the nearest ones), followed by the ``top_k``
        nearest active claims of every other speaker, for any of the new claims. Retracted claims
        are never shown.

        :param index: the embedding index of the memory
        :param memory: the memory
        :param speaker: the name of the speaker of the new claims
        :param new_claims: the claims just extracted
        :return: list of candidate entries
        """
        by_id = {m["id"]: m for m in memory}
        vecs = self._embed([c["claim"] for c in new_claims])
        own = self.active(memory, speaker)
        if len(own) > self.own_cap:
            # retracted entries are left out of the search, so they can't take up slots
            own_ids = {m["id"] for m in own}
            keep = set()
            for v in vecs:
                keep.update(index.nearest(speaker, v, self.own_cap // len(vecs) + 1, own_ids))
            own = [m for m in own if m["id"] in keep]
        others = []
        for other in sorted({m["speaker"] for m in memory if m["speaker"] != speaker}):
            active_ids = {m["id"] for m in self.active(memory, other)}
            ids = set()
            for v in vecs:
                ids.update(index.nearest(other, v, self.top_k, active_ids))
            others += [by_id[i] for i in sorted(ids, key=_entry_index)]
        return own + others

    def consolidate(self, shown: List[dict], new_claims: List[dict]) -> List[dict]:
        """
        Ask the LLM for the relation of each new claim to one of the shown candidates.

        :param shown: the candidate entries
        :param new_claims: the claims just extracted
        :return: the LLM's list of memory updates (unvalidated)
        """
        shown = [
            {
                "id": m["id"],
                "speaker": m["speaker"],
                "claim": m["claim"],
                "turn_id": m["first_turn"],
            }
            for m in shown
        ]
        new = [
            {"speaker": c["speaker"], "claim": c["claim"], "turn_id": c["turn_id"]}
            for c in new_claims
        ]
        prompt = (
            self.prompts["consolidation"]
            .replace("{{retrieved_old_memory_dict}}", json.dumps(shown, indent=1))
            .replace("{{new_retrieved_claims_list}}", json.dumps(new, indent=1))
        )
        return self._chat_json(prompt).get("memory_updates", [])

    @staticmethod
    def repair_update(update: dict, shown: Dict[str, dict], speaker: str) -> dict:
        """
        Turn one LLM output into a validated operation (see the table in the module docstring).

        :param update: one memory update from the LLM
        :param shown: dict of id -> candidate entry shown to the LLM
        :param speaker: the name of the speaker of the new claim
        :return: dict with ``op`` (ADD, RESTATE, REFINE or SUPERSEDE), ``relation``, ``target``
            (an entry id or None) and, for REFINE, ``text``
        """
        rel = update.get("logical_relation")
        rel = rel if rel in RELATIONS else "neutral"
        tid = (update.get("target") or {}).get("id")
        if rel == "neutral" or tid not in shown:
            return {"op": "ADD", "relation": "neutral", "target": None}
        if shown[tid]["speaker"] != speaker:
            return {"op": "ADD", "relation": rel, "target": tid}
        if rel in ("equivalent", "backward_entail"):
            return {"op": "RESTATE", "relation": rel, "target": tid}
        if rel == "forward_entail":
            merged = update.get("merged_claim")
            text = merged.strip() if isinstance(merged, str) and merged.strip() else None
            return {"op": "REFINE", "relation": rel, "target": tid, "text": text}
        if update.get("supersedes") is True:
            return {"op": "SUPERSEDE", "relation": rel, "target": tid}
        return {"op": "ADD", "relation": rel, "target": tid}

    @classmethod
    def decide(
        cls,
        updates: List[dict],
        shown: Dict[str, dict],
        new_claims: List[dict],
        speaker: str,
    ) -> List[dict]:
        """
        Pair each new claim with the LLM's output for it (by claim text, else by position) and
        turn it into a validated operation.

        :param updates: the LLM's memory updates
        :param shown: dict of id -> candidate entry shown to the LLM
        :param new_claims: the claims just extracted
        :param speaker: the name of the speaker of the new claims
        :return: list of operations parallel to ``new_claims`` (see ``repair_update``)
        """
        used, picked = [False] * len(updates), [None] * len(new_claims)
        for i, c in enumerate(new_claims):
            for j, u in enumerate(updates):
                src = (u.get("source") or {}).get("claim") if isinstance(u, dict) else None
                if not used[j] and isinstance(src, str) and src.strip() == c["claim"]:
                    picked[i], used[j] = u, True
                    break
        rest = [u for j, u in enumerate(updates) if not used[j] and isinstance(u, dict)]
        for i in range(len(new_claims)):
            if picked[i] is None and rest:
                picked[i] = rest.pop(0)
        return [cls.repair_update(u or {}, shown, speaker) for u in picked]

    @classmethod
    def apply_updates(
        cls, memory: List[dict], decisions: List[dict], new_claims: List[dict], turn
    ) -> List[dict]:
        """
        Apply the operations to the memory, in place.

        :param memory: the memory
        :param decisions: operations parallel to ``new_claims`` (see ``repair_update``)
        :param new_claims: the claims just extracted
        :param turn: the turn at which the claims were made
        :return: the entries that were created or whose text changed
        """
        by_id = {m["id"]: m for m in memory}
        changed = []
        turn = str(turn)
        for d, c in zip(decisions, new_claims):
            tgt = by_id.get(d["target"]) if d["target"] else None
            sibs = [
                {"turn": turn, "claim": t, "relation": "sibling"} for t in c.get("siblings", [])
            ]
            if d["op"] == "RESTATE":
                tgt["restated"] += [{"turn": turn, "claim": c["claim"], "relation": d["relation"]}]
                tgt["restated"] += sibs
                tgt["last_turn"] = turn
                continue
            if d["op"] == "REFINE":
                text = d["text"] or c["claim"]
                tgt["history"].append(
                    {
                        "turn": turn,
                        "op": "REFINE",
                        "claim": text,
                        "relation": d["relation"],
                        "from": c["claim"],
                    }
                )
                tgt["restated"] += sibs
                tgt["claim"], tgt["last_turn"] = text, turn
                changed.append(tgt)
                continue
            entry = cls.new_entry(memory, c, turn)
            if d["op"] == "SUPERSEDE":
                tgt["status"], tgt["retracted_at"], tgt["superseded_by"] = (
                    "retracted",
                    turn,
                    entry["id"],
                )
                entry["supersedes"] = tgt["id"]
            if tgt is not None:
                entry["links"].append(
                    {
                        "turn": turn,
                        "relation": d["relation"],
                        "to": tgt["id"],
                        "speaker": tgt["speaker"],
                    }
                )
                entry["history"][0].update(relation=d["relation"], target=tgt["id"])
            memory.append(entry)
            by_id[entry["id"]] = entry
            changed.append(entry)
        return changed

    # -- one conversation --------------------------------------------------

    def track(self, conversation: Conversation) -> dict:
        """
        Build the claim memory of one conversation.

        :param conversation: the Conversation
        :return: dict with ``speakers`` (participant name -> speaker id), ``initial`` and ``final``
            (name -> held claims), ``memory`` (all entries) and ``steps`` (what each utterance
            extracted, was shown, returned and decided)
        """
        topic = self.topic_func(conversation)
        utts = conversation.get_chronological_utterance_list()
        name_of = display_names(conversation, self.is_mediator, self.name_func)
        # the memory tells the mediator apart by its name, so a participant can't share it
        name_of = {
            sid: f"{name} (participant)" if name == MEDIATOR else name
            for sid, name in name_of.items()
        }
        speakers = {sid: conversation.get_speaker(sid) for sid in name_of}
        lines = [
            f"{pos}. {MEDIATOR_LABEL if self.is_mediator(u.speaker) else name_of[u.speaker.id]}: {u.text}"
            for pos, u in enumerate(utts)
        ]

        index = _VectorIndex()
        memory, initial, steps = [], {}, []

        for sid, name in name_of.items():
            text = self.seed_text_func(conversation, speakers[sid]) if self.seed_text_func else None
            claims = (
                self.extract_claims(
                    f"Debate topic: {topic}", f"0. {name}: {text}", name, PRE_SURVEY
                )
                if text
                else []
            )
            seeded = []
            for c in claims:
                seeded.append(self.new_entry(memory, c, PRE_SURVEY))
                memory.append(seeded[-1])
            self._sync_vectors(index, seeded)
            initial[name] = [c["claim"] for c in claims]
            steps.append(
                {
                    "turn": PRE_SURVEY,
                    "speaker": name,
                    "extracted": claims,
                    "shown": [],
                    "updates": [],
                    "decisions": [{"op": "SEED"}] * len(claims),
                }
            )

        for pos, utt in enumerate(utts):
            name = MEDIATOR if self.is_mediator(utt.speaker) else name_of[utt.speaker.id]
            context = "\n".join(lines[max(0, pos - self.context_window) : pos])
            claims = self.extract_claims(context or f"Debate topic: {topic}", lines[pos], name, pos)
            step = {
                "turn": str(pos),
                "speaker": name,
                "extracted": [dict(c) for c in claims],
                "shown": [],
                "updates": [],
                "decisions": [],
            }
            if claims:
                shown = self.retrieve_candidates(index, memory, name, claims)
                by_id = {m["id"]: m for m in shown}
                updates = self.consolidate(shown, claims)
                decisions = self.decide(updates, by_id, claims, name)
                changed = self.apply_updates(memory, decisions, claims, pos)
                self._sync_vectors(index, changed)
                step.update(shown=[m["id"] for m in shown], updates=updates, decisions=decisions)
            steps.append(step)

        return {
            "speakers": {name: sid for sid, name in name_of.items()},
            "initial": initial,
            "final": {
                n: [m["claim"] for m in self.held_entries(memory, n)] for n in name_of.values()
            },
            "memory": memory,
            "steps": steps,
        }

    # -- agreement ---------------------------------------------------------

    def claim_agreement(self, claims_a: List[str], claims_b: List[str], name_a: str, name_b: str):
        """
        Judge with the LLM which claims of two lists agree or contradict each other.

        :param claims_a: the first participant's claims
        :param claims_b: the second participant's claims
        :param name_a: the first participant's name
        :param name_b: the second participant's name
        :return: dict with ``pairs`` (the judged pairs, by index into each list) and, for
            ``agree`` and ``contradiction``, the mean over both sides of the share of claims that
            have a counterpart with that relation (None if either list is empty)
        """
        if not claims_a or not claims_b:
            return {"pairs": [], "agree": None, "contradiction": None}
        pairs = self._chat_json(
            self.prompts["agreement"].format(
                name_a=name_a,
                name_b=name_b,
                claims_a="\n".join(f"{i}. {c}" for i, c in enumerate(claims_a)),
                claims_b="\n".join(f"{i}. {c}" for i, c in enumerate(claims_b)),
            )
        ).get("pairs", [])
        pairs = [
            {"a": p["a"], "b": p["b"], "relation": _relation(p.get("relation"))}
            for p in pairs
            if isinstance(p, dict)
            and isinstance(p.get("a"), int)
            and isinstance(p.get("b"), int)
            and p["a"] < len(claims_a)
            and p["b"] < len(claims_b)
            and _relation(p.get("relation")) is not None
        ]

        def frac(rel):
            a = len({p["a"] for p in pairs if p["relation"] == rel}) / len(claims_a)
            b = len({p["b"] for p in pairs if p["relation"] == rel}) / len(claims_b)
            return (a + b) / 2

        return {
            "pairs": pairs,
            "agree": frac("agree"),
            "contradiction": frac("contradiction"),
        }

    def judge_agreement(self, record: dict) -> dict:
        """
        Compute the agreement between every pair of participants, before (seed claims) and
        after (held claims) the conversation.

        :param record: the output of ``track``
        :return: dict with ``pairs`` (per participant pair, the ``claim_agreement`` results
            for ``initial`` and ``final``), the means over pairs ``initial``, ``final``,
            ``contradiction_initial`` and ``contradiction_final``, and ``gain``,
            (final - initial) / (1 - initial) (None if undefined)
        """
        names = list(record["initial"])
        pairs = [
            {
                "a": a,
                "b": b,
                "initial": self.claim_agreement(record["initial"][a], record["initial"][b], a, b),
                "final": self.claim_agreement(record["final"][a], record["final"][b], a, b),
            }
            for a, b in combinations(names, 2)
        ]

        def mean(key, stage):
            values = [p[stage][key] for p in pairs if p[stage][key] is not None]
            return sum(values) / len(values) if values else None

        initial, final = mean("agree", "initial"), mean("agree", "final")
        gain = None
        if initial is not None and final is not None and initial < 1:
            gain = (final - initial) / (1 - initial)
        return {
            "pairs": pairs,
            "initial": initial,
            "final": final,
            "gain": gain,
            "contradiction_initial": mean("contradiction", "initial"),
            "contradiction_final": mean("contradiction", "final"),
        }

    # -- origin ------------------------------------------------------------

    @staticmethod
    def _resolve_about(about, names: List[str]) -> Optional[str]:
        """Match the ``about`` field of a reported claim to a unique participant name, if any."""
        if not isinstance(about, str) or not about.strip():
            return None
        a = about.strip().lower()
        for n in names:
            if n.lower() == a:
                return n
        hits = [n for n in names if n.lower() in a or a in n.lower()]
        return hits[0] if len(hits) == 1 else None

    @classmethod
    def origin_of(cls, entry: dict, by_id: Dict[str, dict], names: List[str], seen=frozenset()):
        """
        Find who held a claim first, following its ``equivalent`` and ``forward_entail`` links.

        A claim the mediator reports on behalf of a participant is attributed to that
        participant.

        :param entry: the memory entry
        :param by_id: dict of id -> entry for the whole memory
        :param names: the participants' names
        :param seen: ids already visited (to avoid cycles)
        :return: (speaker name, turn) of the earliest origin, with the pre-survey as turn -1
        """
        who = entry["speaker"]
        if entry["role"] == "mediator" and entry["kind"] == "reported":
            who = cls._resolve_about(entry["about"], names) or "unknown"
        best = (who, _turn(entry["first_turn"]))
        for link in entry["links"]:
            other = by_id.get(link["to"])
            if other is None or link["relation"] not in ADOPT_RELS or other["id"] in seen:
                continue
            cand = cls.origin_of(other, by_id, names, seen | {entry["id"]})
            if cand[1] < best[1]:
                best = cand
        return best

    def source_shares(self, record: dict) -> Dict[str, dict]:
        """
        Count, per participant, how many of their held claims were first held by the mediator or
        by another participant.

        :param record: the output of ``track``
        :return: dict of participant name -> ``n_final``, ``n_from_mediator``,
            ``n_from_partner``, ``share_from_mediator`` and ``share_from_partner`` (shares are
            None if the participant holds no claims)
        """
        names = list(record["initial"])
        by_id = {m["id"]: m for m in record["memory"]}
        out = {}
        for name in names:
            entries = self.held_entries(record["memory"], name)
            origins = [self.origin_of(m, by_id, names) for m in entries]
            n_med = sum(o[0] == MEDIATOR for o in origins)
            n_partner = sum(o[0] in names and o[0] != name for o in origins)
            out[name] = {
                "n_final": len(entries),
                "n_from_mediator": n_med,
                "n_from_partner": n_partner,
                "share_from_mediator": n_med / len(entries) if entries else None,
                "share_from_partner": n_partner / len(entries) if entries else None,
            }
        return out

    def common_ground_shares(self, record: dict, agreement: dict) -> Dict[str, dict]:
        """
        Compute, per participant, the share of their newly shared claims that came from the
        mediator.

        Newly shared claims are those the participant agrees on with a partner at the end but
        not at the start; a claim counts as coming from the mediator if the earliest origin over
        the claim and the partner claims paired with it is the mediator.

        :param record: the output of ``track``
        :param agreement: the output of ``judge_agreement``
        :return: dict of participant name -> ``n_new_shared`` and ``share_new_common_ground``
            (None if there are no newly shared claims)
        """
        names = list(record["initial"])
        by_id = {m["id"]: m for m in record["memory"]}
        held = {n: self.held_entries(record["memory"], n) for n in names}
        seeds = {
            n: [m for m in record["memory"] if m["speaker"] == n and m["first_turn"] == PRE_SURVEY]
            for n in names
        }
        out = {}
        for name in names:
            partner_of, shared_i = {}, set()
            for pr in agreement["pairs"]:
                if name not in (pr["a"], pr["b"]):
                    continue
                me, other = ("a", "b") if pr["a"] == name else ("b", "a")
                other_name = pr[other]
                for e in pr["final"]["pairs"]:
                    if (
                        e["relation"] == "agree"
                        and e[me] < len(held[name])
                        and e[other] < len(held[other_name])
                    ):
                        partner_of.setdefault(e[me], []).append((other_name, e[other]))
                shared_i |= {
                    seeds[name][e[me]]["id"]
                    for e in pr["initial"]["pairs"]
                    if e["relation"] == "agree" and e[me] < len(seeds[name])
                }
            new_shared = [i for i in partner_of if held[name][i]["id"] not in shared_i]
            from_mediator = 0
            for i in new_shared:
                origins = [self.origin_of(held[name][i], by_id, names)] + [
                    self.origin_of(held[o][k], by_id, names) for o, k in partner_of[i]
                ]
                if min(origins, key=lambda o: o[1])[0] == MEDIATOR:
                    from_mediator += 1
            out[name] = {
                "n_new_shared": len(new_shared),
                "share_new_common_ground": from_mediator / len(new_shared) if new_shared else None,
            }
        return out

    # -- transformer -------------------------------------------------------

    def _track_and_judge(self, conversation: Conversation) -> dict:
        """Track one conversation and add the agreement judgments to the record."""
        record = self.track(conversation)
        record["agreement"] = self.judge_agreement(record)
        return record

    def transform(self, corpus: Corpus, selector: Callable[[Conversation], bool] = lambda c: True):
        """
        Track the claims of the selected conversations and store the results (see the class
        docstring).

        Conversations whose tracking fails are reported and left without results.

        :param corpus: the Corpus to transform
        :param selector: function from Conversation to whether it should be tracked
        :return: the Corpus, with results added
        """
        convos = [convo for convo in corpus.iter_conversations() if selector(convo)]
        results = {}
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = {ex.submit(self._track_and_judge, convo): convo.id for convo in convos}
            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    results[futures[fut]] = fut.result()
                except Exception as exc:
                    print("failed:", futures[fut], exc)
                if self.verbosity and i % self.verbosity == 0:
                    print(i, "/", len(convos))

        for convo in convos:
            record = results.get(convo.id)
            if record is None:
                continue
            sources = self.source_shares(record)
            common = self.common_ground_shares(record, record["agreement"])
            convo.meta[self.claims_attribute_name] = record
            for name, sid in record["speakers"].items():
                info = {
                    "claims_initial": record["initial"][name],
                    "claims_final": record["final"][name],
                    "claims_share_from_mediator": sources[name]["share_from_mediator"],
                    "claims_share_from_partner": sources[name]["share_from_partner"],
                    "claims_share_new_common_ground": common[name]["share_new_common_ground"],
                }
                for key, value in info.items():
                    corpus.set_speaker_convo_info(sid, convo.id, key, value)
        return corpus

    def summarize(
        self, corpus: Corpus, selector: Callable[[Conversation], bool] = lambda c: True
    ) -> pd.DataFrame:
        """
        Collect the per-participant results in one table.

        :param corpus: a Corpus that has been transformed
        :param selector: function from Conversation to whether it should be included
        :return: DataFrame with one row per (conversation, participant) and columns
            ``conversation_id``, ``speaker`` (the speaker id), ``n_initial``, ``n_final``,
            ``agreement_gain`` (the conversation's), ``share_from_mediator``,
            ``share_from_partner`` and ``share_new_common_ground``
        """
        rows = []
        for convo in corpus.iter_conversations():
            record = convo.meta.get(self.claims_attribute_name)
            if not selector(convo) or not record:
                continue
            for name, sid in record["speakers"].items():
                info = corpus.get_speaker_convo_info(sid, convo.id) or {}
                rows.append(
                    {
                        "conversation_id": convo.id,
                        "speaker": sid,
                        "n_initial": len(record["initial"][name]),
                        "n_final": len(record["final"][name]),
                        "agreement_gain": record["agreement"]["gain"],
                        "share_from_mediator": info.get("claims_share_from_mediator"),
                        "share_from_partner": info.get("claims_share_from_partner"),
                        "share_new_common_ground": info.get("claims_share_new_common_ground"),
                    }
                )
        return pd.DataFrame(rows)
