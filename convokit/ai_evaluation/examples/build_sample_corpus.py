"""
Rebuild the sample corpus bundled with the examples (``sample_corpus/``).

This script is only needed to regenerate ``sample_corpus/``; the example notebooks load the
bundled copy and do not need to run it.

The sample corpus contains the completed cohorts of one simulated ConvoArena experiment
(one mediator and two simulated participants per cohort, one cohort per stance pair) for each
of three mediators: a greeting baseline, a paraphrasing baseline, and a constructive mediator.
The experiments are converted to a ConvoKit AI corpus with ``export_to_corpus`` from ``convokit.simulation``, and each conversation's
metadata is extended with ``mediator`` (the mediator label), ``topic``, ``statement`` (the
debate statement) and ``pre_survey`` (speaker id -> the reason each participant gave for their
stance before the conversation).

Usage::

    python build_sample_corpus.py /path/to/simulation_results

This needs ConvoKit's simulation dependencies (``pip install "convokit[simulation]"``).

The ``simulation_results`` folder holds the output of the simulation runs. The script reads
the sub-folders listed in ``SOURCES`` (``baseline_mediators/track 1/clean`` and
``final_pool_batches/batch1/Track 1/clean``, relative to ``simulation_results``), and each of
those sub-folders must contain:

- ``runs.json``: a list of runs, each with ``mediator`` (the mediator's submission file name)
  and ``experiments`` (a list of objects with ``experiment_id`` and ``completed_cohorts``, the
  ids of the cohorts that finished);
- ``<experiment_id>/export.json``: the ConvoArena export of that experiment, whose
  ``cohortMap`` holds the cohorts and whose stages include ``chat-round-1`` (the chat, with the
  debate statement) and ``pre-survey-1`` (the pre-conversation survey, whose answer
  ``pre_q1_a.5`` is the participant's reason for their stance);
- ``<experiment_id>/meta.json``: an object whose ``topic`` is the debate topic.

For each mediator, the first experiment in ``runs.json`` with completed cohorts is used.
"""

import json
import os
import sys

if len(sys.argv) != 2:
    sys.exit("usage: python build_sample_corpus.py /path/to/simulation_results")

from convokit.simulation import export_to_corpus

SIM = sys.argv[1].rstrip("/")
SOURCES = {  # mediator label -> (submission file, folder whose runs.json lists its experiments)
    "greeting": ("greeting_mediator.yaml", f"{SIM}/baseline_mediators/track 1/clean"),
    "paraphrase": (
        "paraphrase_mediator.yaml",
        f"{SIM}/baseline_mediators/track 1/clean",
    ),
    "constructive": (
        "[default] constructive mediator.yaml",
        f"{SIM}/final_pool_batches/batch1/Track 1/clean",
    ),
}
OUT = os.path.dirname(os.path.abspath(__file__))


def load_one(mediator, folder):
    """
    Load the export of the first experiment in ``folder`` run by ``mediator`` that has completed
    cohorts, keeping only those cohorts.

    :param mediator: the mediator's submission file name, as listed in ``runs.json``
    :param folder: folder containing ``runs.json`` and one sub-folder per experiment
    :return: the trimmed export and the experiment's topic
    """
    runs = json.load(open(f"{folder}/runs.json"))
    experiment = next(
        e
        for r in runs
        if r["mediator"] == mediator
        for e in r["experiments"]
        if e.get("completed_cohorts")
    )
    eid, cids = experiment["experiment_id"], experiment["completed_cohorts"]
    export = json.load(open(f"{folder}/{eid}/export.json"))
    export["cohortMap"] = {cid: export["cohortMap"][cid] for cid in cids}
    topic = json.load(open(f"{folder}/{eid}/meta.json"))["topic"]
    return export, topic


def statement(export):
    """
    Get the debate statement shown with the chat, without its ``Statement: "..."`` wrapping.

    :param export: a ConvoArena experiment export
    :return: the statement text
    """
    text = export["stageMap"]["chat-round-1"]["descriptions"]["primaryText"].strip()
    return text.removeprefix("Statement:").strip().strip('"')


exports, extra = [], {}
for label, (mediator, folder) in SOURCES.items():
    export, topic = load_one(mediator, folder)
    exports.append(export)
    for cid, cohort in export["cohortMap"].items():
        answers = cohort["dataMap"]["pre-survey-1"]["participantAnswerMap"]
        extra[cid] = {
            "mediator": label,
            "topic": topic,
            "statement": statement(export),
            "pre_survey": {sid: a["pre_q1_a.5"]["answer"] for sid, a in answers.items()},
        }

corpus = export_to_corpus(exports)
for convo in corpus.iter_conversations():
    for key, value in extra[convo.id].items():
        convo.meta[key] = value
corpus.dump("sample_corpus", base_path=OUT, force_version=1)
corpus.print_summary_stats()
