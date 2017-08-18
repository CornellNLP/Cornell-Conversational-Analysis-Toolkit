"""Command-line interface for computing all features."""

from collections import defaultdict
import argparse
import os
import json
import convokit

def command_line_main():
    parser = argparse.ArgumentParser(description="Social features toolkit.")
    parser.add_argument("filename", help="file to process")
    parser.add_argument("--groups", dest="groups",
            help="file containing groups")
    args = parser.parse_args()

    corpus = convokit.Corpus(filename=args.filename)

    ### coordination
    coord = convokit.Coordination(corpus)

    # pairwise scores
    pairwise_scores = coord.pairwise_scores(corpus.speaking_pairs(
        user_names_only=True))
    pairwise_scores_s = {"'" + s + "' -> '" + t + "'":
            v for (s, t), v in pairwise_scores.items()}
    _, pairwise_average_scores, _, _, _ = coord.score_report(pairwise_scores_s)

    out = {}
    out["pairwise"] = pairwise_scores_s
    out["meta"] = {
        "pairwise-averages": pairwise_average_scores
    }

    # user scores
    coord_to = defaultdict(lambda: defaultdict(list))
    coord_from = defaultdict(lambda: defaultdict(list))
    for (speaker, target), m in pairwise_scores.items():
        for cat, value in m.items():
            coord_to[speaker][cat].append(value)
            coord_from[target][cat].append(value)
    coord_to_avg, coord_from_avg = {}, {}
    for user, m in coord_to.items():
        coord_to[user] = { cat: sum(values) / len(values) for cat, values in
                m.items() }
    for user, m in coord_from.items():
        coord_from[user] = { cat: sum(values) / len(values) for cat, values in
                m.items() }
    user_scores = {}
    for user in coord_to.keys() | coord_from.keys():
        user_scores[user] = {}
        if user in coord_to:
            user_scores[user]["out"] = coord_to[user]
            user_scores[user]["out-average"] = sum(coord_to[user].values()) / \
                    len(coord_to[user].values())
        if user in coord_from:
            user_scores[user]["in"] = coord_from[user]
            user_scores[user]["in-average"] = sum(coord_from[user].values()) / \
                    len(coord_from[user].values())
    out["user"] = user_scores

    # group scores
    if args.groups is not None:
        groups = json.load(open(args.groups, "r"))
        scores = {}
        score_reports = {}
        for name, (a, b) in groups.items():
            scores[name] = coord.score(a, b)
            score_reports[name] = coord.score_report(scores[name])
        out["group"] = { name: scores[name] for name in scores }
        out["meta"]["group-averages"] = {
            name: {
                "average-by-marker": mkr,
                "aggregate-1": agg1,
                "aggregate-2": agg2,
                "aggregate-3": agg3
            }
            for name, (_, mkr, agg1, agg2, agg3) in score_reports.items()
        }

    json.dump(out, open(os.path.splitext(args.filename)[0] + "-out.json", "w"),
            indent=2, sort_keys=True)

