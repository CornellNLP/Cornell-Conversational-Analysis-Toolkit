import unittest

from convokit import Corpus, Speaker, Utterance
from convokit.ranker import Ranker


def construct_tied_corpus():
    """Four utterances whose scores contain a three-way and a two-way tie."""
    scores = {"0": 10, "1": 10, "2": 5, "3": 10}
    return Corpus(
        utterances=[
            Utterance(
                id=uid,
                text=f"utterance {uid}",
                speaker=Speaker(id=f"s{uid}"),
                meta={"score": score},
            )
            for uid, score in scores.items()
        ]
    )


class TestRankerTies(unittest.TestCase):
    def setUp(self):
        self.corpus = construct_tied_corpus()
        self.ranker = Ranker(
            obj_type="utterance",
            score_func=lambda utt: utt.meta["score"],
            score_attribute_name="score_value",
            rank_attribute_name="rank_value",
        )

    def test_transform_gives_tied_objects_the_same_rank(self):
        self.ranker.transform(self.corpus)
        ranks = {utt.id: utt.meta["rank_value"] for utt in self.corpus.iter_utterances()}
        # Three utterances score 10 and one scores 5. Competition ranking gives the
        # tied objects the same rank and skips the ones they consumed.
        assert ranks["0"] == ranks["1"] == ranks["3"] == 1
        assert ranks["2"] == 4

    def test_transform_objs_gives_tied_objects_the_same_rank(self):
        objs = list(self.corpus.iter_utterances())
        self.ranker.transform_objs(objs)
        ranks = {utt.id: utt.meta["rank_value"] for utt in objs}
        assert ranks["0"] == ranks["1"] == ranks["3"] == 1
        assert ranks["2"] == 4

    def test_distinct_scores_still_rank_consecutively(self):
        corpus = Corpus(
            utterances=[
                Utterance(id="a", text="a", speaker=Speaker(id="sa"), meta={"score": 3}),
                Utterance(id="b", text="b", speaker=Speaker(id="sb"), meta={"score": 2}),
                Utterance(id="c", text="c", speaker=Speaker(id="sc"), meta={"score": 1}),
            ]
        )
        self.ranker.transform(corpus)
        ranks = {utt.id: utt.meta["rank_value"] for utt in corpus.iter_utterances()}
        assert ranks == {"a": 1, "b": 2, "c": 3}


if __name__ == "__main__":
    unittest.main()
