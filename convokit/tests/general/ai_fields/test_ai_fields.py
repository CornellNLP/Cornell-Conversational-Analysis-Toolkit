import json
import os
import shutil
import tempfile
import unittest

import pandas as pd

from convokit.model import Corpus, PrivateAssistant, Speaker, Support, Utterance
from convokit.tests.test_utils import reload_corpus_in_db_mode


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeClient:
    """LLM client stand-in that records prompts and returns a fixed reply."""

    def __init__(self, reply="generated"):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return FakeResponse(self.reply)


def build_ai_corpus():
    alice = Speaker(id="alice")
    bot = Speaker(
        id="bot",
        is_ai=True,
        ai_meta={"role": "public assistant", "config": {"prompt": "Mediate.", "model": "x"}},
    )
    utts = [
        Utterance(
            id="u0", speaker=alice, conversation_id="u0", reply_to=None, timestamp=0, text="hi"
        ),
        Utterance(
            id="u1",
            speaker=bot,
            conversation_id="u0",
            reply_to="u0",
            timestamp=1,
            text="hello",
            ai_meta={"config": {"prompt": "Mediate."}},
        ),
        Utterance(
            id="u2", speaker=alice, conversation_id="u0", reply_to="u1", timestamp=2, text="bye"
        ),
    ]
    corpus = Corpus(utterances=utts, meta={"name": "toy", "ai_meta": {"simulation_config": 1}})
    convo = corpus.get_conversation("u0")
    convo.alias = {"alice": "Bear", "bot": "Mediator"}
    convo.private_assistants = [
        PrivateAssistant(
            id="pa",
            config={"prompt": "Help Bear."},
            speakers=["alice"],
            conversation_id="u0",
            supports=[
                Support(
                    id="s0",
                    text="be nice",
                    reply_to="u1",
                    draft="ok",
                    private_assistant_id="pa",
                    timestamp=1,
                )
            ],
        )
    ]
    return corpus


class TestAIFields(unittest.TestCase):
    BACKEND = "mem"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def build(self):
        corpus = build_ai_corpus()
        return corpus if self.BACKEND == "mem" else reload_corpus_in_db_mode(corpus)

    def load(self, path):
        return Corpus(filename=path, backend=self.BACKEND)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_construction(self):
        corpus = self.build()
        self.assertTrue(corpus.has_ai)
        self.assertEqual(corpus.ai_meta, {"simulation_config": 1})
        self.assertEqual(dict(corpus.meta), {"name": "toy"})
        self.assertTrue(corpus.get_speaker("bot").is_ai)
        self.assertFalse(corpus.get_speaker("alice").is_ai)
        self.assertEqual(corpus.get_speaker("bot").ai_meta["role"], "public assistant")
        convo = corpus.get_conversation("u0")
        self.assertEqual(convo.get_ai_speakers(), ["bot"])
        self.assertEqual([s.id for s in convo.supports], ["s0"])
        self.assertEqual([a.id for a in corpus.iter_private_assistants()], ["pa"])
        self.assertEqual([s.id for s in corpus.iter_supports()], ["s0"])
        self.assertEqual(corpus.get_private_assistant("pa").speakers, ["alice"])

    def test_ai_fields_in_meta_move_to_attributes(self):
        speaker = Speaker(id="bot", meta={"is_ai": True, "ai_meta": {"role": "x"}, "other": 1})
        self.assertTrue(speaker.is_ai)
        self.assertEqual(speaker.ai_meta, {"role": "x"})
        self.assertEqual(dict(speaker.meta), {"other": 1})
        utt = Utterance(id="u", speaker=speaker, meta={"ai_meta": {"config": {}}, "k": 2})
        self.assertEqual(utt.ai_meta, {"config": {}})
        self.assertEqual(dict(utt.meta), {"k": 2})

    def test_transcript(self):
        convo = self.build().get_conversation("u0")
        self.assertEqual(convo.get_transcript(), "Bear: hi\nMediator: hello\nBear: bye")
        self.assertEqual(
            convo.get_transcript(supports=True),
            "Bear: hi\nMediator: hello\n    PrivateAssistant[pa] -> Bear:\n"
            "        draft: ok\n        pa: be nice\nBear: bye",
        )
        self.assertEqual(convo.get_utterance("u1").get_transcript(), "Bear: hi\nMediator: hello")

    def test_dump_and_load_roundtrip(self):
        self.build().dump("ai", base_path=self.tmp)
        path = os.path.join(self.tmp, "ai")

        # on disk, the AI fields are regular metadata
        with open(os.path.join(path, "speakers.json")) as f:
            speakers = json.load(f)
        self.assertTrue(speakers["bot"]["meta"]["is_ai"])
        with open(os.path.join(path, "corpus.json")) as f:
            self.assertTrue(json.load(f)["has_ai"])

        corpus = self.load(path)
        self.assertTrue(corpus.has_ai)
        self.assertEqual(corpus.ai_meta, {"simulation_config": 1})
        self.assertEqual(dict(corpus.meta), {"name": "toy"})
        bot = corpus.get_speaker("bot")
        self.assertTrue(bot.is_ai)
        self.assertNotIn("is_ai", bot.meta)
        self.assertNotIn("ai_meta", corpus.meta_index.get_index("speaker"))
        convo = corpus.get_conversation("u0")
        self.assertEqual(convo.alias, {"alice": "Bear", "bot": "Mediator"})
        self.assertEqual(convo.private_assistants[0].supports[0].text, "be nice")
        self.assertEqual(corpus.get_utterance("u1").ai_meta, {"config": {"prompt": "Mediate."}})

        # dumping again leaves no AI fields behind in memory
        corpus.dump("ai2", base_path=self.tmp)
        self.assertNotIn("is_ai", corpus.get_speaker("bot").meta)
        self.assertTrue(self.load(os.path.join(self.tmp, "ai2")).get_speaker("bot").is_ai)

    def test_plain_corpus_unchanged(self):
        speaker = Speaker(id="a", meta={"role": "judge"})
        corpus = Corpus(utterances=[Utterance(id="u", speaker=speaker, text="x")])
        self.assertFalse(corpus.has_ai)
        corpus.dump("plain", base_path=self.tmp)
        path = os.path.join(self.tmp, "plain")
        with open(os.path.join(path, "corpus.json")) as f:
            self.assertEqual(json.load(f), {})
        with open(os.path.join(path, "speakers.json")) as f:
            self.assertEqual(json.load(f)["a"]["meta"], {"role": "judge"})
        # "role" is only treated as a legacy AI field in corpora with AI fields
        loaded = self.load(path)
        self.assertEqual(dict(loaded.get_speaker("a").meta), {"role": "judge"})
        self.assertEqual(loaded.get_speaker("a").ai_meta, {})

    def test_legacy_format(self):
        self.build().dump("legacy", base_path=self.tmp)
        path = os.path.join(self.tmp, "legacy")
        with open(os.path.join(path, "conversations.json")) as f:
            convos = json.load(f)
        ai_meta = convos["u0"]["meta"]["ai_meta"]
        ai_meta["assistants"] = ai_meta.pop("private_assistants")
        with open(os.path.join(path, "conversations.json"), "w") as f:
            json.dump(convos, f)
        with open(os.path.join(path, "speakers.json")) as f:
            speakers = json.load(f)
        speakers["alice"]["meta"]["role"] = "participant"
        with open(os.path.join(path, "speakers.json"), "w") as f:
            json.dump(speakers, f)
        with open(os.path.join(path, "supports.json"), "w") as f:
            json.dump(
                [{"id": "s1", "text": "t", "assistant_id": "other", "conversation_id": "u0"}], f
            )

        corpus = self.load(path)
        convo = corpus.get_conversation("u0")
        self.assertEqual([a.id for a in convo.private_assistants], ["pa"])
        self.assertEqual(sorted(s.id for s in convo.supports), ["s0", "s1"])
        alice = corpus.get_speaker("alice")
        self.assertEqual(alice.ai_meta, {"role": "participant"})
        self.assertNotIn("role", alice.meta)

    def test_utterance_role_survives_reload(self):
        # e.g. MediatorRedirection.label_roles writes utt.meta["role"]
        corpus = self.build()
        corpus.get_utterance("u1").meta["role"] = "mediator"
        corpus.dump("roles", base_path=self.tmp)
        loaded = self.load(os.path.join(self.tmp, "roles"))
        self.assertEqual(loaded.get_utterance("u1").meta["role"], "mediator")

    def test_filter_reindex_merge(self):
        filtered = Corpus.filter_utterances(self.build(), lambda u: u.id != "u2")
        self.assertEqual(filtered.ai_meta, {"simulation_config": 1})
        self.assertEqual(filtered.get_conversation("u0").alias["alice"], "Bear")
        self.assertTrue(filtered.get_speaker("bot").is_ai)

        reindexed = Corpus.reindex_conversations(self.build(), ["u0"], verbose=False)
        self.assertEqual(reindexed.ai_meta, {"simulation_config": 1})
        self.assertEqual(
            [a.id for a in reindexed.get_conversation("u0").private_assistants], ["pa"]
        )

        other = Corpus(
            utterances=[
                Utterance(
                    id="v0",
                    speaker=Speaker(id="bot", is_ai=True, ai_meta={"extra": 1}),
                    conversation_id="v0",
                    text="x",
                )
            ],
            meta={"ai_meta": {"other": 2}},
        )
        merged = Corpus.merge(self.build(), other, warnings=False)
        self.assertEqual(merged.ai_meta, {"simulation_config": 1, "other": 2})
        self.assertEqual(merged.get_speaker("bot").ai_meta["extra"], 1)
        self.assertEqual(merged.get_speaker("bot").ai_meta["role"], "public assistant")
        self.assertEqual(merged.get_conversation("u0").alias["bot"], "Mediator")

    def test_from_pandas(self):
        utts = pd.DataFrame(
            {
                "id": ["u0"],
                "speaker": ["bot"],
                "timestamp": [0],
                "conversation_id": ["u0"],
                "reply_to": [None],
                "text": ["hi"],
                "meta.ai_meta": [{"config": {"prompt": "p"}}],
            }
        )
        speakers = pd.DataFrame({"id": ["bot"], "meta.is_ai": [True]}).set_index("id")
        corpus = Corpus.from_pandas(utts, speakers_df=speakers)
        self.assertTrue(corpus.get_speaker("bot").is_ai)
        self.assertNotIn("is_ai", corpus.get_speaker("bot").meta)
        self.assertEqual(corpus.get_utterance("u0").ai_meta, {"config": {"prompt": "p"}})

    def test_generate(self):
        corpus = self.build()
        convo = corpus.get_conversation("u0")
        client = FakeClient("reply")
        utt = corpus.get_speaker("bot").generate(convo, append=True, client=client, id="u3")
        self.assertEqual(corpus.get_utterance("u3").text, "reply")
        self.assertEqual(utt.reply_to, "u2")
        self.assertIn("as Mediator", client.prompts[0])

        support = convo.get_private_assistant("pa").generate(
            convo, draft="d", append=True, client=client, id="s9"
        )
        self.assertEqual(support.reply_to, "u3")
        self.assertIn("s9", [s.id for s in corpus.get_conversation("u0").supports])
        with self.assertRaises(ValueError):
            corpus.get_speaker("alice").generate(convo, client=client)


class TestAIFieldsDB(TestAIFields):
    """The same tests with the db backend, plus restoring the AI fields with reconnect_to_db."""

    BACKEND = "db"

    def reconnect(self, corpus):
        return Corpus.reconnect_to_db(corpus.id)

    def assert_ai_fields(self, corpus, support_ids=("s0",)):
        self.assertTrue(corpus.has_ai)
        self.assertEqual(corpus.ai_meta, {"simulation_config": 1})
        self.assertEqual(dict(corpus.meta), {"name": "toy"})
        bot = corpus.get_speaker("bot")
        self.assertTrue(bot.is_ai)
        self.assertEqual(bot.ai_meta["role"], "public assistant")
        self.assertNotIn("is_ai", bot.meta)
        self.assertFalse(corpus.get_speaker("alice").is_ai)
        self.assertEqual(corpus.get_utterance("u1").ai_meta["config"], {"prompt": "Mediate."})
        convo = corpus.get_conversation("u0")
        self.assertEqual(convo.alias, {"alice": "Bear", "bot": "Mediator"})
        self.assertEqual(sorted(s.id for s in convo.supports), sorted(support_ids))
        self.assertEqual(convo.get_private_assistant("pa").speakers, ["alice"])

    def test_reconnect_after_load(self):
        corpus = self.build()
        self.assert_ai_fields(corpus)
        self.assert_ai_fields(self.reconnect(corpus))

    def test_reconnect_after_building_in_db(self):
        # components created directly in a db-backed corpus
        source = build_ai_corpus()
        corpus = Corpus(utterances=list(source.iter_utterances()), backend="db")
        convo = corpus.get_conversation("u0")
        convo.ai_meta = source.get_conversation("u0").ai_meta
        corpus.ai_meta = {"simulation_config": 1}
        corpus.meta["name"] = "toy"
        self.assert_ai_fields(self.reconnect(corpus))

    def test_reconnect_after_changes(self):
        corpus = self.build()
        corpus.get_speaker("alice").ai_meta = {"role": "participant"}
        corpus.get_utterance("u2").supports = [Support(id="s2", text="t", reply_to="u2")]
        convo = corpus.get_conversation("u0")
        convo.get_private_assistant("pa").generate(
            convo, append=True, client=FakeClient("tip"), id="s3"
        )
        corpus.get_speaker("bot").generate(convo, append=True, client=FakeClient("m"), id="u3")

        reconnected = self.reconnect(corpus)
        self.assert_ai_fields(reconnected, support_ids=("s0", "s2", "s3"))
        self.assertEqual(reconnected.get_speaker("alice").ai_meta, {"role": "participant"})
        self.assertEqual(reconnected.get_utterance("u3").text, "m")
        self.assertEqual(reconnected.get_utterance("u3").ai_meta["config"]["prompt"], "Mediate.")

    def test_reconnect_after_filter_conversations_by(self):
        corpus = self.build()
        corpus.filter_conversations_by(lambda c: True)
        self.assert_ai_fields(self.reconnect(corpus))

    def test_reconnect_plain_corpus(self):
        speaker = Speaker(id="a", meta={"role": "judge"})
        corpus = Corpus(utterances=[Utterance(id="u", speaker=speaker, text="x")], backend="db")
        reconnected = self.reconnect(corpus)
        self.assertFalse(reconnected.has_ai)
        self.assertEqual(dict(reconnected.get_speaker("a").meta), {"role": "judge"})
        self.assertFalse(
            corpus.backend_mapper.has_data_for_component("meta", corpus._ai_backend_key)
        )


if __name__ == "__main__":
    unittest.main()
