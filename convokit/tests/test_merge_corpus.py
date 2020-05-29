import unittest
from convokit.model import Utterance, Speaker, Corpus

class CorpusMerge(unittest.TestCase):
    def test_no_overlap(self):
        """
        Basic merge: no overlap in utterance id
        """
        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ])

        corpus2 = Corpus(utterances = [
            Utterance(id="3", text="i like pie", speaker=Speaker(id="delta")),
            Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
            Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
        ])

        merged = corpus1.merge(corpus2)
        self.assertEqual(len(list(merged.iter_utterances())), 6)
        self.assertEqual(len(list(merged.iter_speakers())), 6)
        self.assertEqual(len(list(corpus1.iter_utterances())), 3)
        self.assertEqual(len(list(corpus2.iter_utterances())), 3)

    def test_with_overlap(self):
        """
        Basic merge: with overlap in utterance id (but utterance has same data & metadata)
        """
        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ])

        corpus2 = Corpus(utterances = [
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
            Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
            Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
        ])

        merged = corpus1.merge(corpus2)
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_speakers())), 5)
        self.assertEqual(len(list(corpus1.iter_utterances())), 3)
        self.assertEqual(len(list(corpus2.iter_utterances())), 3)

    def test_overlap_diff_data(self):
        """
        Merge with overlap in utterance id and utterance has diff data but same metadata

        Warning should be printed. Original utterance data should be preserved.
        """
        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ])

        corpus2 = Corpus(utterances = [
            Utterance(id="2", text="this is a test2", speaker=Speaker(id="candace")),
            Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
            Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
        ])

        merged = corpus1.merge(corpus2)
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_speakers())), 5)
        self.assertEqual(len(list(corpus1.iter_utterances())), 3)
        self.assertEqual(len(list(corpus2.iter_utterances())), 3)

        self.assertEqual(merged.get_utterance("2").text, "this is a test")
        self.assertEqual(merged.get_utterance("2").speaker, Speaker(id="charlie"))

    def test_overlap_diff_metadata(self):
        """
        Merge with overlap in utterance id and utterance has same data but diff metadata

        Second corpus utterance metadata should override if the keys are the same.
        """
        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie"), meta={'hey': 'jude', 'the': 'beatles'}),
        ])

        corpus2 = Corpus(utterances = [
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie"),
                      meta={'hey': 'jude', 'the': 'ringo', 'let it': 'be'}),
            Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
            Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
        ])

        merged = corpus1.merge(corpus2)
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_speakers())), 5)

        self.assertEqual(len(merged.get_utterance("2").meta), 3)
        self.assertEqual(merged.get_utterance("2").meta['the'], 'ringo')

    def test_overlap_convo_metadata(self):
        """
        Merge with overlap in conversation with metadata differences.

        Expect second corpus convo metadata to override if keys are the same
        """
        corpus1 = Corpus(utterances = [
            Utterance(id="0", conversation_id='convo1', text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", conversation_id='convo1', text="my name is bob", speaker=Speaker(id="bob")),
            Utterance(id="2", conversation_id='convo1', text="this is a test", speaker=Speaker(id="charlie")),
        ])

        corpus2 = Corpus(utterances = [
            Utterance(id="2", conversation_id='convo1', text="this is a test", speaker=Speaker(id="charlie")),
            Utterance(id="4", conversation_id='convo1', text="this is a sentence", speaker=Speaker(id="echo")),
            Utterance(id="5", conversation_id='convo1', text="goodbye", speaker=Speaker(id="foxtrot")),
        ])


        corpus1.get_conversation('convo1').add_meta('hey', 'jude')
        corpus1.get_conversation('convo1').add_meta('hello', 'world')

        corpus2.get_conversation('convo1').add_meta('hey', 'jude')
        corpus2.get_conversation('convo1').add_meta('hello', 'food')
        corpus2.get_conversation('convo1').add_meta('what', 'a mood')

        merged = corpus1.merge(corpus2)
        self.assertEqual(len(merged.get_conversation('convo1').meta), 3)
        self.assertEqual(merged.get_conversation('convo1').meta['hello'], 'food')

    def test_corpus_metadata(self):
        """
        Merge with overlap in corpus metadata

        Expect second corpus metadata to override if keys are the same
        """
        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ])

        corpus2 = Corpus(utterances = [
            Utterance(id="3", text="i like pie", speaker=Speaker(id="delta")),
            Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
            Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
        ])

        corpus1.add_meta('politeness', 0.95)
        corpus1.add_meta('toxicity', 0.8)

        corpus2.add_meta('toxicity', 0.9)
        corpus2.add_meta('paggro', 1.0)

        merged = corpus1.merge(corpus2)
        self.assertEqual(len(merged.meta), 3)
        self.assertEqual(merged.meta['toxicity'], 0.9)

    def test_add_utterance(self):
        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie"), meta={'hey': 'jude', 'hello': 'world'}),
        ])

        utts = [
            Utterance(id="1", text="i like pie", speaker=Speaker(id="delta")),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie"), meta={'hello': 'food', 'what': 'a mood'}),
            Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
        ]
        added = corpus1.add_utterances(utts)

        self.assertEqual(len(list(added.iter_utterances())), 4)
        self.assertEqual(len(added.get_utterance("2").meta), 3)
        self.assertEqual(added.get_utterance("2").meta['hello'], 'food')


if __name__ == '__main__':
    unittest.main()
