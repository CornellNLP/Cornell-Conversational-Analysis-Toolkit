import unittest

import os
os.chdir("..")
from convokit import Utterance, Corpus, Speaker
os.chdir("./tests")

class CorpusMerge(unittest.TestCase):
    def test_dump_and_load_with_binary(self):
        """
        Dump a corpus containing speakers with binary metadata and utterances with binary metadata
        Check that dumped corpus is successfully loaded with the same data
        """

        speaker_byte_arr1 = bytearray([120, 3, 255, 0, 100])
        speaker_byte_arr2 = bytearray([110, 3, 255, 90])
        utt_byte_arr1 = bytearray([99, 44, 33])
        utt_byte_arr2 = bytearray([110, 200, 220, 28])

        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice", meta={'speaker_binary_data': speaker_byte_arr1,
                                                                                'index': 99}), meta={'utt_binary_data': utt_byte_arr1}),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob", meta={'speaker_binary_data': speaker_byte_arr2}), meta={'utt_binary_data': utt_byte_arr2}),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ])


        alice = corpus1.utterances["0"].speaker
        bob = corpus1.utterances["1"].speaker

        corpus1.dump('test_corpus', './')
        corpus2 = Corpus(filename="test_corpus")

        alice2 = corpus2.utterances["0"].speaker
        bob2 = corpus2.utterances["1"].speaker

        self.assertEqual(alice.meta, alice2.meta)
        self.assertEqual(corpus1.utterances["0"].meta, corpus2.utterances["0"].meta)
        self.assertEqual(bob.meta, bob2.meta)
        self.assertEqual(corpus1.utterances["1"].meta, corpus2.utterances["1"].meta)

    def test_partial_loading(self):
        speaker_byte_arr1 = bytearray([120, 3, 255, 0, 100])
        speaker_byte_arr2 = bytearray([110, 3, 255, 90])
        utt_byte_arr1 = bytearray([99, 44, 33])
        utt_byte_arr2 = bytearray([110, 200, 220, 28])

        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice", meta={'speaker_binary_data': speaker_byte_arr1}), meta={'utt_binary_data': utt_byte_arr1}),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob", meta={'speaker_binary_data': speaker_byte_arr2}), meta={'utt_binary_data': utt_byte_arr2}),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ])

        corpus1.dump('test_corpus', './')

        corpus2 = Corpus(filename="test_corpus", utterance_start_index=0, utterance_end_index=1)

        self.assertEqual(len(list(corpus2.iter_utterances())), 2)
        self.assertEqual(corpus1.get_utterance("0"), corpus2.get_utterance("0"))
        self.assertEqual(corpus1.get_utterance("1"), corpus2.get_utterance("1"))

    def test_partial_load_start_idx_specified_only(self):
        speaker_byte_arr1 = bytearray([120, 3, 255, 0, 100])
        speaker_byte_arr2 = bytearray([110, 3, 255, 90])
        utt_byte_arr1 = bytearray([99, 44, 33])
        utt_byte_arr2 = bytearray([110, 200, 220, 28])

        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice", meta={'speaker_binary_data': speaker_byte_arr1}), meta={'utt_binary_data': utt_byte_arr1}),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob", meta={'speaker_binary_data': speaker_byte_arr2}), meta={'utt_binary_data': utt_byte_arr2}),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ])

        corpus1.dump('test_corpus', './')

        corpus2 = Corpus(filename="test_corpus", utterance_start_index=1)

        self.assertEqual(len(list(corpus2.iter_utterances())), 2)
        self.assertEqual(corpus1.get_utterance("1"), corpus2.get_utterance("1"))
        self.assertEqual(corpus1.get_utterance("2"), corpus2.get_utterance("2"))

    def test_partial_load_end_idx_specified_only(self):
        speaker_byte_arr1 = bytearray([120, 3, 255, 0, 100])
        speaker_byte_arr2 = bytearray([110, 3, 255, 90])
        utt_byte_arr1 = bytearray([99, 44, 33])
        utt_byte_arr2 = bytearray([110, 200, 220, 28])

        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice", meta={'speaker_binary_data': speaker_byte_arr1}), meta={'utt_binary_data': utt_byte_arr1}),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob", meta={'speaker_binary_data': speaker_byte_arr2}), meta={'utt_binary_data': utt_byte_arr2}),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ])

        corpus1.dump('test_corpus', './')

        corpus2 = Corpus(filename="test_corpus", utterance_end_index=0)

        self.assertEqual(len(list(corpus2.iter_utterances())), 1)
        self.assertEqual(corpus1.get_utterance("0"), corpus2.get_utterance("0"))

    def test_partial_load_invalid_start_index(self):
        speaker_byte_arr1 = bytearray([120, 3, 255, 0, 100])
        speaker_byte_arr2 = bytearray([110, 3, 255, 90])
        utt_byte_arr1 = bytearray([99, 44, 33])
        utt_byte_arr2 = bytearray([110, 200, 220, 28])

        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice", meta={'speaker_binary_data': speaker_byte_arr1}), meta={'utt_binary_data': utt_byte_arr1}),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob", meta={'speaker_binary_data': speaker_byte_arr2}), meta={'utt_binary_data': utt_byte_arr2}),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ])

        corpus1.dump('test_corpus', './')

        corpus2 = Corpus(filename="test_corpus", utterance_start_index=99)

        self.assertEqual(len(list(corpus2.iter_utterances())), 0)

    def test_partial_load_invalid_end_index(self):
        speaker_byte_arr1 = bytearray([120, 3, 255, 0, 100])
        speaker_byte_arr2 = bytearray([110, 3, 255, 90])
        utt_byte_arr1 = bytearray([99, 44, 33])
        utt_byte_arr2 = bytearray([110, 200, 220, 28])

        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice", meta={'speaker_binary_data': speaker_byte_arr1}), meta={'utt_binary_data': utt_byte_arr1}),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob", meta={'speaker_binary_data': speaker_byte_arr2}), meta={'utt_binary_data': utt_byte_arr2}),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ])

        corpus1.dump('test_corpus', './')

        corpus2 = Corpus(filename="test_corpus", utterance_end_index=-1)

        self.assertEqual(len(list(corpus2.iter_utterances())), 0)


if __name__ == '__main__':
    unittest.main()