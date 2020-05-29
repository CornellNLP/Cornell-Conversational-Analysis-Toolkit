import unittest
from convokit.model import Utterance, Speaker, Corpus

class CorpusTraversal(unittest.TestCase):

    def setUp(self) -> None:
        """
        Basic Conversation tree (left to right within subtree => earliest to latest)
                   0
            1      2      3
          4 5 6   7 8     9
        10                11
        """
        self.corpus = Corpus(utterances = [
            Utterance(id="0", reply_to=None, conversation_id="0", speaker=Speaker(id="alice"), timestamp=0),

            Utterance(id="2", reply_to="0", conversation_id="0", speaker=Speaker(id="alice"), timestamp=2),
            Utterance(id="1", reply_to="0", conversation_id="0", speaker=Speaker(id="alice"), timestamp=1),
            Utterance(id="3", reply_to="0", conversation_id="0", speaker=Speaker(id="alice"), timestamp=3),

            Utterance(id="4", reply_to="1", conversation_id="0", speaker=Speaker(id="alice"), timestamp=4),
            Utterance(id="5", reply_to="1", conversation_id="0", speaker=Speaker(id="alice"), timestamp=5),
            Utterance(id="6", reply_to="1", conversation_id="0", speaker=Speaker(id="alice"), timestamp=6),

            Utterance(id="7", reply_to="2", conversation_id="0", speaker=Speaker(id="alice"), timestamp=4),
            Utterance(id="8", reply_to="2", conversation_id="0", speaker=Speaker(id="alice"), timestamp=5),

            Utterance(id="9", reply_to="3", conversation_id="0", speaker=Speaker(id="alice"), timestamp=4),

            Utterance(id="10", reply_to="4", conversation_id="0", speaker=Speaker(id="alice"), timestamp=5),

            Utterance(id="11", reply_to="9", conversation_id="0", speaker=Speaker(id="alice"), timestamp=10),

            Utterance(id="other", reply_to=None, conversation_id="other", speaker=Speaker(id="alice"), timestamp=99)
        ])
        self.corpus.get_conversation("0").meta['hey'] = 'jude'
        self.corpus.meta['foo'] = 'bar'

    def test_broken_convos(self):
        """
        Test basic meta functions
        """

        corpus1 = Corpus(utterances = [
            Utterance(id="0", text="hello world", reply_to=None, speaker=Speaker(id="alice"), timestamp=0),
            Utterance(id="1", text="my name is bob", reply_to="0", speaker=Speaker(id="bob"), timestamp=2),
            Utterance(id="2", text="this is a test", reply_to="1", speaker=Speaker(id="charlie"), timestamp=1),
            Utterance(id="3", text="hello world 2", reply_to=None, speaker=Speaker(id="alice2"), timestamp=0),
        ])

        corpus2 = Corpus(utterances = [
            Utterance(id="0", text="hello world", reply_to=None, speaker=Speaker(id="alice"), timestamp=0),
            Utterance(id="1", text="my name is bob", reply_to="0", speaker=Speaker(id="bob"), timestamp=2),
            Utterance(id="2", text="this is a test", reply_to="1", speaker=Speaker(id="charlie"), timestamp=1),
            Utterance(id="3", text="hello world 2", reply_to="9", speaker=Speaker(id="alice2"), timestamp=0),
        ])

        # test broken convo where there are multiple conversation_ids
        convo = corpus1.get_conversation(None)
        self.assertRaises(ValueError, lambda: list(convo.traverse("dfs", as_utterance=True)))

        # test broken convo where utterance replies to something not in Conversation
        convo = corpus2.get_conversation(None)
        self.assertRaises(ValueError, lambda: list(convo.traverse("dfs", as_utterance=True)))

    def test_bfs_traversal(self):
        convo = self.corpus.get_conversation("0")
        bfs_traversal = [utt.id for utt in convo.traverse("bfs", as_utterance=True)]
        self.assertEqual(bfs_traversal, [str(i) for i in range(12)])

    def test_dfs_traversal(self):
        convo = self.corpus.get_conversation("0")
        dfs_traversal = [utt.id for utt in convo.traverse("dfs", as_utterance=True)]
        self.assertEqual(dfs_traversal, [str(i) for i in [0, 1, 4, 10, 5, 6, 2, 7, 8, 3, 9, 11]])

    def test_postorder_traversal(self):
        convo = self.corpus.get_conversation("0")
        postorder_traversal = [utt.id for utt in convo.traverse("postorder", as_utterance=True)]
        self.assertEqual(postorder_traversal, ['10', '4', '5', '6', '1', '7', '8', '2', '11', '9', '3', '0'])

    def test_preorder_traversal(self):
        convo = self.corpus.get_conversation("0")
        preorder_traversal = [utt.id for utt in convo.traverse("preorder", as_utterance=True)]
        self.assertEqual(preorder_traversal, ['0', '1', '4', '10', '5', '6', '2', '7', '8', '3', '9', '11'])

    def test_subtree(self):
        convo = self.corpus.get_conversation("0")
        node = convo.get_subtree("1")
        self.assertEqual([node.utt.id for node in node.bfs_traversal()], ['1', '4', '5', '6', '10'])

    def test_conversation_id_to_leaf_paths(self):
        convo = self.corpus.get_conversation("0")
        paths = convo.get_root_to_leaf_paths()
        path_tuples = [tuple(utt.id for utt in paths[i]) for i in range(6)]
        self.assertIn(('0', '1', '4', '10'), path_tuples)
        self.assertIn(('0', '1', '5'), path_tuples)
        self.assertIn(('0', '1', '6'), path_tuples)
        self.assertIn(('0', '2', '7'), path_tuples)
        self.assertIn(('0', '2', '8'), path_tuples)
        self.assertIn(('0', '3', '9', '11'), path_tuples)

    def test_one_utt_convo(self):
        convo = self.corpus.get_conversation("other")
        self.assertEqual([utt.id for utt in convo.traverse('bfs')], ["other"])
        self.assertEqual([utt.id for utt in convo.traverse('dfs')], ["other"])
        self.assertEqual([utt.id for utt in convo.traverse('postorder')], ["other"])
        self.assertEqual([utt.id for utt in convo.traverse('preorder')], ["other"])

    def test_reindex_corpus(self):
        new_convo_conversation_ids = ['1', '2', '3']
        new_corpus = self.corpus.reindex_conversations(new_convo_conversation_ids)
        # checking for correct number of conversations and utterances
        self.assertEqual(len(list(new_corpus.iter_conversations())), 3)
        self.assertEqual(len(list(new_corpus.iter_utterances())), 11)

        # checking that corpus and conversation metadata was preserved
        for convo in new_corpus.iter_conversations():
            self.assertEqual(convo.meta['original_convo_meta'], self.corpus.get_conversation("0").meta)

        self.assertEqual(self.corpus.meta, new_corpus.meta)

    def test_reindex_corpus2(self):
        new_convo_conversation_ids = ['1', '2', '3']
        new_corpus = self.corpus.reindex_conversations(new_convo_conversation_ids,
                                                       preserve_convo_meta=False,
                                                       preserve_corpus_meta=False)
        # checking for correct number of conversations and utterances
        self.assertEqual(len(list(new_corpus.iter_conversations())), 3)
        self.assertEqual(len(list(new_corpus.iter_utterances())), 11)

        # checking that corpus and conversation metadata was preserved
        for convo in new_corpus.iter_conversations():
            self.assertEqual(convo.meta, dict())

        self.assertEqual(new_corpus.meta, dict())

if __name__ == '__main__':
    unittest.main()
