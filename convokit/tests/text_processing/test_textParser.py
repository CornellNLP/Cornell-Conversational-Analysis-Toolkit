import unittest

from convokit.text_processing.textParser import _process_sentence, _process_token, process_text
from convokit.tests.utils import buffalo_doc, fox_doc, fox_buffalo_doc, BUFFALO_TEXT, FOX_TEXT, FOX_BUFFALO_TEXT


class TestTextParser(unittest.TestCase):
    def test_process_token_tokenize_mode(self):
        doc = fox_doc()
        token = doc[3]

        expected = {
            'tok': 'fox'
        }
        actual = _process_token(token, mode='tokenize')
        self.assertDictEqual(expected, actual)


    def test_process_token_tag_mode(self):
        doc = fox_doc()
        token = doc[3]

        expected = {
            'tok': 'fox',
            'tag': 'NN'
        }
        actual = _process_token(token, mode='tag')
        self.assertDictEqual(expected, actual)

    def test_process_token_parse_mode(self):
        doc = fox_doc()
        token = doc[3]

        expected = {
            'tok': 'fox',
            'tag': 'NN',
            'dn': [],
            'up': 2,
            'dep': 'NN'
        }
        actual = _process_token(token, mode='parse', offset=2)
        self.assertDictEqual(expected, actual)

    def test_process_sentence_parse_mode(self):
        doc = fox_doc()

        expected = {
            'rt': 1,
            'toks': [
                {'tok': 'A', 'tag': 'DT', 'dep': 'det', 'up': 1, 'dn': []},
                {'tok': 'quick', 'tag': 'JJ', 'dep': 'amod', 'up': 1, 'dn': []},
                {'tok': 'brown', 'tag': 'JJ', 'dep': 'amod', 'up': 1, 'dn': []},
                {'tok': 'fox', 'tag': 'NN', 'dep': 'NN', 'up': 1, 'dn': []},
                {'tok': 'jumps', 'tag': 'NNS', 'dep': 'ROOT', 'dn': [-3, -2, -1, 0, 2]},
                {'tok': 'over', 'tag': 'IN', 'dep': 'prep', 'up': 1, 'dn': [5]},
                {'tok': 'the', 'tag': 'DT', 'dep': 'det', 'up': 5, 'dn': []},
                {'tok': 'lazy', 'tag': 'JJ', 'dep': 'amod', 'up': 5, 'dn': []},
                {'tok': 'dog', 'tag': 'NN', 'dep': 'pobj', 'up': 2, 'dn': [3, 4]}
            ]
        }
        actual = _process_sentence(next(doc.sents), mode='parse', offset=3)

        self.assertDictEqual(expected, actual)

    def test_process_sentence_nonparse_mode(self):
        doc = fox_doc()

        expected = {
            'toks': [
                {'tok': 'A'},
                {'tok': 'quick'},
                {'tok': 'brown'},
                {'tok': 'fox'},
                {'tok': 'jumps'},
                {'tok': 'over'},
                {'tok': 'the'},
                {'tok': 'lazy'},
                {'tok': 'dog'}
            ]
        }
        actual = _process_sentence(next(doc.sents), mode='tokenize', offset=3)

        self.assertDictEqual(expected, actual)

    def test_process_text_parse_mode(self):
        def fake_spacy_nlp(input_text):
            return fox_buffalo_doc()
        
        expected = process_text(text=FOX_BUFFALO_TEXT, mode='parse', spacy_nlp=fake_spacy_nlp)
        actual = [
            {
                "rt": 4,
                "toks": [
                    {
                        "tok": "A",
                        "tag": "DT",
                        "dep": "det",
                        "up": 4,
                        "dn": []
                    },
                    {
                        "tok": "quick",
                        "tag": "JJ",
                        "dep": "amod",
                        "up": 4,
                        "dn": []
                    },
                    {
                        "tok": "brown",
                        "tag": "JJ",
                        "dep": "amod",
                        "up": 4,
                        "dn": []
                    },
                    {
                        "tok": "fox",
                        "tag": "NN",
                        "dep": "compound",
                        "up": 4,
                        "dn": []
                    },
                    {
                        "tok": "jumps",
                        "tag": "NNS",
                        "dep": "ROOT",
                        "dn": [0, 1, 2, 3, 5, 9]
                    },
                    {
                        "tok": "over",
                        "tag": "IN",
                        "dep": "prep",
                        "up": 4,
                        "dn": [
                            8
                        ]
                    },
                    {
                        "tok": "the",
                        "tag": "DT",
                        "dep": "det",
                        "up": 8,
                        "dn": []
                    },
                    {
                        "tok": "lazy",
                        "tag": "JJ",
                        "dep": "amod",
                        "up": 8,
                        "dn": []
                    },
                    {
                        "tok": "dog",
                        "tag": "NN",
                        "dep": "pobj",
                        "up": 5,
                        "dn": [
                            6,
                            7
                        ]
                    },
                    {
                        "tok": ".",
                        "tag": ".",
                        "dep": "punct",
                        "up": 4,
                        "dn": []
                    }
                ]
            },
            {
                "rt": 1,
                "toks": [
                    {
                        "tok": "Buffalo",
                        "tag": "NNP",
                        "dep": "compound",
                        "up": 1,
                        "dn": []
                    },
                    {
                        "tok": "buffalo",
                        "tag": "NNP",
                        "dep": "ROOT",
                        "dn": [
                            0
                        ]
                    }
                ]
            },
            {
                "rt": 3,
                "toks": [
                    {
                        "tok": "Buffalo",
                        "tag": "NNP",
                        "dep": "compound",
                        "up": 1,
                        "dn": []
                    },
                    {
                        "tok": "buffalo",
                        "tag": "NNP",
                        "dep": "compound",
                        "up": 2,
                        "dn": [
                            0
                        ]
                    },
                    {
                        "tok": "buffalo",
                        "tag": "NNP",
                        "dep": "nsubj",
                        "up": 3,
                        "dn": [
                            1
                        ]
                    },
                    {
                        "tok": "buffalo",
                        "tag": "NNP",
                        "dep": "ROOT",
                        "dn": [
                            2
                        ]
                    }
                ]
            },
            {
                "rt": 1,
                "toks": [
                    {
                        "tok": "Buffalo",
                        "tag": "NNP",
                        "dep": "compound",
                        "up": 1,
                        "dn": []
                    },
                    {
                        "tok": "buffalo",
                        "tag": "NNP",
                        "dep": "ROOT",
                        "dn": [
                            0
                        ]
                    }
                ]
            }
        ]

        self.assertListEqual(expected, actual)

    def test_process_text_non_parse_mode(self):
        def fake_spacy_nlp(text):
            if text == FOX_TEXT:
                return fox_doc()
            if text == BUFFALO_TEXT:
                return buffalo_doc()
            
            raise Exception(f'Received text that matched neither expected test doc: {text}')

        class FakeSentenceTokenizer:
            def tokenize(self, text):
                return [FOX_TEXT, BUFFALO_TEXT]
        
        expected = [
            {
                "toks": [
                    {"tok": "A"},
                    {"tok": "quick"},
                    {"tok": "brown"},
                    {"tok": "fox"},
                    {"tok": "jumps"},
                    {"tok": "over"},
                    {"tok": "the"},
                    {"tok": "lazy"},
                    {"tok": "dog"}
                ]
            },
            {
                "toks": [
                    {"tok": "Buffalo"},
                    {"tok": "buffalo"},
                    {"tok": "Buffalo"},
                    {"tok": "buffalo"},
                    {"tok": "buffalo"},
                    {"tok": "buffalo"},
                    {"tok": "Buffalo"},
                    {"tok": "buffalo"}
                ]
            }
        ]
        actual = process_text(
            FOX_BUFFALO_TEXT,
            mode='tokenize',
            sent_tokenizer=FakeSentenceTokenizer(),
            spacy_nlp=fake_spacy_nlp
        )

        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
