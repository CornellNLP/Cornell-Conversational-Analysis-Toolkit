from spacy.tests.util import get_doc
from spacy.util import get_lang_class

FOX_TEXT = 'A quick brown fox jumps over the lazy dog.'
BUFFALO_TEXT = 'Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo'
FOX_BUFFALO_TEXT = 'A quick brown fox jumps over the lazy dog. Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo'


def en_vocab():
    return get_lang_class("en").Defaults.create_vocab()


def fox_doc():
    return get_doc(
        vocab=en_vocab(),
        words=['A', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
        heads=[4, 3, 2, 1, 0, -1, 2, 1, -3],
        deps=['det', 'amod', 'amod', 'NN', 'ROOT', 'prep', 'det', 'amod', 'pobj'],
        tags=['DT', 'JJ', 'JJ', 'NN', 'NNS', 'IN', 'DT', 'JJ', 'NN']
    )


def buffalo_doc():
    return get_doc(
        vocab=en_vocab(),
        words=['Buffalo', 'buffalo', 'Buffalo', 'buffalo', 'buffalo', 'buffalo', 'Buffalo', 'buffalo'],
        heads=[1, 0, 1, 1, 1, 0, 1, 0],
        deps=['compound', 'ROOT', 'compound', 'compound', 'nsubj', 'ROOT', 'compound', 'ROOT'],
        tags=['NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP']
    )


def fox_buffalo_doc():
    return get_doc(
        vocab=en_vocab(),
        words=['A', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.', 'Buffalo', 'buffalo', 'Buffalo', 'buffalo', 'buffalo', 'buffalo', 'Buffalo', 'buffalo'],
        heads=[4, 3, 2, 1, 0, -1, 2, 1, -3, -5, 1, 0, 1, 1, 1, 0, 1, 0],
        deps=['det', 'amod', 'amod', 'compound', 'ROOT', 'prep', 'det', 'amod', 'pobj', 'punct', 'compound', 'ROOT', 'compound', 'compound', 'nsubj', 'ROOT', 'compound', 'ROOT'],
        tags=['DT', 'JJ', 'JJ', 'NN', 'NNS', 'IN', 'DT', 'JJ', 'NN', '.', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP']
    )
