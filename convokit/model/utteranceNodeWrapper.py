from .utterance import Utterance
from typing import List
from itertools import chain


class UtteranceNodeWrapper:
    def __init__(self, utt: Utterance):
        self.utt = utt
        self.children = []

    def set_children(self, children: List['UtteranceNodeWrapper']):
        self.children = sorted(children, key=lambda w: w.utt.timestamp) # earliest to latest utt

    def pre_order(self):
        if len(self.children) == 0:
            return [self]
        return [self] + list(chain.from_iterable([c.pre_order() for c in self.children]))

    def post_order(self):
        if len(self.children) == 0:
            return [self]
        return list(chain.from_iterable([c.post_order() for c in self.children])) + [self]

    def bfs_traversal(self):
        ls = [self]
        while len(ls) > 0:
            curr = ls.pop(0)
            ls.extend(curr.children)
            yield curr

    def dfs_traversal(self):
        ls = [self]
        while len(ls) > 0:
            curr = ls.pop(0)
            ls = curr.children + ls
            yield curr
