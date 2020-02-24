

from typing import Callable, Generator, Tuple, List, Dict, Set, Optional, Hashable


# from collections import defaultdict

from convokit.politeness_api.features.politeness_strategies import get_politeness_strategy_features
# from convokit.politeness_api.features.vectorizer import get_unigrams_and_bigrams

from convokit.transformer import Transformer
from convokit.model import Corpus

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PolitenessStrategies(Transformer):
    """
    Encapsulates extraction of politeness strategies from utterances in a
    Corpus.

    :param verbose: whether or not to print status messages while computing features
    """

    def __init__(self, verbose: bool=False):
        self.ATTR_NAME = "politeness_strategies"
        self.MRKR_NAME = "politeness_markers"
        self.verbose = verbose

    def transform(self, corpus: Corpus, markers: bool=False):
        """
        Extract politeness strategies from each utterances in the corpus and annotate
        the utterances with the extracted strategies. Requires that the corpus has previously
        been transformed by a Parser, such that each utterance has dependency parse info in
        its metadata table.
        
        :param corpus: the corpus to compute features for.
        :type corpus: Corpus
        :param markers: whether or not to add politeness occurance markers 
        """
        for utt in corpus.utterances.values():
            for i, sent in enumerate(utt.meta["parsed"]):
                for p in sent["toks"]:
                    p["tok"] = re.sub("[^a-z,.:;]","",p["tok"].lower())
            utt.meta[self.ATTR_NAME], marks = get_politeness_strategy_features(utt)
            
            if markers == True:
                utt.meta[self.MRKR_NAME] = marks
        
        return corpus
    
    def get_scores(self, corpus: Corpus, selector: Optional[Callable[[], bool]] = None):
        """
        Calculates average occurance per utterance. Used in summarize()
        
        :param corpus: the corpus used to compute averages
        :param selector: lambda function which takes in meta data and returns a boolean.
        """
        
        utts = [corpus.get_utterance(x) for x in corpus.get_utterance_ids()]
    
        if self.MRKR_NAME not in utts[0].meta:
            corpus = self.transform(corpus, markers=True)
            
        if selector != None:
            utts = [x for x in utts if selector(x.meta)]
            if len(utts) == 0:
                raise Exception("No query matches")
    
        counts = {k[21:len(k)-2]:0 for k in utts[0].meta[self.MRKR_NAME].keys()}
    
        for utt in utts:
            for k, v in utt.meta[self.MRKR_NAME].items():
                counts[k[21:len(k)-2]] += len(v)
        scores = {k:v/len(utts) for k,v in counts.items()}
        return scores
    
    def summarize(self, corpus: Corpus, selector=None, plot: bool = False, y_lim = None):
        """
        Calculates average occurance per utterance and outputs graph if plot == True
        
        :param corpus: the corpus used to compute averages
        :param selector: lambda function which takes in meta data and returns a boolean.
        
            For example, if selector is: lambda x : sum(x["politeness_strategies"].values()) == 1
            Then only utterances which have one politeness feature will be used in the calculation.
        
        :param plot: whether or not to output graph.
        """
        scores = self.get_scores(corpus, selector)
    
        if plot:
            plt.figure(dpi=200, figsize=(9,6))
            plt.bar(list(range(len(scores))), scores.values(), tick_label = list(scores.keys()), align="edge")
            plt.xticks(np.arange(.4, len(scores)+.4), rotation=45, ha="right")
            plt.ylabel("Occurance per Utterance", size = 20)
            plt.yticks(size=15)
            if y_lim != None:
                plt.ylim(0, y_lim)
            plt.show()
    
    
        return pd.DataFrame.from_dict(scores, orient='index', columns = ["Averages"])
            
        
            
        
        
        
