import numpy as np
import pandas as pd
from collections import defaultdict
from convokit import Transformer
from convokit.model import Corpus, CorpusComponent, Utterance
from convokit.speaker_convo_helpers import SpeakerConvoAttrs
from itertools import chain
from math import log2
from numpy import inf
from pandas.core.groupby.generic import SeriesGroupBy
from sklearn.feature_extraction.text import CountVectorizer
from typing import Callable, List, Tuple, Union

def _cross_entropy(model, target, context, smooth=True):
  # H(P,Q) = -sum_{x\inX}(P(x) * log(Q(x)))
  N_target, N_context = target.sum(), context.sum()
  if N_context == 0: return np.nan
  V = np.sum(context > 0) if smooth else 0
  k = 1 if smooth else 0
  if not smooth: context[context == 0] = 1
  context_log_probs = -np.log(context + k / (N_context + V))
  return np.dot(target / N_target, context_log_probs)

def sample(toks: np.ndarray, sample_size: int, n_samples=50):
  if toks.size < sample_size: return None
  return np.random.choice(toks, (n_samples, sample_size))


class Surprise(Transformer):
  """
  TODO: Docs
  """
  df_col_name = {
    'conversation': 'conversation_id',
    'speaker': 'speaker',
    'utterance': 'id'
  }

  obj_field_getter = {
    'id': lambda obj: obj.id,
    'speaker': lambda obj: obj.get_speaker().id,
    'conversation_id': lambda obj: obj.get_conversation().id
  }

  def __init__(self, cv=CountVectorizer(), target_sample_size=100, context_sample_size=100, n_samples=50, 
      sampling_fn: Callable[[List[str], int], List[str]]=sample, surprise_attr_name="surprise"):
    self.cv = cv
    self.target_sample_size = target_sample_size
    self.context_sample_size = context_sample_size
    self.n_samples = n_samples
    self.sampling_fn = sampling_fn
    self.surprise_attr_name = surprise_attr_name
  
  def fit(self, corpus: Corpus, 
      group_models_by: List[str]=[],
      selector: Callable[[Utterance], bool]=lambda utt: True):
    utterances = corpus.get_utterances_dataframe()
    if group_models_by:
      model_groups = utterances.groupby([self.df_col_name[group] for group in group_models_by])
      self.mapped_models = model_groups['text'].apply(self.fit_cv).dropna()
    else:
      self.mapped_models = pd.Series(self.cv.fit(utterances['text']))
    return self

  def fit_cv(self, text):
    try:
      cv = CountVectorizer().set_params(**self.cv.get_params())
      cv.fit(text)
      return cv
    except ValueError:
      return None

  def transform(self, corpus: Corpus,
      obj_type: str,
      group_target_by: List[str] = [],
      context_selector: Callable[[pd.Series, Tuple], np.ndarray]=lambda s, t: np.ones(len(s)).astype(bool), 
      model_selector: Callable[[Tuple], Union[str, int]]=lambda _: 0,
      selector: Callable[[CorpusComponent], bool]=lambda _: True,
      smooth: bool=True):
    utterances = corpus.get_utterances_dataframe()
    if group_target_by:
      grouped_utterances = utterances.groupby([self.df_col_name[group] for group in group_target_by])['text'].apply(lambda x: self.cv.build_analyzer()(' '.join(x)))
    else:
      grouped_utterances = pd.Series(self.cv.build_analyzer()(' '.join(utterances['text'])))
    surprise_scores = {}
    for ind, target in grouped_utterances.items():
      model_ind = model_selector(ind)
      if model_ind in self.mapped_models:
        model = self.mapped_models[model_ind]
        context = list(chain(*grouped_utterances[context_selector(grouped_utterances, ind)]))
        surprise_scores[ind] = self.compute_surprise(model, target, context, smooth)
      else:
        surprise_scores[ind] = np.nan
    for obj in corpus.iter_objs(obj_type):
      if selector(obj):
        ind = tuple(map(lambda x: self.obj_field_getter[x](obj), grouped_utterances.index.names))
        if len(ind) == 1:
          ind = ind[0]
        obj.add_meta(self.surprise_attr_name, surprise_scores[ind])
    return corpus

  def compute_surprise(self, model: CountVectorizer, target: List[str], context: List[str], smooth):
    target_samples = self.sampling_fn(np.array(target), self.target_sample_size, self.n_samples)
    context_samples = self.sampling_fn(np.array(context), self.context_sample_size, self.n_samples)
    if target_samples is None or context_samples is None:
      return np.nan
    sample_entropies = np.empty(self.n_samples)
    for i in range(self.n_samples):
      target_doc_terms = np.asarray(model.transform(target_samples[i]).sum(axis=0)).squeeze()
      context_doc_terms = np.asarray(model.transform(context_samples[i]).sum(axis=0)).squeeze()
      sample_entropies[i] = _cross_entropy(model, target_doc_terms, context_doc_terms, smooth)
    return np.nanmean(sample_entropies)
    