import numpy as np
import pandas as pd
from collections import defaultdict
from convokit import Transformer
from convokit.model import Corpus, CorpusComponent, Conversation, Speaker, Utterance
from convokit.speaker_convo_helpers import SpeakerConvoAttrs
from itertools import chain
from math import log2
from numpy import inf
from pandas.core.groupby.generic import SeriesGroupBy
from sklearn.feature_extraction.text import CountVectorizer
from typing import Callable, List, Tuple, Union

def _cross_entropy(target, context, smooth=True):
  """
  Calculates H(P,Q) = -sum_{x\inX}(P(x) * log(Q(x)))

  :param target: term-doc matrix for target text (P)
  :param context: term-doc matrix for context (Q)
  :param smooth: whether to use laplace smoothing for OOV tokens

  :return: cross entropy
  """
  N_target, N_context = target.sum(), context.sum()
  if N_context == 0: return np.nan
  V = np.sum(context > 0) if smooth else 0
  k = 1 if smooth else 0
  if not smooth: context[context == 0] = 1
  context_log_probs = -np.log(context + k / (N_context + V))
  return np.dot(target / N_target, context_log_probs)

def sample(toks: Union[np.ndarray, List[str]], sample_size: int, n_samples=50):
  """
  Generates random samples from a list of tokens.

  :param toks: the list of tokens to sample from (either a numpy array or list of strings).
  :param sample_size: the number of tokens to include in each sample.
  :param n_samples: the number of samples to take.

  :return: numpy array where each row is a sample of tokens
  """
  if len(toks) < sample_size: return None
  rng = np.random.default_rng()
  return rng.choice(toks, (n_samples, sample_size))


class Surprise(Transformer):
  df_col_name = {
    'conversation': 'conversation_id',
    'speaker': 'speaker',
    'utterance': 'id'
  }

  """
  Computes how surprising a target is based on some context. The measure for surprise used is cross entropy.
  Uses fixed size samples from target and context text to mitigate effects of length on cross entropy.

  :param cv: optional CountVectorizer used to tokenize text and create term document matrices. 
      default: scikit learn's default CountVectorizer
  :param target_sample_size: number of tokens to sample from each target
  :param context_sample_size: number of tokens to sample form each context
  :param n_samples: number of samples to take for each target-context pair
  :param sampling_fn: function for generating samples
  """
  def __init__(self, cv=CountVectorizer(), target_sample_size=100, context_sample_size=100, n_samples=50, 
      sampling_fn: Callable[[np.ndarray, int], np.ndarray]=sample, surprise_attr_name="surprise"):
    self.cv = cv
    self.target_sample_size = target_sample_size
    self.context_sample_size = context_sample_size
    self.n_samples = n_samples
    self.sampling_fn = sampling_fn
    self.surprise_attr_name = surprise_attr_name
  
  def fit(self, corpus: Corpus, 
      group_models_by: List[str]=[]):
    """
    Fit CountVectorizers to utterances in a corpus. Can optionally group utterances and fit vectorizers for each group.

    :param corpus: corpus to fit vectorizers for
    :param group_models_by: attributes to group utterances by before fitting vectorizers
    """
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
    """
    Annotates `obj_type` components in `corpus` with surprise scores. Should be called after fit().

    :param corpus: corpus to compute surprise for.
    :param obj_type: the type of corpus components to annotate.
    :param group_target_by: list of attributes to group target text tokens. 
        i.e. utterance, speaker, conversation. 
    :param context_selector: function to select context tokens from pandas series to compare a target to.
        function should return boolean mask over input series.
    :param model_selector: function to select model (created in `fit` method) to use for a target-context pair.
        function should return a string or int corresponding to the index of the model in `self.mapped_models`.
    :param selector: function to select objects to annotate. if function returns true, object will be annotated.
    :param smooth: whether to use laplace smoothing in surprise calculation.
    """
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
        ind = tuple(map(lambda x: Surprise.get_obj_field(obj, x), grouped_utterances.index.names))
        if len(ind) == 1:
          ind = ind[0]
        obj.add_meta(self.surprise_attr_name, surprise_scores[ind])
    return corpus

  def compute_surprise(self, model: CountVectorizer, target: List[str], context: List[str], smooth):
    """
    :param model: the CountVectorizer to use for finding term-doc matrices
    :param target: a list of tokens in the target
    :param context: a list of tokens in the context
    :param smooth: whether to use laplace smoothing for cross entropy calculation
    """
    target_samples = self.sampling_fn(np.array(target), self.target_sample_size, self.n_samples)
    context_samples = self.sampling_fn(np.array(context), self.context_sample_size, self.n_samples)
    if target_samples is None or context_samples is None:
      return np.nan
    sample_entropies = np.empty(self.n_samples)
    for i in range(self.n_samples):
      target_doc_terms = np.asarray(model.transform(target_samples[i]).sum(axis=0)).squeeze()
      context_doc_terms = np.asarray(model.transform(context_samples[i]).sum(axis=0)).squeeze()
      sample_entropies[i] = _cross_entropy(target_doc_terms, context_doc_terms, smooth)
    return np.nanmean(sample_entropies)

  @staticmethod
  def get_obj_field(obj, field):
    if type(obj) == Conversation:
      if field == 'conversation_id':
        return obj.id
    elif type(obj) == Speaker:
      if field == 'speaker':
        return obj.id
    else:
      if field == 'id':
        return obj.id
      if field == 'conversation_id':
        return obj.get_conversation().id
      if field == 'speaker':
        return obj.get_speaker().id
    