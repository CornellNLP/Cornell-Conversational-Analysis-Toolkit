import numpy as np
from collections import defaultdict
from convokit import Transformer
from convokit.model import Corpus, CorpusComponent, Utterance
from itertools import chain
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

def sample(toks: Union[np.ndarray, List[str]], sample_size: int, n_samples=50, p=None):
  """
  Generates random samples from a list of tokens.

  :param toks: the list of tokens to sample from (either a numpy array or list of strings).
  :param sample_size: the number of tokens to include in each sample.
  :param n_samples: the number of samples to take.

  :return: numpy array where each row is a sample of tokens
  """
  if len(toks) < sample_size: return None
  rng = np.random.default_rng()
  return rng.choice(toks, (n_samples, sample_size), p=p)


class Surprise(Transformer):
  """
  Computes how surprising a target is based on some context. The measure for surprise used is cross entropy.
  Uses fixed size samples from target and context text to mitigate effects of length on cross entropy.

  :param model_key_selector: function that defines how utterances should be mapped to models
  :param cv: optional CountVectorizer used to tokenize text and create term document matrices. 
      default: scikit learn's default CountVectorizer
  :param surprise_attr_name: 
  :param target_sample_size: number of tokens to sample from each target
  :param context_sample_size: number of tokens to sample form each context
  :param n_samples: number of samples to take for each target-context pair
  :param sampling_fn: function for generating samples
  :param smooth: whether to use laplace smoothing when calculating surprise
  """
  def __init__(self, model_key_selector: Callable[[Utterance], str],
      cv=CountVectorizer(), 
      surprise_attr_name="surprise",
      target_sample_size=100, context_sample_size=100, n_samples=50, 
      sampling_fn: Callable[[np.ndarray, int], np.ndarray]=sample, 
      smooth: bool=True):
    self.model_key_selector = model_key_selector
    self.cv = cv
    self.surprise_attr_name = surprise_attr_name
    self.target_sample_size = target_sample_size
    self.context_sample_size = context_sample_size
    self.n_samples = n_samples
    self.sampling_fn = sampling_fn
    self.smooth = smooth
  
  def fit(self, corpus: Corpus,
      model_text_selector: Callable[[Utterance], List[str]]):
    """
    Fit CountVectorizers to utterances in a corpus. Can optionally group utterances and fit vectorizers for each group.

    :param corpus: corpus to fit models on
    """
    model_groups = {}
    for utt in corpus.iter_utterances():
      key = self.model_key_selector(utt)
      if key not in model_groups:
        model_groups[key] = model_text_selector(utt)
    self.models = {key: self.fit_cv(text) for key, text in model_groups.items()}
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
      group_and_models: Callable[[Utterance], Tuple[str, List[str]]]=None,
      selector: Callable[[CorpusComponent], bool]=lambda _: True):
    """
    Annotates `obj_type` components in `corpus` with surprise scores. Should be called after fit().

    :param corpus: corpus to compute surprise for.
    :param obj_type: the type of corpus components to annotate.
    :param group_and_models: 
    :param selector: function to select objects to annotate. if function returns true, object will be annotated.
    """
    if obj_type == 'corpus':
      utt_groups = defaultdict(list)
      group_models = defaultdict(list)
      for utt in corpus.iter_utterances():
        if group_and_models:
          group_name, models = group_and_models(utt)
        else:
          group_name = self.model_key_selector(utt)
          models = [group_name]
        utt_groups[group_name].append(utt.text)
        group_models[group_name] += models
      surprise_scores = {}
      for group_name in utt_groups:
        for model_key in group_models[group_name]:
          surprise_scores[Surprise.format_attr_key(group_name, model_key)] = self.compute_surprise(self.models[model_key], utt_groups[group_name])
      corpus.add_meta(self.surprise_attr_name, surprise_scores)
    elif obj_type == 'utterance':
      for utt in corpus.iter_utterances(selector=selector):
        if group_and_models:
          group_name, models = group_and_models(utt)
          surprise_scores = {}
          for model_key in models:
            surprise_scores[Surprise.format_attr_key(group_name, model_key)] = self.compute_surprise(self.models[model_key], utt.text)
          utt.add_meta(self.surprise_attr_name, surprise_scores)
        else:
          group_name = self.model_key_selector(utt)
          utt.add_meta(self.surprise_attr_name, self.compute_surprise(self.models[group_name], utt.text))
    else:
      for obj in corpus.iter_objs(obj_type, selector=selector):
        utt_groups = defaultdict(list)
        group_models = defaultdict(list)
        for utt in obj.iter_utterances():
          if group_and_models:
            group_name, models = group_and_models(utt)
          else:
            group_name = self.model_key_selector(utt)
            models = [group_name]
          utt_groups[group_name].append(utt.text)
          group_models[group_name] += models
        surprise_scores = {}
        for group_name in utt_groups:
          for model_key in group_models[group_name]:
            assert (model_key in self.models), 'invalid model key'
            if not self.models[model_key]: continue
            surprise_scores[Surprise.format_attr_key(group_name, model_key)] = self.compute_surprise(self.models[model_key], utt_groups[group_name])
        obj.add_meta(self.surprise_attr_name, surprise_scores)
    return corpus

  def compute_surprise(self, model: CountVectorizer, target: List[str]):
    """
    :param model: the CountVectorizer to use for finding term-doc matrices
    :param target: a list of tokens in the target
    :param context: a list of tokens in the context
    """
    model_vocab, model_vocab_freq = list(map(np.array, zip(*model.vocabulary_.items())))
    model_vocab_prob = model_vocab_freq / np.sum(model_vocab_freq)
    target_tokens = model.build_analyzer()(' '.join(target))
    target_samples = self.sampling_fn(np.array(target_tokens), self.target_sample_size, self.n_samples)
    context_samples = self.sampling_fn(model_vocab, self.context_sample_size, self.n_samples, p=model_vocab_prob)
    if target_samples is None or context_samples is None:
      return np.nan
    sample_entropies = np.empty(self.n_samples)
    for i in range(self.n_samples):
      target_doc_terms = np.asarray(model.transform(target_samples[i]).sum(axis=0)).squeeze()
      context_doc_terms = np.asarray(model.transform(context_samples[i]).sum(axis=0)).squeeze()
      sample_entropies[i] = _cross_entropy(target_doc_terms, context_doc_terms, self.smooth)
    return np.nanmean(sample_entropies)

  @staticmethod
  def format_attr_key(group_name, model_key):
    return 'GROUP_{}__MODEL_{}'.format(group_name, model_key)
    