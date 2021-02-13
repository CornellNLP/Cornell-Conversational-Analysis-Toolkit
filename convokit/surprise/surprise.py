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
  context_log_probs = -np.log((context + k) / (N_context + V))
  return np.dot(target / N_target, context_log_probs)

def sample(tokens: List[Union[np.ndarray, List[str]]], sample_size: int, n_samples=50, p=None):
  """
  Generates random samples from a list of lists of tokens.

  :param toks: a list of lists of tokens to sample from.
  :param sample_size: the number of tokens to include in each sample.
  :param n_samples: the number of samples to take.

  :return: numpy array where each row is a sample of tokens
  """
  tokens_list = [toks for toks in tokens if len(toks) >= sample_size]
  if len(tokens_list) == 0: return None
  rng = np.random.default_rng()
  sample_idxes = rng.integers(0, len(tokens_list), size=(n_samples))
  return [rng.choice(tokens_list[i], sample_size) for i in sample_idxes]


class Surprise(Transformer):
  """
  Computes how surprising a target is based on some context. The measure for surprise used is cross entropy.
  Uses fixed size samples from target and context text to mitigate effects of length on cross entropy.

  :param model_key_selector: function that defines how utterances should be mapped to models. 
      Takes in an utterance and returns the key to use for mapping the utterance to a corresponding model.
  :param cv: optional CountVectorizer used to tokenize text and create term document matrices. 
      default: scikit learn's default CountVectorizer
  :param surprise_attr_name: the name for the metadata attribute to add to objects.
      default: surprise
  :param target_sample_size: number of tokens to sample from each target (test text).
  :param context_sample_size: number of tokens to sample from each context (training text).
  :param n_samples: number of samples to take for each target-context pair.
  :param sampling_fn: function for generating samples of tokens.
  :param smooth: whether to use laplace smoothing when calculating surprise.
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
      text_func: Callable[[Utterance], List[str]]=None,
      selector: Callable[[Utterance], bool]=lambda utt: True):
    """
    Fits a model for each group of utterances in a corpus. The group that an 
    utterance belongs to is determined by the `model_key_selector` parameter in 
    the transformer's constructor. The type of model used is defined by the `cv` 
    parameter in the constructor.

    :param corpus: corpus to fit models on.
    :param text_func: optional function to define how the text a model is trained 
        on should be selected. Takes an utterance as input and returns a list of 
        strings to train the model corresponding to that utterance on. The model 
        corresponding to the utterance is determined by `self.model_key_selector`. 
        For every utterance corresponding to the same model key, this function 
        should return the same result.
        If `text_func` is `None`, a model will be trained on the text from all 
        the utterances that belong to its group.
    :param selector: determines which utterances in the corpus to train models for.
    """
    self.model_groups = defaultdict(list)
    for utt in corpus.iter_utterances(selector=selector):
      key = self.model_key_selector(utt)
      if text_func:
        if key not in self.model_groups:
          self.model_groups[key] = text_func(utt)
      else:
        self.model_groups[key].append(utt.text)
    self.models = {key: self.fit_cv(text) for key, text in self.model_groups.items()}
    return self

  def fit_cv(self, text: List[str]):
    """
    Helper function to fit a new model to some text.
    """
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
    Annotates `obj_type` components in a corpus with surprise scores. Should be 
    called after fit().

    :param corpus: corpus to compute surprise for.
    :param obj_type: the type of corpus components to annotate. Should be either 
        'utterance', 'speaker', 'conversation', or 'corpus'. 
    :param group_and_models: optional function that defines how an utterance should 
        be grouped to form a target text and what models (contexts) the group should 
        be compared to when calculating surprise. Takes in an utterance and returns 
        a tuple containing the name of the group the utterance belongs to and a 
        list of models to calculate how surprising that group is against. Objects 
        will be annotated with a metadata field `self.surprise_attr_name` that is 
        a mapping 'GROUP_groupname_MODEL_modelkey' to the surprise score for 
        utterances in `groupname` group when compared to `modelkey` model.
        If `group_and_models` is `None`, `self.model_key_selector` will be used 
        to select the group that an utterance belongs to. The surprise score will 
        be calculated for each group of utterances compared to the model in 
        `self.models` corresponding to the group.
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
          model = self.models[model_key]
          context = self.model_groups[model_key]
          surprise_scores[Surprise.format_attr_key(group_name, model_key)] = self.compute_surprise(model, utt_groups[group_name], context)
      corpus.add_meta(self.surprise_attr_name, surprise_scores)
    elif obj_type == 'utterance':
      for utt in corpus.iter_utterances(selector=selector):
        if group_and_models:
          group_name, models = group_and_models(utt)
          surprise_scores = {}
          for model_key in models:
            model = self.models[model_key]
            context = self.model_groups[model_key]
            surprise_scores[Surprise.format_attr_key(group_name, model_key)] = self.compute_surprise(self.models[model_key], [utt.text], context)
          utt.add_meta(self.surprise_attr_name, surprise_scores)
        else:
          group_name = self.model_key_selector(utt)
          model = self.models[group_name]
          context = self.model_groups[group_name]
          utt.add_meta(self.surprise_attr_name, self.compute_surprise(model, [utt.text], context))
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
            model = self.models[model_key]
            context = self.model_groups[model_key]
            surprise_scores[Surprise.format_attr_key(group_name, model_key)] = self.compute_surprise(model, utt_groups[group_name], context)
        obj.add_meta(self.surprise_attr_name, surprise_scores)
    return corpus

  def compute_surprise(self, model: CountVectorizer, target: List[str], context):
    """
    Computes how surprising a target text is based on a model trained on a context. 
    Surprise scores are calculated using cross entropy. To mitigate length based 
    effects on cross entropy, several random samples of fixed size are taken from 
    the target and context. Returns the average of the cross entropies for all 
    pairs of samples.

    :param model: the CountVectorizer to use for finding term-doc matrices
    :param target: a list of tokens in the target
    :param context: the term document matrix for the context
    """
    target_tokens = np.array(model.build_analyzer()(' '.join(target)))
    context_tokens = np.array(model.build_analyzer()(' '.join(context)))
    target_samples = self.sampling_fn([target_tokens], self.target_sample_size, self.n_samples)
    context_samples = self.sampling_fn([context_tokens], self.context_sample_size, self.n_samples)
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
    