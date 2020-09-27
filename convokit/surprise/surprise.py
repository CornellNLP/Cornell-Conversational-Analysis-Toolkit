import numpy as np
from collections import defaultdict
from convokit import Transformer
from convokit.model import Corpus, CorpusComponent
from convokit.speaker_convo_helpers import SpeakerConvoAttrs
from itertools import chain
from math import log2
from numpy import inf
from sklearn.feature_extraction.text import CountVectorizer
from typing import Callable, List

def _cross_entropy(model, target, context):
  # H(P,Q) = -sum_{x\inX}(P(x) * log(Q(x)))
  N_target, N_context = target.sum(), context.sum()
  context_log_probs = np.log2(context / N_context)
  # log(0) gives us -inf, so replace all -inf with 0
  context_log_probs[context_log_probs == -inf] = 0 
  return -np.dot(target / N_target, context_log_probs)

def sample(toks: np.ndarray, sample_size: int, n_samples=50):
  if toks.size < sample_size: return None
  return np.random.choice(toks, (n_samples, sample_size))


class Surprise(Transformer):
  """
  TODO: Docs
  """
  def __init__(self, min_target_length=100, min_context_length=100, n_samples=50, 
      sampling_fn: Callable[[List[str], int], List[str]]=sample, surprise_attr_name="surprise"):
    self.min_target_length = min_target_length
    self.min_context_length = min_context_length
    self.n_samples = n_samples
    self.sampling_fn = sampling_fn
    self.surprise_attr_name = surprise_attr_name
  
  def fit(self, corpus: Corpus, 
      target_selector: Callable[[CorpusComponent], bool]=lambda _: True, 
      context_selector: Callable[[CorpusComponent], bool]=lambda _:True,
      context_groupby: List[str]=[]):

    speaker_convo_utterances = defaultdict(lambda: defaultdict(list))
    for speaker in corpus.iter_objs('speaker'):
      for utt in speaker.iter_utterances():
        if utt.text:
          speaker_convo_utterances[speaker.id][utt.conversation_id].append(utt.text)

    self.grouped_text = {
      speaker: {
        convo: ' '.join(convo_utterances) for convo, convo_utterances in convos.items()
      } for speaker, convos in speaker_convo_utterances.items()
    }

    models = defaultdict(dict)
    for speaker, convos in self.grouped_text.items():
      speaker_convos = []
      for convo_id, convo_text in convos.items():
        if convo_text:
          speaker_convos.append(convo_text)
      try:
        cv = CountVectorizer()
        cv.fit(speaker_convos)
        models[speaker] = cv
      except ValueError:
        continue
    self.mapped_models = models
    self.mapped_tokenizers = {speaker: model.build_analyzer() for speaker, model in models.items()}
    return self

  def transform(self, corpus: Corpus,
      obj_type: str,
      selector: Callable[[CorpusComponent], bool]=lambda _: True):
    count = 0
    seen = defaultdict(dict)
    for obj in corpus.iter_objs(obj_type):
      if selector(obj):
        speaker_id = obj.get_speaker().get_id()
        convo_id = obj.get_conversation().get_id()
        if speaker_id in seen and convo_id in seen[speaker_id]:
          obj.add_meta(self.surprise_attr_name, seen[speaker_id][convo_id])
        elif speaker_id in self.mapped_models and convo_id in self.grouped_text[speaker_id]:
          speaker_model = self.mapped_models[speaker_id]
          speaker_tokenizer = self.mapped_tokenizers[speaker_id]
          target_toks = speaker_tokenizer(self.grouped_text[speaker_id][convo_id])
          context_toks = list(chain(*[speaker_tokenizer(text) for convo, text in self.grouped_text[speaker_id].items() if convo != convo_id]))
          surprise_score = self.compute_surprise(speaker_model, target_toks, context_toks)
          obj.add_meta(self.surprise_attr_name, surprise_score)
          seen[speaker_id][convo_id] = surprise_score
        else:
          obj.add_meta(self.surprise_attr_name, None)
      else:
        obj.add_meta(self.surprise_attr_name, None)
      count += 1
      if count % 2500 == 0:
        print('transformed: {} utterances'.format(count))
    return corpus

  def compute_surprise(self, model: CountVectorizer, target: List[str], context: List[str]):
    target_samples = self.sampling_fn(np.array(target), self.min_target_length, self.n_samples)
    context_samples = self.sampling_fn(np.array(context), self.min_context_length, self.n_samples)
    if target_samples is None or context_samples is None:
      return np.nan
    sample_entropies = np.empty(self.n_samples)
    for i in range(self.n_samples):
      target_doc_terms = np.asarray(model.transform(target_samples[i]).sum(axis=0)).squeeze()
      context_doc_terms = np.asarray(model.transform(context_samples[i]).sum(axis=0)).squeeze()
      sample_entropies[i] = _cross_entropy(model, target_doc_terms, context_doc_terms)
    return np.nanmean(sample_entropies)
    