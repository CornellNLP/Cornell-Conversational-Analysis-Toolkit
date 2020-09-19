import numpy as np
from collections import defaultdict
from convokit import Transformer
from convokit.model import Corpus, CorpusComponent
from convokit.speaker_convo_helpers import SpeakerConvoAttrs
from math import log2
from numpy import inf
from sklearn.feature_extraction.text import CountVectorizer
from typing import Callable, List

def _cross_entropy(min_target_length, min_context_length, model, target, context):
  # H(P,Q) = -sum_{x\inX}(P(x) * log(Q(x)))
  N_target, N_context = target.sum(), context.sum()
  if N_target < min_target_length or N_context < min_context_length: return np.nan
  context_log_probs = np.log2(context / N_context)
  # log(0) gives us -inf, so replace all -inf with 0
  context_log_probs[context_log_probs == -inf] = 0 
  return -np.dot(target / N_target, context_log_probs)


class Surprise(Transformer):
  """
  TODO: Docs
  """
  def __init__(self, min_target_length=100, min_context_length=100, surprise_attr_name="surprise"):
    self.min_target_length = min_target_length
    self.min_context_length = min_context_length
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
          target_text = [self.grouped_text[speaker_id][convo_id]]
          context_text = [y for x, y in self.grouped_text[speaker_id].items() if x != convo_id]
          surprise_score = self.compute_surprise(speaker_model, target_text, context_text)
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
    target_doc_terms = np.asarray(model.transform(target).sum(axis=0)).squeeze()
    context_doc_terms = np.asarray(model.transform(context).sum(axis=0)).squeeze()
    return _cross_entropy(self.min_target_length, self.min_context_length, model, target_doc_terms, context_doc_terms)
    