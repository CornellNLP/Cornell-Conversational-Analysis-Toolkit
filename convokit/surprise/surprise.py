import numpy as np
from collections import defaultdict
from convokit import Transformer
from convokit.model import Corpus, CorpusComponent
from convokit.speaker_convo_helpers import SpeakerConvoAttrs
from math import log2
from sklearn.feature_extraction.text import CountVectorizer
from typing import Callable, List

def _cross_entropy(model, target, context):
  # H(P,Q) = -sum_{x\inX}(P(x) * log(Q(x)))
  N_target, N_context = target.sum(), context.sum()
  return -np.dot(target / N_target, np.log2(context / N_context))

class Surprise(Transformer):
  """
  TODO: Docs
  """
  def __init__(self, target_groupby: List[str]=[], surprise_attr_name="surprise"):
    self.target_groupby = target_groupby
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
    for obj in corpus.iter_objs(obj_type):
      if selector(obj):
        speaker_id = obj.get_speaker().get_id()
        convo_id = obj.get_conversation().get_id()
        if speaker_id in self.mapped_models and convo_id in self.grouped_text[speaker_id]:
          speaker_model = self.mapped_models[speaker_id]
          target_text = [self.grouped_text[speaker_id][convo_id]]
          context_text = [y for x, y in self.grouped_text[speaker_id].items() if x != convo_id]
          obj.add_meta(
            self.surprise_attr_name, 
            Surprise.compute_surprise(speaker_model, target_text, context_text)
          )
        else:
          obj.add_meta(self.surprise_attr_name, None)
      else:
        obj.add_meta(self.surprise_attr_name, None)
    return corpus

  @staticmethod
  def compute_surprise(model: CountVectorizer, target: List[str], context: List[str]):
    target_doc_terms = np.asarray(model.transform(target).sum(axis=0)).squeeze()
    context_doc_terms = np.asarray(model.transform(context).sum(axis=0)).squeeze()
    