import os
import pkg_resources
import numpy as np
import json

from convokit import Corpus, QuestionTypology, download, MotifsExtractor, QuestionTypologyUtils

import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

from ast import literal_eval as make_tuple
from collections import defaultdict
from scipy import sparse
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import Normalizer
from spacy.en import English
from spacy.symbols import *
from spacy.tokens.doc import Doc

#Initialize QuestionTypology class pretrained on Parliament Dataset

num_clusters = 8

data_dir = download('parliament-corpus')
motifs_dir = download('parliament-motifs')

corpus = Corpus(filename=os.path.join(data_dir, 'parliament-corpus'))

questionTypology = QuestionTypology(corpus, data_dir, dataset_name='parliament', motifs_dir=motifs_dir, num_dims=25,
  num_clusters=num_clusters, verbose=False, random_seed=164)

#Determine type of input question

example_question = "Does my right hon Friend agree that excellent regional universities—for example , the University of Northumbria at Newcastle and Sunderland—are anxious that they will be at a disadvantage if an élite group of universities , mainly in the south - east of England , are allowed to raise their fees to figures upwards of £ 10,000 a year , as today 's newspapers reported the Minister for Lifelong Learning and Higher Education as saying ?"
# example_question = "What is the minister going to do about?"
question_matrix, mtx, label = questionTypology.classify_question(example_question)
print('Question: ', example_question)
print('Cluster: ', label)
