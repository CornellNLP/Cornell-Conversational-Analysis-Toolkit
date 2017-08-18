# This example uses the question typology court corpus to reproduce figures 1A and 1B from
#   the asking too much paper (http://www.cs.cornell.edu/~cristian/Asking_too_much.html).
#
# The plots answer these questions:
# - ?

import os
import pkg_resources
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np

from convokit import Utterance, Corpus, QuestionTypology, download

#Initialize QuestionTypology class
data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads')
corpus = Corpus(filename=download('parliament-corpus'))
# corpus = Corpus(filename='/Users/comake/ish/Google_Drive/git/Cornell-Conversational-Analysis-Toolkit/datasets/parliament-corpus/full.json')
questionTypology = QuestionTypology(corpus, data_dir)

#Output required data representations
