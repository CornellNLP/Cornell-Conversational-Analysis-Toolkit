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

#Download precomputed motifs
motifs_dir = download('parliament-motifs')

#Initialize QuestionTypology class
data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads')
corpus = Corpus(filename=download('parliament-corpus'))
questionTypology = QuestionTypology(corpus, data_dir, motifs_dir)

#Output required data representations
