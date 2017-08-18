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

# Need to fill in

#Output Figure 1A

# users_with_specific_affiliation = corpus.users(lambda u: u.info["is-admin"])
# [where instead of admins you would select users with a specific the affiliation]

# and
 
# log odds ratio (http://dept.stat.lsa.umich.edu/~kshedden/Python-Workshop/stats_calculations.html) 

# and of course they typology counts you would get using QuestionTypology
# questionTypology.display_question_type_log_odds_graph()
