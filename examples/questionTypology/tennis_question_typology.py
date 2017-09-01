# This example uses the question typology court corpus to reproduce figures 1A and 1B from
#   the asking too much paper (http://www.cs.cornell.edu/~cristian/Asking_too_much.html).
#
# The plots answer these questions:
# - ?

import os
import pkg_resources
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import numpy as np

from convokit import Utterance, Corpus, QuestionTypology, download

DEBUG = False
DEBUG_DIR = '/Users/ishaanjhaveri/Google_Drive/git/Cornell-Conversational-Analysis-Toolkit/datasets/tennis-corpus/downloads'

#Initialize QuestionTypology class
data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads', 'tennis-spacy') if not DEBUG else DEBUG_DIR
corpus = Corpus(filename=data_dir+"/tennis-corpus") if not DEBUG else Corpus(filename=DEBUG_DIR+'/full.json')
corpus.filter_utterances_by(other_kv_pairs={'result':1})
questionTypology = QuestionTypology(corpus, data_dir, dataset_name="tennis")

#Output required data representations

# Need to fill in

#Output Figure 1A

# users_with_specific_affiliation = corpus.users(lambda u: u.info["is-admin"])
# [where instead of admins you would select users with a specific the affiliation]

# and
 
# log odds ratio (http://dept.stat.lsa.umich.edu/~kshedden/Python-Workshop/stats_calculations.html) 

# and of course they typology counts you would get using QuestionTypology
# questionTypology.display_question_type_log_odds_graph()


# questionTypology.classify_question(question_answer_pair)

print('5 examples for type 1-8:')
for i in range(8):
    questionTypology.display_motifs_for_type(i)
    questionTypology.display_answer_fragments_for_type(i)
    questionTypology.display_questions_for_type(i)

