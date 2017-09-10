# This example extracts question types from the Tennis Interviews dataset (released with the Tie-breaker paper http://www.cs.cornell.edu/~liye/tennis.html)


import os
import pkg_resources

from convokit import Corpus, QuestionTypology, download

# =================== DEBUG VERSION WITH 1/10 OF DATA =======================
# num_clusters = 8
# DEBUG_DIR = '/Users/ishaanjhaveri/Google_Drive/git/Cornell-Conversational-Analysis-Toolkit/datasets/tennis-corpus/downloads/tennis'
# data_dir = DEBUG_DIR

# #Initialize QuestionTypology class

# corpus = Corpus(filename=os.path.join(data_dir, 'full.json'))
# corpus.filter_utterances_by(other_kv_pairs={'result':1})
# questionTypology = QuestionTypology(corpus, data_dir, dataset_name="tennis", num_dims=5, 
#   num_clusters=num_clusters, verbose=False)


# ========================== REGULAR VERSION ===============================
num_clusters = 8

#Initialize QuestionTypology class

data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads', 'tennis')
corpus = Corpus(filename=os.path.join(data_dir, 'tennis-corpus'))
corpus.filter_utterances_by(other_kv_pairs={'result':1})
questionTypology = QuestionTypology(corpus, data_dir, dataset_name="tennis", num_dims=25, 
  num_clusters=num_clusters, verbose=False)

#Output required data representations

questionTypology.display_totals()
print('10 examples for type 1-8:')
for i in range(num_clusters):
    questionTypology.display_motifs_for_type(i, num_egs=10)
    questionTypology.display_answer_fragments_for_type(i, num_egs=10)
    questionTypology.display_questions_for_type(i, num_egs=10)

