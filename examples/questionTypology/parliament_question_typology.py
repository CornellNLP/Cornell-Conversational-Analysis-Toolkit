# This example extracts question types from the UK Parliament Question Answer Sessions
#   reproducing the asking too much paper (http://www.cs.cornell.edu/~cristian/Asking_too_much.html).
#   (due to the non-deterministic nature of clustering, the order of the clusters and some cluster assignments will vary)

import os
import pkg_resources

from convokit import Corpus, QuestionTypology, download

# =================== DEBUG VERSION WITH 1/10 OF DATA =======================
num_clusters = 8
DEBUG_DIR = '/Users/ishaanjhaveri/Google_Drive/git/Cornell-Conversational-Analysis-Toolkit/datasets/parliament-corpus/downloads'
data_dir = DEBUG_DIR

# #Initialize QuestionTypology class

corpus = Corpus(filename=os.path.join(data_dir, 'full.json'))
questionTypology = QuestionTypology(corpus, data_dir, num_dims=5, 
  num_clusters=num_clusters, verbose=5000)


# ========================== REGULAR VERSION ===============================
# num_clusters = 8

# #Initialize QuestionTypology class

# data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads', 'parliament')
# corpus = Corpus(filename=os.path.join(data_dir, 'parliament-corpus'))
# questionTypology = QuestionTypology(corpus, data_dir, num_dims=25, 
#   num_clusters=num_clusters, verbose=False)

# #Output required data representations

questionTypology.display_totals()
print('10 examples for type 1-8:')
for i in range(num_clusters):
    questionTypology.display_motifs_for_type(i, num_egs=10)
    questionTypology.display_answer_fragments_for_type(i, num_egs=10)
    questionTypology.display_questions_for_type(i, num_egs=10)

