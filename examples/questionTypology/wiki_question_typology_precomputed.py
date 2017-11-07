# This example extracts question types from the UK Parliament Question Answer Sessions
#   reproducing the asking too much paper (http://www.cs.cornell.edu/~cristian/Asking_too_much.html).
#   (due to the non-deterministic nature of clustering, the order of the clusters and some cluster assignments will vary)

import os
import pkg_resources

from convokit import Corpus, QuestionTypology, download

#Initialize QuestionTypology class

num_clusters = 8

# Get precomputed motifs. data_dir contains the downloaded data.
# motifs_dir is the specific path within data_dir that contains the precomputed motifs
data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads', 'wiki')
motifs_dir = os.path.join(data_dir, 'wiki-motifs')

#Load the corpus
corpus = Corpus(filename=os.path.join(data_dir, 'wiki-corpus'))

#Extract clusters of the motifs and assign questions to these clusters
questionTypology = QuestionTypology(corpus, data_dir, motifs_dir=motifs_dir, dataset_name="wiki", num_dims=25,
                                    num_clusters=num_clusters, verbose=False, random_seed=15)

#Output required data representations

questionTypology.display_totals()
print('100 examples for types 1-8:')
for i in range(num_clusters):
    questionTypology.display_question_answer_pairs_for_type(i, num_egs=100)
    questionTypology.display_motifs_for_type(i, num_egs=100)
    questionTypology.display_answer_fragments_for_type(i, num_egs=100)

# Does it contain a question mark or not?
# If it does, identify what is a question
# Refactor code - dont use pair index, use

# When you see a reply, check if what its replying to is a question
# How to check if something is a question? Check the function ____
# Input an is_question function, else use the default one
# In the parliament case, every pair is a question-answer pair
# If there is no question ignore it, if there is a question treat it as a pair
# In the case of the parliament/tennis stuff, we hope nothing changes
# See some difference in questioning
# New pull request

