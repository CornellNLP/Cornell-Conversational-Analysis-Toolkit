# This example extracts question types from the UK Parliament Question Answer Sessions
#   reproducing the asking too much paper (http://www.cs.cornell.edu/~cristian/Asking_too_much.html).
#   (due to the non-deterministic nature of clustering, the order of the clusters and some cluster assignments will vary)

import os
import pkg_resources
import numpy as np

from convokit import Corpus, QuestionTypology, download

#Initialize QuestionTypology class

num_clusters = 8

# Get precomputed motifs. data_dir contains the downloaded data. 
# motifs_dir is the specific path within data_dir that contains the precomputed motifs
data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads', 'parliament')

#Load the corpus
corpus = Corpus(filename=os.path.join(data_dir, 'parliament-corpus'))

#Extract clusters of the motifs and assign questions to these clusters
questionTypology = QuestionTypology(corpus, data_dir, num_dims=25, 
  num_clusters=num_clusters, verbose=False, random_seed=164)


#Output required data representations

questionTypology.display_totals()
print('10 examples for type 1-8:')
for i in range(num_clusters):
    questionTypology.display_motifs_for_type(i, num_egs=10)
    questionTypology.display_answer_fragments_for_type(i, num_egs=10)
    questionTypology.display_questions_for_type(i, num_egs=10)
