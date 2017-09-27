# The paper that developed these methods can be found here: (http://www.cs.cornell.edu/~cristian/Asking_too_much.html).
#
# The plots answer these questions:
# - ?
# This example extracts question types from the Tennis Interviews dataset (released with the Tie-breaker paper http://www.cs.cornell.edu/~liye/tennis.html)

import os
import pkg_resources

from convokit import Corpus, QuestionTypology, download

num_clusters = 8

#Initialize QuestionTypology class

num_clusters = 8
# Get precomputed motifs. data_dir contains the downloaded data. 

data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads', 'tennis')

#Load the corpus and filter out all non-winning tennis players. So the only question-answer pairs in this model
#are from reporters to winners
corpus = Corpus(filename=os.path.join(data_dir, 'tennis-corpus'))
corpus.filter_utterances_by(other_kv_pairs={'result':1})

#Extract clusters of the motifs and assign questions to these clusters
questionTypology = QuestionTypology(corpus, data_dir, dataset_name="tennis", num_dims=25, 
                                    num_clusters=num_clusters, verbose=False, random_seed=125)

#Output required data representations

questionTypology.display_totals()
print('10 examples for type 1-8:')
for i in range(num_clusters):
    questionTypology.display_motifs_for_type(i, num_egs=10)
    questionTypology.display_answer_fragments_for_type(i, num_egs=10)
    questionTypology.display_questions_for_type(i, num_egs=10)
