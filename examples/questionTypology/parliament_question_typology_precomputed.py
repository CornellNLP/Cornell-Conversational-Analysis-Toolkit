# This example extracts question types from the UK Parliament Question Answer Sessions
#   reproducing the asking too much paper (http://www.cs.cornell.edu/~cristian/Asking_too_much.html).
#   (due to the non-deterministic nature of clustering, the order of the clusters and some cluster assignments will vary)
# This version uses precomputed motifs for speed.

import os
import pkg_resources
import numpy as np

from convokit import Corpus, QuestionTypology, download

# =================== DEBUG VERSION WITH 1/10 OF DATA =======================
# num_clusters = 8
# DEBUG_DIR = '/Users/ishaanjhaveri/Google_Drive/git/Cornell-Conversational-Analysis-Toolkit/datasets/parliament-corpus/downloads'


# # #Get precomputed motifs
# data_dir = DEBUG_DIR
# motifs_dir = os.path.join(data_dir, 'parliament-motifs')


# # # #Initialize QuestionTypology class

# corpus = Corpus(filename=os.path.join(data_dir, 'full.json'))
# questionTypology = QuestionTypology(corpus, data_dir, motifs_dir=motifs_dir, num_dims=5, 
#   num_clusters=num_clusters, verbose=False)


# ========================== REGULAR VERSION ===============================
num_clusters = 8

# Get precomputed motifs. data_dir contains the downloaded data. 
# motifs_dir is the specific path within data_dir that contains the precomputed motifs
data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads', 'parliament')
motifs_dir = os.path.join(data_dir, 'parliament-motifs')

# Initialize QuestionTypology class

corpus = Corpus(filename=os.path.join(data_dir, 'parliament-corpus'))
questionTypology = QuestionTypology(corpus, data_dir, motifs_dir=motifs_dir, num_dims=25, 
  num_clusters=num_clusters, verbose=False)

# questionTypology.types_to_data contains the necessary data that is computed in the step above
# its keys are the indices of the clusters (here 0-7). The values are dictionaries with the following keys:
# "motifs": the motifs, as a list of tuples of the motif terms
# "motif_dists": the corresponding distances of each motif from the centroid of the cluster this motif is in
# "fragments": the answer fragments, as a list of tuples of answer terms
# "fragment_dists": the corresponding distances of each fragment from the centroid of the cluster this 
# fragment is in
# "questions": the IDs of the questions in this cluster. You can get the corresponding question text by using the
# get_question_text_from_pair_idx(pair_idx) method.
# "question_dists": the corresponding distances of each question from the centroid of the cluster 
# this question is in

# #Output required data representations

questionTypology.display_totals()
print('10 examples for type 1-8:')
for i in range(num_clusters):
    questionTypology.display_motifs_for_type(i, num_egs=10)
    questionTypology.display_answer_fragments_for_type(i, num_egs=10)
    questionTypology.display_questions_for_type(i, num_egs=10)

example_question = "I thank the Minister for his response . He will be aware that the \
Northern Ireland Policing Board and the Chief Constable are concerned about a possible \
reduction in the police budget in the forthcoming financial year , and that there are \
increasing pressures on the budget as a result of policing the past , the ongoing \
inquiries , and the cost of the legal advice that the police need to secure in order \
to participate in them . However , does he agree that it is right that the Government \
provide adequate funding for the ordinary policing in the community that tackles all the \
matters that concern the people of Northern Ireland ? Does he accept that there should not \
be a reduction in the police budget , given the increasing costs of the inquiries that \
I have mentioned ? Will the Government do something to reduce the cost of the inquiries \
, and ensure that adequate policing is provided for all the victims of crime in \
Northern Ireland ?"

example_question = "What is the minister going to do about?"

print('Given a new question, we will now find the appropriate cluster for it:')
print('Question: ', example_question)
print('Cluster: ', questionTypology.classify_question(example_question))

print('Figure 1A from the paper will now display')
questionTypology.display_question_type_log_odds_graph()
