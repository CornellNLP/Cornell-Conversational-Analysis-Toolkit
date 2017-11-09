import os
import pkg_resources
import numpy as np
import json

from convokit import Corpus, QuestionTypology, download, MotifsExtractor, QuestionTypologyUtils

import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

from ast import literal_eval as make_tuple
from collections import defaultdict
from scipy import sparse
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import Normalizer
from spacy.en import English
from spacy.symbols import *
from spacy.tokens.doc import Doc

#Initialize QuestionTypology class pretrained on Parliament Dataset

num_clusters = 8

data_dir = os.path.join(pkg_resources.resource_filename("convokit", ""), 'downloads', 'parliament')
motifs_dir = os.path.join(data_dir, 'parliament-motifs')

corpus = Corpus(filename=os.path.join(data_dir, 'parliament-corpus'))

questionTypology = QuestionTypology(corpus, data_dir, dataset_name='parliament', motifs_dir=motifs_dir, num_dims=25,
  num_clusters=num_clusters, verbose=False, random_seed=164)

#Preprocessing
#create spacy object
spacy_NLP = spacy.load('en')
vocab = English().vocab

question_fit_file = os.path.join(questionTypology.motifs_dir, 'question_fits.json')

superset_file = os.path.join(questionTypology.motifs_dir, 'question_supersets_arcset_to_super.json')

question_to_leaf_fits = []

question_threshold = questionTypology.question_threshold

super_mappings = {}
with open(superset_file) as f:
    for line in f.readlines():
        entry = json.loads(line)
        super_mappings[tuple(entry['arcset'])] = tuple(entry['super'])

with open(question_fit_file) as f:
    for idx, line in enumerate(f.readlines()):
        entry = json.loads(line)
        motif = tuple(entry['arcset'])
        super_motif = super_mappings[motif]
        if entry['arcset_count'] < question_threshold: continue
        if entry['max_valid_child_count'] < question_threshold:
            question_to_leaf_fits.append(super_motif)

# if none of its children are in all_motifs, increment question_matrix
# else recurse on those children that are in all_motifs
def identify_sinks(parent, relevant_children, downlinks, question_matrix, all_motifs):
    children_in_all_motifs = [motif in all_motifs and motif != parent for motif in relevant_children]
    if any(children_in_all_motifs):
        for i in range(len(relevant_children)):
            if children_in_all_motifs[i]:
                identify_sinks(relevant_children[i], list(downlinks[relevant_children[i]].keys()), downlinks, question_matrix, all_motifs)
    else:
        j = all_motifs.index(parent)
        question_matrix[j] = 1

def compute_question_matrix(question_text):
        '''
            Helper function to classify_question. Computes and returns a representation of
            question_text as a matrix in the latent space
        '''
        spacy_q_obj = Doc(vocab).from_bytes(spacy_NLP(question_text).to_bytes())

        #extract question fragments
        for span_idx, span in enumerate(spacy_q_obj.sents):
            curr_arcset = MotifsExtractor.get_arcs(span.root, True)
            fragments = list(curr_arcset)
        fragment_dict = {}
        fragment_dict['1'] = list(fragments)
        itemset_counts, span_to_itemsets = MotifsExtractor.count_frequent_itemsets(fragment_dict, 
                                                                                   questionTypology.min_support, 
                                                                                   questionTypology.item_set_size, 
                                                                                   questionTypology.verbose)

        
        itemsets = []
        for count in itemset_counts:
            for itemset in itemset_counts[count]:
                if itemset in question_to_leaf_fits:
                    itemsets.append(itemset)
        
        new_itemset_counts = {}
        for setsize, size_dict in itemset_counts.items():
            for k,v in size_dict.items():
                new_itemset_counts[k] = v
        itemset_counts = new_itemset_counts
        itemset_counts[('*',)] = len(fragment_dict)

        sorted_counts = sorted(itemset_counts.items(),key=lambda x: (-x[1],len(x[0]),x[0][0]))

        edges = []
        uplinks = defaultdict(dict)
        downlinks = defaultdict(dict)

        for itemset,count in itemset_counts.items():
            parents = []
            set_size = len(itemset)
            if set_size == 1:
                arc = itemset[0]
                if arc.endswith('*'):
                    parents.append(('*',))
                elif '_' in arc:
                    parents.append((arc.split('_')[0] + '_*',))
                elif '>' in arc:
                    parents.append((arc.split('>')[0] + '>*',))

            else:
                for idx in range(set_size):
                    parents.append(itemset[:idx] + itemset[idx+1:])
            for parent in parents:
                parent_count = itemset_counts[parent]
                pr_child = count / itemset_counts[parent]
                edges.append({'child': itemset, 'child_count': count,
                            'parent': parent, 'parent_count': parent_count,
                            'pr_child': pr_child})
                uplinks[itemset][parent] = {'pr_child': pr_child, 'parent_count': parent_count}
                downlinks[parent][itemset] = {'pr_child': pr_child, 'child_count': count}


        all_motifs = list(questionTypology.mtx_obj['q_terms'])
    
        # create question_matrix
        question_matrix = np.zeros((questionTypology.num_motifs, 1))
        identify_sinks(('*',), list(downlinks[('*',)].keys()), downlinks, question_matrix, all_motifs)
        question_matrix = Normalizer(norm=questionTypology.norm).fit_transform(question_matrix)
        return question_matrix

def classify_question(question_text):
        '''
            Returns the type of question_text
        '''
        question_matrix = compute_question_matrix(question_text)
        mtx = np.matmul(question_matrix.T, questionTypology.lq)
        label = questionTypology.km.predict(mtx)
        return question_matrix, mtx, label

#Determine type of input question

example_question = "Does my right hon Friend agree that excellent regional universities—for example , the University of Northumbria at Newcastle and Sunderland—are anxious that they will be at a disadvantage if an élite group of universities , mainly in the south - east of England , are allowed to raise their fees to figures upwards of £ 10,000 a year , as today 's newspapers reported the Minister for Lifelong Learning and Higher Education as saying ?"
# example_question = "What is the minister going to do about?"
question_matrix, mtx, label = classify_question(example_question)
print('Question: ', example_question)
print('Cluster: ', label)
