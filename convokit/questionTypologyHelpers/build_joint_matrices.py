import os, json, numpy as np
from collections import defaultdict

from questionTypology.utils.tree_utils import read_arcs, read_downlinks, read_nodecounts
def get_text_idx(span_idx):
	return '.'.join(span_idx.split('.')[:-1])

def get_motifs_per_question(question_fit_file, answer_arc_file, superset_file,question_threshold, answer_threshold,  verbose=5000):
	question_to_fits = defaultdict(set)
	question_to_leaf_fits = defaultdict(set)
	motif_counts = defaultdict(int)


	super_mappings = {}
	with open(superset_file) as f:
		for line in f.readlines():
			entry = json.loads(line)
			super_mappings[tuple(entry['arcset'])] = tuple(entry['super'])

	with open(question_fit_file) as f:
		for idx, line in enumerate(f.readlines()):
			if verbose and (idx > 0) and (idx % verbose == 0):
				print('\t%03d' % idx)
			entry = json.loads(line)
			motif = tuple(entry['arcset'])
			super_motif = super_mappings[motif]
			if entry['arcset_count'] < question_threshold: continue
			if entry['max_child_count'] < question_threshold:
				question_to_leaf_fits[entry['text_idx']].add(super_motif)
				#if leaves_only: continue
			question_to_fits[entry['text_idx']].add(super_motif)
			motif_counts[super_motif] += 1
	question_to_fits = {k: [x for x in v if motif_counts[x] >= question_threshold] for k,v in question_to_fits.items()}
	motif_counts = {k:v for k,v in motif_counts.items() if v >= question_threshold}
	question_to_leaf_fits = {k: [x for x in v if motif_counts.get(x,0) >= question_threshold] for k,v in question_to_leaf_fits.items()}

	question_to_arcs = defaultdict(set)
	arc_sets = read_arcs(answer_arc_file)
	arc_counts = defaultdict(int)
	for span_idx, arcs in arc_sets.items():
		question_to_arcs[get_text_idx(span_idx)].update(arcs)
		for arc in arcs:
			arc_counts[arc] += 1
	question_to_arcs = {k: [x for x in v if arc_counts[x] >= answer_threshold] for k,v in question_to_arcs.items()}
	arc_counts = {k:v for k,v in arc_counts.items() if v >= answer_threshold}
	return question_to_fits, question_to_leaf_fits, motif_counts, question_to_arcs, arc_counts 

def build_joint_matrix(question_fit_file, answer_arc_file, superset_file, outfile, question_threshold, answer_threshold, verbose=5000):
	if verbose:
		print('\treading arcs and motifs')

	question_to_fits, question_to_leaf_fits, motif_counts, question_to_arcs, arc_counts =\
		 get_motifs_per_question(question_fit_file, answer_arc_file, superset_file, question_threshold, answer_threshold, verbose)
	question_term_list = list(motif_counts.keys())
	answer_term_list = list(arc_counts.keys())

	question_term_to_idx = {k:idx for idx,k in enumerate(question_term_list)}
	answer_term_to_idx = {k:idx for idx,k in enumerate(answer_term_list)}

	if verbose:
		print('\tbuilding matrices')
	question_term_idxes = []
	question_leaves = []
	question_doc_idxes = []
	answer_term_idxes = []
	answer_doc_idxes = []

	pair_idxes = list(set(question_to_fits.keys()).intersection(set(question_to_arcs.keys())))

	for idx, p_idx in enumerate(pair_idxes):
		if verbose and (idx > 0) and (idx % verbose == 0):
			print('\t%03d' % idx)

		question_terms = question_to_fits[p_idx]
		answer_terms = question_to_arcs[p_idx]

		for term in question_terms:
			term_idx = question_term_to_idx[term]
			question_term_idxes.append(term_idx)
			question_doc_idxes.append(idx)
			question_leaves.append(term in question_to_leaf_fits.get(p_idx,[]))
		for term in answer_terms:
			term_idx = answer_term_to_idx[term]
			answer_term_idxes.append(term_idx)
			answer_doc_idxes.append(idx)
	if verbose:
		print('\twriting stuff')
	
	np.save(outfile + '.q.tidx.npy', question_term_idxes)
	np.save(outfile + '.q.leaves.npy', question_leaves)
	np.save(outfile + '.a.tidx.npy', answer_term_idxes)
	np.save(outfile + '.q.didx.npy', question_doc_idxes)
	np.save(outfile + '.a.didx.npy', answer_doc_idxes)
	with open(outfile + '.q.terms.txt', 'w') as f:
		f.write('\n'.join('%d\t%s' % (motif_counts[term],term) for term in question_term_list))
	with open(outfile + '.a.terms.txt', 'w') as f:
		f.write('\n'.join('%d\t%s' % (arc_counts[term],term) for term in answer_term_list))
	with open(outfile + '.docs.txt', 'w') as f:
		f.write('\n'.join(pair_idxes))

