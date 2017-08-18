import json, os
import itertools
from questionTypology.utils.tree_utils import read_arcs, read_downlinks, read_nodecounts

def contains_candidate(container, candidate):
	return set(candidate).issubset(container)

def fit_question(arc_set, downlinks, node_counts):
	fit_nodes = {}
	node_stack = [('*',)]
	while len(node_stack) > 0:
		next_node = node_stack.pop()
		node_count = node_counts.get(next_node,None)
		if node_count:
			entry = {'arcset': next_node, 'arcset_count': node_count}
			children = downlinks.get(next_node, [])
			valid_children = [child for child,_ in children if contains_candidate(arc_set, child)]

			if len(valid_children) == 0:
				entry['max_valid_child_count'] = 0
			else:
				entry['max_valid_child_count'] = max(node_counts.get(child,0) for child in valid_children)
			node_stack += valid_children
			fit_nodes[next_node] = entry
	return fit_nodes

def get_text_idx(span_idx):
	return '.'.join(span_idx.split('.')[:-1])

def fit_all(arc_file, tree_file, outfile, verbose=5000):
	'''
		figures out which motifs occur in each piece of text. 
		arc_file: listing of arcs per text, from extract_arcs
		tree_file: the motif graph, from make_arc_tree. note that
			this doesn't have to come from the same dataset as arc_file, in which case you're basically fitting a new dataset to motifs extracted elsewhere.
		outfile: where to put things.
	'''
	if verbose:
		print('\treading tree')
	arc_sets = read_arcs(arc_file, verbose)	

	downlinks = read_downlinks(tree_file + '_downlinks.json')
	node_counts = read_nodecounts(tree_file + '_arc_set_counts.tsv')


	if verbose:
		print('\tfitting arcsets')
	span_fit_entries = []
	for idx, (span_idx,arcs) in enumerate(arc_sets.items()):
		if verbose and (idx > 0) and (idx % verbose == 0):
			print('\t%03d' % idx)
		text_idx = get_text_idx(span_idx)
		fit_nodes = fit_question(set(arcs), downlinks, node_counts)
		for fit_info in fit_nodes.values():
			fit_info['span_idx'] = span_idx
			fit_info['text_idx'] = text_idx
			span_fit_entries.append(fit_info)
	if verbose:
		print('\twriting fits')
	with open(outfile, 'w') as f:
		f.write('\n'.join(json.dumps(entry) for entry in span_fit_entries))





