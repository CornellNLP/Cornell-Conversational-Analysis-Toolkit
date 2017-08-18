import json, os
from collections import defaultdict
from questionTypology.utils.tree_utils import read_arcs, read_downlinks, read_nodecounts


def deduplicate_motifs(question_fit_file, outfile, threshold=.9, verbose=5000):

	if verbose:
		print('\treading raw fits')
	span_to_fits = defaultdict(set)
	arcset_counts = defaultdict(int)
	with open(question_fit_file) as f:
		for idx,line in enumerate(f.readlines()):
			if verbose and (idx > 0) and (idx % verbose == 0):
				print('\t%03d' % idx)
			entry = json.loads(line)
			span_to_fits[entry['span_idx']].add(tuple(entry['arcset']))
			arcset_counts[tuple(entry['arcset'])] += 1
	if verbose:
		print('\tcounting cooccs')
	coocc_counts = defaultdict(lambda: defaultdict(int))
	for idx, (span_idx, fit_arcs) in enumerate(span_to_fits.items()):
		if verbose and (idx > 0) and (idx % verbose == 0):
			print('\t%03d' % idx)
		fit_arcs = list(fit_arcs)
		for i in range(len(fit_arcs)):
			for j in range(i+1,len(fit_arcs)):
				arc1 = fit_arcs[i]
				arc2 = fit_arcs[j]
				coocc_counts[arc1][arc2] += 1
				coocc_counts[arc2][arc1] += 1
	if verbose:
		print('\tdeduplicating')
	superset_idx = 0
	supersets = defaultdict(set)
	arcset_to_superset = {}
	for arcset, count in arcset_counts.items():
		if arcset in arcset_to_superset: continue
		arcset_to_superset[arcset] = superset_idx
		supersets[superset_idx].add(arcset)
		stack = [arc2 for arc2,count2 in coocc_counts.get(arcset,{}).items()
					if (count2/count >= threshold) and (count2/arcset_counts[arc2] >= threshold)]
		while len(stack) > 0:
			neighbour = stack.pop()
			neighbour_count = arcset_counts[neighbour]
			arcset_to_superset[neighbour] = superset_idx
			supersets[superset_idx].add(neighbour)
			stack += [arc2 for arc2,count2 in coocc_counts.get(neighbour,{}).items()
					if (count2/neighbour_count >= threshold) and (count2/arcset_counts[arc2] >= threshold) and (arc2 not in arcset_to_superset)]
		superset_idx += 1
	superset_ids = {}
	for idx, superset in supersets.items():
		superset_ids[idx] = sorted(superset, key=lambda x: (arcset_counts[x],len(x)), reverse=True)[0]
	arcset_to_ids = {k: superset_ids[v] for k,v in arcset_to_superset.items()}
	supersets_by_id = [{'idx': k, 'id': superset_ids[k], 'items': list(v)} for k,v in supersets.items()]

	if verbose:
		print('\twriting')
	with open(outfile + '_arcset_to_super.json', 'w') as f:
		f.write('\n'.join(json.dumps({'arcset': k, 'super': v}) for k,v in arcset_to_ids.items()))
	with open(outfile + '_sets.json', 'w') as f:
		f.write('\n'.join(json.dumps(entry) for entry in supersets_by_id))
def get_text_idx(span_idx):
	return '.'.join(span_idx.split('.')[:-1])

def postprocess_fits(question_fit_file, question_tree_file, question_superset_file, verbose=5000):
	'''
		this entire file consists of two quite hacky scripts to remove 
		redundant motifs (i.e. p(m1|m2), p(m2|m1) > threshold)

	'''
	downlinks = read_downlinks(question_tree_file + '_downlinks.json')
	super_mappings = {}
	with open(question_superset_file) as f:
		for line in f.readlines():
			entry = json.loads(line)
			super_mappings[tuple(entry['arcset'])] = tuple(entry['super'])
	super_counts = defaultdict(int)
	span_to_fits = defaultdict(set)
	with open(question_fit_file) as f:
		for idx,line in enumerate(f.readlines()):
			if verbose and (idx > 0) and (idx % verbose == 0):
				print('\t%03d' % idx)
			entry = json.loads(line)
			span_to_fits[entry['span_idx']].add(tuple(entry['arcset']))
	for span_idx, fit_set in span_to_fits.items():
		super_fit_set = set([super_mappings[x] for x in fit_set if x != ('*',)])
		for x in super_fit_set:
			super_counts[x] += 1
		#span_to_super_fits[span_idx] = super_fit_set 
	if verbose:
		print('\tmaking new entries')
	new_entries = []
	for idx, (span_idx, fit_set) in enumerate(span_to_fits.items()):
		if verbose and (idx > 0) and (idx % verbose == 0):
			print('\t%03d' % idx)
		text_idx = get_text_idx(span_idx)
		super_to_superchildren = defaultdict(set)
		for set_ in fit_set:
			if set_ == ('*',): continue
			superset = super_mappings[set_]
			super_to_superchildren[superset].update([super_mappings[child] for child,_ in downlinks.get(set_, []) if child in fit_set])
		for superset, superchildren in super_to_superchildren.items():
			entry = {'arcset': superset, 'arcset_count': super_counts[superset],
					'text_idx': text_idx, 'span_idx': span_idx}
			if len(superchildren) == 0:
				entry['max_child_count'] = 0
			else:
				entry['max_child_count'] = max(super_counts.get(child,0) for child in superchildren)
			new_entries.append(entry)
	with open(question_fit_file + '.super', 'w') as f:
		f.write('\n'.join(json.dumps(entry) for entry in new_entries))