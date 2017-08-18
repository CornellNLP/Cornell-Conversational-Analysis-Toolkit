import json, os
import itertools
from collections import defaultdict
from questionTypology.utils.tree_utils import read_arcs
 
def get_sorted_combos(itemset, k):
	combos = set()
	for set_ in itertools.combinations(itemset,k):
		combos.add(tuple(sorted(set_)))
	return combos
def get_mini_powerset(itemset,k=5):
	powerset = set()
	for k in range(1,min(k+1,len(itemset)+1)):
		powerset.update(get_sorted_combos(itemset,k))
	return powerset

def count_frequent_itemsets(arc_sets,min_support,k=5, verbose=5000):
	itemset_counts = defaultdict(lambda: defaultdict(int))
	span_to_itemsets = defaultdict(lambda: defaultdict(set))
	if verbose:
		print('\tfirst pass')
	for idx, (span_idx,arcs) in enumerate(arc_sets.items()):
		if verbose and (idx > 0) and (idx % verbose == 0):
			print('\t%03d' % idx)
		for itemset in get_mini_powerset(arcs,k):
			itemset_counts[len(itemset)][itemset] += 1
			span_to_itemsets[span_idx][len(itemset)].add(itemset)
	
	for span_idx, count_dicts in span_to_itemsets.items():
		for i in range(1,k+1):
			count_dicts[i] = [arcset for arcset in count_dicts[i] if itemset_counts[i][arcset] >= min_support]
	if verbose:
		print('\tand then the rest')
	setsize = k+1
	spans_to_check = [span_idx for span_idx,span_dict in span_to_itemsets.items() if len(span_dict[k]) > 0]
	while len(spans_to_check) > 0:
		if verbose:
			print('\t',setsize,len(spans_to_check))
		for idx, span_idx in enumerate(spans_to_check):
			if verbose and (idx > 0) and (idx % verbose == 0):
				print('\t%03d' % idx)
			arcs = arc_sets[span_idx]
			if len(arcs) < setsize: continue
			sets_to_check = [set_ for set_ in span_to_itemsets[span_idx].get(setsize-1,[]) 
								if itemset_counts[setsize-1].get(set_,0) >= min_support]
			if len(sets_to_check) == 0: continue
			
			newsets = set()
			for arc in arcs:
				if itemset_counts[1].get((arc,),0) >= min_support:
					for set_ in sets_to_check:
						newset = tuple(sorted(set(set_+ (arc,))))
						if len(newset) == setsize:
							newsets.add(newset)
			for newset in newsets:
				itemset_counts[setsize][newset] += 1
				span_to_itemsets[span_idx][setsize].add(newset)
		for span_idx, count_dicts in span_to_itemsets.items():
			count_dicts[setsize] = [arcset for arcset in count_dicts[setsize] if itemset_counts[setsize][arcset] >= min_support]
		spans_to_check = [span_idx for span_idx,span_dict in span_to_itemsets.items() if len(span_dict[setsize]) > 0]
		setsize+=1
	return itemset_counts, span_to_itemsets

def make_arc_tree(arc_file, outname, min_support=5, verbose=5000):
	'''
		makes the tree of motifs. (G in the paper.)
	'''

	if verbose:
		print('\treading arcs')
	arc_sets = read_arcs(arc_file, verbose)

	if verbose:
		print('\tcounting itemsets')
	itemset_counts, span_to_itemsets = count_frequent_itemsets(arc_sets,min_support)
	new_itemset_counts = {}
	for setsize, size_dict in itemset_counts.items():
		for k,v in size_dict.items():
			if v >= min_support:
				new_itemset_counts[k] = v
	itemset_counts = new_itemset_counts
	itemset_counts[('*',)] = len(arc_sets)
	if verbose:
		print('\twriting itemsets')
	sorted_counts = sorted(itemset_counts.items(),key=lambda x: (-x[1],len(x[0]),x[0][0]))
	with open(outname + '_arc_set_counts.tsv', 'w') as f:
		for k,v in sorted_counts:
			f.write('%d\t%d\t%s\n' % (v, len(k), '\t'.join(k)))
	
	if verbose:
		print('\tbuilding tree')
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

	with open(outname + '_edges.json', 'w') as f:
		f.write('\n'.join(json.dumps(edge) for edge in edges))
	with open(outname + '_uplinks.json', 'w') as f:
		uplink_list = []
		for child, parent_dict in uplinks.items():
			uplink_list.append({'child': child, 'parents': sorted(parent_dict.items(),key=lambda x: x[1]['pr_child'])})
		uplink_list = sorted(uplink_list, key=lambda x: itemset_counts[x['child']], reverse=True)
		f.write('\n'.join(json.dumps(up) for up in uplink_list))
	with open(outname + '_downlinks.json', 'w') as f:
		downlink_list = []
		for parent, child_dict in downlinks.items():
			downlink_list.append({'parent': parent, 'children': sorted(child_dict.items(),key=lambda x: x[1]['pr_child'])})
		downlink_list = sorted(downlink_list, key=lambda x: itemset_counts[x['parent']], reverse=True)
		f.write('\n'.join(json.dumps(down) for down in downlink_list))








