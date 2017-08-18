import os, json


def read_arcs(arc_file, verbose=5000):
	arc_sets = {}
	with open(arc_file) as f:
		for idx,line in enumerate(f.readlines()):
			if (idx > 0) and (idx % verbose == 0):
				print('\t%03d' % idx)
			entry = json.loads(line)
			arc_sets[entry['idx']] = entry['arcs']
	return arc_sets

def read_uplinks(uplink_file):
	uplinks = {}
	with open(uplink_file) as f:
		for line in f.readlines():
			entry = json.loads(line)
			uplinks[tuple(entry['child'])] = [(tuple(x),y) for x,y in entry['parents']]
	return uplinks

def read_downlinks(downlink_file):
	downlinks = {}
	with open(downlink_file) as f:
		for line in f.readlines():
			entry = json.loads(line)
			downlinks[tuple(entry['parent'])] = [(tuple(x),y) for x,y in entry['children']]
	return downlinks

def read_nodecounts(nodecount_file):

	node_counts = {}
	with open(nodecount_file) as f:
		for line in f:
			split = line.split('\t')
			count = int(split[0])
			set_size = int(split[1])
			itemset = tuple([x.strip() for x in split[2:]])
			node_counts[itemset] = count
	return node_counts