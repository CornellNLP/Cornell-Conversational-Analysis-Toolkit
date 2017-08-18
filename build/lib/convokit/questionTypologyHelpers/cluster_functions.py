import numpy as np
from scipy import sparse
import json, os
from ast import literal_eval as make_tuple

from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import pairwise_distances

def load_joint_mtx(rootname):
	mtx_obj = {}
	#rootname = os.path.join(LATENT_DIR, data_name, feature_name)

	print('reading question tidxes')
	mtx_obj['q_tidxes'] = np.load(rootname + '.q.tidx.npy')
	print('reading question leaves')
	mtx_obj['q_leaves'] = np.load(rootname + '.q.leaves.npy')
	print('reading answer tidxes')
	mtx_obj['a_tidxes'] = np.load(rootname + '.a.tidx.npy')

	print('reading question didxes')
	mtx_obj['q_didxes'] = np.load(rootname + '.q.didx.npy')
	print('reading answer didxes')
	mtx_obj['a_didxes'] = np.load(rootname + '.a.didx.npy')

	print('reading question terms')
	mtx_obj['q_terms'] = []
	mtx_obj['q_term_to_idx'] = {}
	mtx_obj['q_term_counts'] = []
	fname = rootname + '.q.terms.txt'
	with open(fname) as f:
		for idx, line in enumerate(f.readlines()):
			count,term = line.split('\t')
			term = term.strip()
			term = make_tuple(term)
			mtx_obj['q_term_counts'].append(int(count))
			mtx_obj['q_terms'].append(term)
			mtx_obj['q_term_to_idx'][term] = idx
	mtx_obj['q_terms'] = np.array(mtx_obj['q_terms'])
	mtx_obj['q_term_counts'] = np.array(mtx_obj['q_term_counts'])

	print('reading answer terms')
	mtx_obj['a_terms'] = []
	mtx_obj['a_term_to_idx'] = {}
	mtx_obj['a_term_counts'] = []
	fname = rootname + '.a.terms.txt'
	with open(fname) as f:
		for idx, line in enumerate(f.readlines()):
			count,term = line.split('\t')
			term = term.strip()
			mtx_obj['a_term_counts'].append(int(count))
			mtx_obj['a_terms'].append(term)
			mtx_obj['a_term_to_idx'][term] = idx
	mtx_obj['a_terms'] = np.array(mtx_obj['a_terms'])
	mtx_obj['a_term_counts'] = np.array(mtx_obj['a_term_counts'])

	print('reading docs')
	mtx_obj['docs'] = []
	mtx_obj['doc_to_idx'] = {}
	with open(rootname + '.docs.txt') as f:
		for idx, line in enumerate(f.readlines()):
			doc_id = line.strip()
			mtx_obj['docs'].append(doc_id)
			mtx_obj['doc_to_idx'][doc_id] = idx 
	mtx_obj['docs'] = np.array(mtx_obj['docs'])

	print('done!')
	return mtx_obj

def build_mtx(mtx_obj, data_type, norm='l2', idf=False, leaves_only=False):
	N_terms = len(mtx_obj[data_type + '_terms'])
	N_docs = len(mtx_obj['docs'])
	if idf:
		data = np.log(N_docs) - np.log(mtx_obj[data_type + '_term_counts'][mtx_obj[data_type + '_tidxes']])
	else:
		data = np.ones_like(mtx_obj[data_type + '_tidxes'])
		if leaves_only:
			data[~mtx_obj[data_type + '_leaves']] = 0
	mtx = sparse.csr_matrix((data, (mtx_obj[data_type + '_tidxes'], mtx_obj[data_type + '_didxes'])), shape=(N_terms,N_docs))
	if norm:
		mtx = Normalizer(norm=norm).fit_transform(mtx)
	
	return mtx

def run_simple_pipe(rootname, verbose=True):
	mtx_obj = load_joint_mtx(rootname)
	q_mtx = build_mtx(mtx_obj, 'q')
	a_mtx = build_mtx(mtx_obj, 'a', idf=True)
	return q_mtx, a_mtx, mtx_obj
def do_sparse_svd(mtx, k=50):
	u,s,v = sparse.linalg.svds(mtx, k=k) # ugh, right order dammit
	return u[:,::-1],s[::-1],v[::-1,:]
def run_lowdim_pipe(q_mtx, a_mtx, k=50, snip=True):
	a_u, a_s, a_v = do_sparse_svd(a_mtx,k + int(snip))
	lq = q_mtx * (a_v.T * a_s**-1)
	if snip:
		return snip_first_dim(lq, a_u, a_s, a_v)
	else:
		return lq, a_u, a_s, a_v

def inspect_latent_space(mtx, names, dim_iter=None, num_dims=5, num_egs=10, which_end=None, skip_first=True, dim_names={},s=None):
	mtx = Normalizer().fit_transform(mtx).T
	if dim_iter is None:
		dim_iter = range(int(skip_first), num_dims + int(skip_first))
	for dim in dim_iter:
		if s is not None:
			print(dim,s[dim])
		else:
			print(dim)
		row = mtx[dim]
		argsorted = np.argsort(row)
		if (not which_end) or (which_end == -1):
			print('\tbottom',dim_names.get((dim,-1), ''))
			for i in range(num_egs):
				print('\t\t',names[argsorted[i]], '%+.4f' % row[argsorted[i]])
		if (not which_end) or (which_end == 1):
			print('\ttop',dim_names.get((dim,1), ''))
			for i in range(num_egs):
				print('\t\t',names[argsorted[-1-i]], '%+.4f' % row[argsorted[-1-i]])
		print()


def run_kmeans(X, in_dim, k):
	km = KMeans(n_clusters=k,max_iter=1000)
	km.fit(X)
	return km

def inspect_kmeans_run(q_mtx, a_mtx, num_svd_dims, num_clusters, q_terms, a_terms, km=None, remove_first=False, num_egs=10):
	if remove_first:
		q_mtx = q_mtx[:,1:(num_svd_dims + 1)]
		a_mtx = a_mtx[:,1:(num_svd_dims + 1)]
	else:
		q_mtx = q_mtx[:,:num_svd_dims]
		a_mtx = a_mtx[:,:num_svd_dims]
	q_mtx = Normalizer().fit_transform(q_mtx)
	a_mtx = Normalizer().fit_transform(a_mtx)
	if km:
		q_km = km
	else:
		q_km = run_kmeans(q_mtx, num_svd_dims, num_clusters)
	if num_egs > 0:
		q_dists = q_km.transform(q_mtx)
		q_assigns = q_km.labels_
		a_dists = q_km.transform(a_mtx)
		a_assigns = q_km.predict(a_mtx)
		for cl in range(num_clusters):
			print('cluster',cl)
			q_assigned = q_assigns == cl
			median_qdist = np.median(q_dists[:,cl][q_assigned])
			print('\tq assigns:',q_assigned.sum(),  'median dist:', '%.4f' % median_qdist)
			a_assigned = a_assigns == cl
			median_adist = np.median(a_dists[:,cl][a_assigned])
			print('\ta assigns:',a_assigned.sum(),  'median dist:', '%.4f' % median_adist)
			if num_egs == 0: continue
			argsorted_qdists = np.argsort(q_dists[:,cl])
			argsorted_qdists = argsorted_qdists[np.in1d(argsorted_qdists, np.where(q_assigned)[0])]
			print('\tqs:')
			for i in range(min(num_egs,q_assigned.sum())):
				curr_qdist = q_dists[:,cl][argsorted_qdists[i]]
				if curr_qdist > median_qdist:
					diststr = '%.4f ~~' %  curr_qdist
				else:
					diststr = '%.4f' % curr_qdist
				print('\t\t', q_terms[argsorted_qdists[i]], diststr)
			argsorted_adists = np.argsort(a_dists[:,cl])
			argsorted_adists = argsorted_adists[np.in1d(argsorted_adists, np.where(a_assigned)[0])]
			print('\tas:')
			for i in range(min(num_egs,a_assigned.sum())):
				curr_adist = a_dists[:,cl][argsorted_adists[i]]
				if curr_adist > median_adist:
					diststr = '%.4f ~~' %  curr_adist
				else:
					diststr = '%.4f' % curr_adist
				print('\t\t', a_terms[argsorted_adists[i]], diststr)
			print('========================')
	return q_km

def snip_first_dim(lq, a_u, a_s, a_v):
	return lq[:,1:], a_u[:,1:], a_s[1:], a_v[1:]

def assign_clusters(km, lq, a_u, mtx_obj, n_dims):
	km_qdists = km.transform(Normalizer().fit_transform(lq[:,:n_dims]))
	km_qlabels = km.predict(Normalizer().fit_transform(lq[:,:n_dims]))
	km_adists = km.transform(Normalizer().fit_transform(a_u[:,:n_dims]))
	km_alabels = km.predict(Normalizer().fit_transform(a_u[:,:n_dims]))

	motif_df_entries = []
	for idx, motif in enumerate(mtx_obj['q_terms']):
	    entry = {'idx': idx, 'motif': motif, 'cluster': km_qlabels[idx],
	            'count': mtx_obj['q_term_counts'][idx]}
	    entry['cluster_dist'] = km_qdists[idx,entry['cluster']]
	    motif_df_entries.append(entry)
	motif_df = pd.DataFrame(motif_df_entries).set_index('idx')

	aarc_df_entries = []
	for idx, aarc in enumerate(mtx_obj['a_terms']):
	    entry = {'idx': idx, 'aarc': aarc, 'cluster': km_alabels[idx], 
	            'count': mtx_obj['a_term_counts'][idx]}
	    entry['cluster_dist'] = km_adists[idx,entry['cluster']]
	    aarc_df_entries.append(entry)
	aarc_df = pd.DataFrame(aarc_df_entries).set_index('idx')

	q_leaves = build_mtx(mtx_obj,'q',leaves_only=True)
	qdoc_vects = q_leaves.T * Normalizer().fit_transform(parl_lq)
	km_qdoc_dists = km.transform(Normalizer().fit_transform(qdoc_vects[:,:n_dims]))
	km_qdoc_labels = km.predict(Normalizer().fit_transform(qdoc_vects[:,:n_dims]))
	qdoc_df_entries = []
	for idx, qdoc in enumerate(mtx_obj['docs']):
	    entry = {'idx': idx, 'q_idx': qdoc, 'cluster': km_qdoc_labels[idx]}
	    entry['cluster_dist'] = km_qdoc_dists[idx,entry['cluster']]
	    qdoc_df_entries.append(entry)
	qdoc_df = pd.DataFrame(qdoc_df_entries).set_index('idx')

	return motif_df, aarc_df, qdoc_df
