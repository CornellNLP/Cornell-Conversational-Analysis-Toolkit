import os
from questionTypology.clusters.build_joint_matrices import build_joint_matrix
from questionTypology.clusters.cluster_functions import run_simple_pipe, run_lowdim_pipe, inspect_kmeans_run
from sklearn.externals import joblib


def build_matrix(motif_dir, matrix_dir, question_threshold, answer_threshold):
	'''
		convenience pipeline to build the question answer matrices. 
		motif_dir: wherever extract_motifs wrote to
		matrix_dir: where to put the matrices
		question_threshold: minimum # of questions in which a question motif has to occur to be considered
	'''
	print('building q-a matrices')
	question_fit_file = os.path.join(motif_dir, 'question_fits.json.super')
	answer_arc_file = os.path.join(motif_dir, 'answer_arcs.json')
	superset_file = os.path.join(motif_dir, 'question_supersets_arcset_to_super.json')

	try:
		os.mkdir(matrix_dir)
	except:
		print('matrix dir %s exists!' % matrix_dir)

	outfile = os.path.join(matrix_dir, 'qa_mtx')
	build_joint_matrix(question_fit_file, answer_arc_file,superset_file, outfile, question_threshold, answer_threshold, verbose=5000)



def extract_clusters(matrix_dir,km_file,k=8, d=25,num_egs=10):
	'''
		convenience pipeline to get latent q-a dimensions and clusters. 

		km_file: where to write the kmeans object
		k: num clusters
		d: num latent dims
		
	'''
	matrix_file = os.path.join(matrix_dir, 'qa_mtx')
	q_mtx, a_mtx, mtx_obj = run_simple_pipe(matrix_file)
	lq, a_u, a_s, a_v = run_lowdim_pipe(q_mtx,a_mtx,d)
	km = inspect_kmeans_run(lq,a_u,d,k,mtx_obj['q_terms'], mtx_obj['a_terms'], num_egs=num_egs)
	joblib.dump(km, km_file)

