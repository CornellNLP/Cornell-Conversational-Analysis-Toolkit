import os, json
from questionTypology.utils.spacy_utils import load_vocab
from questionTypology.motifs.extract_arcs import extract_arcs, is_question
from questionTypology.motifs.make_arc_tree import make_arc_tree
from questionTypology.motifs.fit_questions import fit_all
from questionTypology.motifs.deduplicate_motifs import deduplicate_motifs, postprocess_fits

def is_uppercase_question(x):
	'''
		for reasonably well-formatted datasets like transcripts of some proceedings, i've included this filter that questions start w/ uppercase and end in a question mark. this filter can be varied/swapped out.
	'''
	text = x.text.strip()
	return (text[-1] == '?') and (text[0].isupper())

def is_uppercase(x):
	'''
		mainly because we otherwise get a bunch of badly parsed half-lines,
		enforce that answer sentences have to start in uppercase (reliable
		provided your data is well-formatted...)
	'''
	text = x.text.strip()
	return text[0].isupper()


def extract_question_motifs(question_text_iter, spacy_filename, motif_dir,
	question_filter_fn = is_uppercase_question,
	follow_conj=True,
	min_question_itemset_support=5,
	deduplicate_threshold=.9,
	verbose=5000):
	'''
		convenience pipeline to get question motifs. (see pipelines/extract_*_motifs for examples)
		question_text_iter: iterates over all questions
		spacy_filename: location of spacy objects
		motif_dir: directory where all motifs written
		question_filter_fn: only uses sentences in a question which corresponds to a question. can redefine.
		follow_conj: follows conjunctions to compound questions ("why...and how")
		min_question_itemset_support: the minimum number of times an itemset has to show up for the frequent itemset counter to consider it.
		deduplicate_threshold: how often two motifs co-occur (i.e. p(x|y) and p(y|x) for us to consider them redundant)
	'''
	print('running motif extraction pipeline')

	try:
		os.mkdir(motif_dir)
	except:
		print('\tmotif dir %s exists!' % motif_dir)

	print('loading spacy vocab')
	vocab = load_vocab()

	print('getting question arcs')
	question_arc_outfile = os.path.join(motif_dir, 'question_arcs.json')
	extract_arcs(question_text_iter, spacy_filename, question_arc_outfile, vocab, use_span=question_filter_fn, follow_conj=follow_conj, verbose=verbose)

	print('making motif tree')
	question_tree_outfile = os.path.join(motif_dir, 'question_tree')
	make_arc_tree(question_arc_outfile, question_tree_outfile, min_question_itemset_support, verbose=verbose)

	print('fitting motifs to questions')
	question_fit_outfile = os.path.join(motif_dir, 'question_fits.json')
	fit_all(question_arc_outfile, question_tree_outfile, question_fit_outfile, verbose=verbose)

	print('handling redundant motifs')
	question_super_outfile = os.path.join(motif_dir, 'question_supersets')
	deduplicate_motifs(question_fit_outfile, question_super_outfile, deduplicate_threshold, verbose=verbose)
	postprocess_fits(question_fit_outfile, question_tree_outfile, question_super_outfile + '_arcset_to_super.json')

	print('done motif extraction')

def extract_answer_arcs(answer_text_iter, spacy_filename, motif_dir, answer_filter_fn=is_uppercase, follow_conj=True, verbose=5000):
	'''
		convenience pipeline to get answer motifs
	'''

	print('running answer arc pipeline')
	try:
		os.mkdir(motif_dir)
	except:
		print('\tmotif dir %s exists!' % motif_dir)

	print('loading spacy vocab')
	vocab = load_vocab()

	print('getting answer arcs')
	answer_arc_outfile = os.path.join(motif_dir, 'answer_arcs.json')
	extract_arcs(answer_text_iter, spacy_filename, answer_arc_outfile, vocab, use_span=answer_filter_fn, follow_conj=follow_conj, verbose=verbose)

	print('done answer arc extraction')
