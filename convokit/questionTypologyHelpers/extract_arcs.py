import os, json
from questionTypology.utils.spacy_utils import get_spacy_dict, load_vocab

from spacy.symbols import *
NP_LABELS = set([nsubj, nsubjpass, dobj, iobj, pobj, attr])

def is_noun_ish(word):
	return (word.dep in NP_LABELS) or (word.tag_.startswith('NN') or word.tag_.startswith('PRP')) or (word.tag_.endswith('DT'))

def has_w_det(token):
	if token.tag_.startswith('W'): return token.text
	first_tok = next(token.subtree)
	if (first_tok.tag_.startswith('W')): return first_tok.text
	return False

def get_tok(token):
	if is_noun_ish(token):
		has_w = has_w_det(token)
		if has_w:
			return has_w.lower(), True
		else:
			return 'NN', True
	else:
		return token.text.lower(), False

def get_clean_tok(tok):
	out_tok, is_noun = get_tok(tok)
	return out_tok.replace('--','').strip(), is_noun

def is_alpha_ish(text):
	return text.isalpha() or text[1:].isalpha()

def is_usable(text):
	return is_alpha_ish(text) and (text != 'NN')


def get_arcs(root, follow_conj):

	# todo: could imagine version where nouns allowed
	arcs = set()
	root_tok, _ = get_clean_tok(root)
	if not is_usable(root_tok): return arcs

	arcs.add(root_tok + '_*')
	conj_elems = []
	for idx, kid in enumerate(root.children):
		if kid.dep_ in ['punct','cc']:
			continue
		elif kid.dep_ == 'conj':
			if follow_conj:
				conj_elems.append(kid)
		else:
			kid_tok, _ = get_clean_tok(kid)
			if is_usable(kid_tok):
				arcs.add(root_tok + '_' + kid_tok)

	first_elem = next(root.subtree)
	first_tok, _ = get_clean_tok(first_elem)
	if is_usable(first_tok):
		arcs.add(first_tok + '>*')
		try:
			second_elem = first_elem.nbor()
			second_tok, _ = get_clean_tok(second_elem)
			if is_usable(second_tok):
				arcs.add(first_tok + '>' + second_tok)
		except:
			pass

	for conj_elem in conj_elems:
		arcs.update(get_arcs(conj_elem, follow_conj))
	return arcs

def is_question(span):
	span_text = span.text.strip()
	return span_text[-1] == '?'

def extract_arcs(text_iter, spacy_filename, outfile, vocab, use_span=is_question ,
	follow_conj=True, verbose=5000):

	'''
		extracts all arcs going out of the root in a sentence. used to find question motifs.

		text_iter: iterates over text for which arcs are extracted
		spacy_filename: location of spacy objects (from spacy_utils.py)
		outfile: where to write the arcs.
		vocab: pre-loaded spacy vocabulary. if you pass None it will load vocab for you, but that's slow.
		use_span: filter to decide which sentences to use. the function takes in a spacy sentence object.
		follow_conj: whether to follow conjunctions and treat subtrees as sentences too.

	'''

	if verbose:
		print('reading spacy')
	spacy_dict = get_spacy_dict(spacy_filename, vocab)

	arc_entries = []
	for idx, (text_idx,text) in enumerate(text_iter):
		if verbose and (idx > 0) and (idx % verbose == 0):
			print('\t%03d' % idx)
		spacy_obj = spacy_dict[text_idx]
		for span_idx, span in enumerate(spacy_obj.sents):
			if use_span(span):
				curr_arcset = get_arcs(span.root, follow_conj)
				arc_entries.append({'idx': '%s_%d' % (text_idx, span_idx), 'arcs': list(curr_arcset)})
	if verbose:
		print('\twriting arcs')
	with open(outfile, 'w') as f:
		f.write('\n'.join(json.dumps(arc_entry) for arc_entry in arc_entries))




