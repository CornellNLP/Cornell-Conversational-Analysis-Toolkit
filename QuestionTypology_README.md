# Question Typology
Question typology is a method for extracting surface motifs that recur in questions, and for grouping them according to their latent rhetorical role (see the [Asking too much](http://www.cs.cornell.edu/~cristian/Asking_too_much.html) paper). This readme contains information about  the [Installation](#installation), [Basic Usage](#basic-usage), [Dataset Source](#dataset-source), [Dataset Details](#dataset-details), [Examples](#examples) and [Documentation](#documentation).

## Installation
1. The toolkit requires Python 3. If you don't have it install it by running `pip install python3` or using the Anaconda distribution. That can be found [here](https://www.anaconda.com/download/#macos).
2. Install the required packages by running `pip install -r requirements.txt` (Note if your default version of `pip` is for Python 2.7 you might have to use `pip3 install -r requirements.txt` instead)
3. Run `python3 setup.py install` to install the package.
4.  Use `import convokit` to import it into your project.

## Basic usage
1. Load corpus: `corpus = convokit.Corpus(filename=...)`
2. Create QuestionTypology object (discover typology): `questionTypology = QuestionTypology(`
3. Explore 10 questions of type `type_num`: `questionTypology.display_questions_for_type(type_num, 10)`
4. Explore 10 resulting motifs of type `type_num`: `questionTypology.display_motifs_for_type(cluster_num, 10)`
5. Explore 10 resulting answer fragments from answers to questions of type `type_num`: `questionTypology.display_answer_fragments_for_type(cluster_num, 10)`
6. Explore 10 question-answer pairs from the training data of type `type_num`: `questionTypology.display_question_answer_pairs_for_type(cluster_num, 10)`

## Dataset Source
The datasets currently released with the Question typology functionality are the UK Parliament Question Answer Sessions Dataset, the Wimbledon Winner Interviews Dataset and the Wikipedia Moderators Dataset.

They can all be found [here](zissou.infosci.cornell.edu/data/askingtoomuch/).

## Dataset Details

TODO: Where to host the text files

## Examples
See [`examples`](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/tree/master/examples/questionTypology) for guided examples and reproductions of charts from the original papers.

## Documentation
Documentation is hosted [here](http://zissou.infosci.cornell.edu/socialkit/documentation/questionTypology.html).

The documentation is built with [Sphinx](http://www.sphinx-doc.org/en/1.5.1/) (`pip3 install sphinx`). To build it yourself, navigate to `doc/` and run `make html`. 
