# Coordination
Linguistic Coordination is measure of relative power between individuals or groups based on their use of function words (see the [Echoes of Power](https://www.cs.cornell.edu/~cristian/Echoes_of_power.html) paper). This readme contains information about  the [Installation](#installation), [Basic Usage](#basic-usage), [Dataset Source](#dataset-source), [Dataset Details](#dataset-details), [Examples](#examples) and [Documentation](#documentation).

## Installation
1. The toolkit requires Python 3. If you don't have it install it by running `pip install python3` or using the Anaconda distribution. That can be found [here](https://www.anaconda.com/download/#macos).
2. Install the required packages by running `pip install -r requirements.txt` (Note if your default version of `pip` is for Python 2.7 you might have to use `pip3 install -r requirements.txt` instead)
3. Run `python3 setup.py install` to install the package.
4.  Use `import convokit` to import it into your project.

## Basic usage
1. Load corpus: `corpus = convokit.Corpus(filename=...)`
2. Create coordination object: `coord = convokit.Coordination(corpus)`
3. Define groups using `corpus.users`:
        `group_A = corpus.users(lambda user: user.info["is-justice"])  # [roberts, ginsburg, ...]`
4. Compute coordination: `scores = coord.score(group_A, group_B)`
5. (Optional) get aggregate scores:
        `average_by_marker_agg1, average_by_marker, agg1, agg2, agg3 = coord.score_report(scores)`


## Dataset Source
TODO
They can all be found [here](zissou.infosci.cornell.edu/data/).

## Dataset Details
TODO: 

## Examples
See [`examples`](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/tree/master/examples/coordination) for guided examples and reproductions of charts from the original papers.

## Documentation
Documentation is hosted [here](http://zissou.infosci.cornell.edu/socialkit/documentation/coordination.html).

The documentation is built with [Sphinx](http://www.sphinx-doc.org/en/1.5.1/) (`pip3 install sphinx`). To build it yourself, navigate to `doc/` and run `make html`. 
