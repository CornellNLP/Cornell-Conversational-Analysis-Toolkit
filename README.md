# Cornell Conversational Analysis Toolkit
This toolkit contains tools to extract conversational features and analyze social phenomena in conversations.  Several large [conversational datasets](http://zissou.infosci.cornell.edu/socialkit/datasets/) are included together with scripts exemplifying the use of the toolkit on these datasets.

Currently implements features for:

- Linguistic coordination, a measure of relative power between individuals or
  groups based on their use of function words (see the [Echoes of
  Power](https://www.cs.cornell.edu/~cristian/Echoes_of_power.html) paper)
  
- Question typology, a method for extracting surface motifs that recur in questions, and for grouping them according to their latent rhetorical role (see the [Asking too much](http://www.cs.cornell.edu/~cristian/Asking_too_much.html) paper)

- Coming soon: Politeness, currently available here: [Politeness API](https://github.com/sudhof/politeness)

- Coming soon: Basic message and turn features, currently available here [Constructive conversations](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/tree/constructive/cornellversation/constructive)

## Installing
The toolkit requires Python 3. Run `python3 setup.py install` to install the package.
Use `import convokit` to import it into your project.

## Examples
See `examples/` for guided examples and reproductions of charts from the original
papers.

## Documentation
Documentation is hosted [here](http://zissou.infosci.cornell.edu/socialkit/documentation/).

The documentation is built with [Sphinx](http://www.sphinx-doc.org/en/1.5.1/) (`pip3 install sphinx`). To build it yourself, navigate to `doc/` and run `make html`. 

## Basic usage
### Coordination
1. Load corpus: `corpus = convokit.Corpus(filename=...)`
2. Create coordination object: `coord = convokit.Coordination(corpus)`
3. Define groups using `corpus.users`:
        `group_A = corpus.users(lambda user: user.info["is-justice"])  # [roberts, ginsburg, ...]`
4. Compute coordination: `scores = coord.score(group_A, group_B)`
5. (Optional) get aggregate scores:
        `average_by_marker_agg1, average_by_marker, agg1, agg2, agg3 = coord.score_report(scores)`

### Question Typology
1. Load corpus: `corpus = convokit.Corpus(filename=...)`
2. Create QuestionTypology object (discover typology): `questionTypology = QuestionTypology(`
3. Explore resulting types `questionTypology.display_questions_for_type(i, num_egs=10)`


## Acknowledgements

Andrew Wang (azw7@cornell.edu)  wrote the Coordination code and the respective example script, wrote the heper functions and designed the structure of the toolkit.

Ishaan Jhaveri (iaj8@cornell.edu) refactored the Question Typology code and wrote the respective example scripts.
