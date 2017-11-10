# Cornell Conversational Analysis Toolkit
This toolkit contains tools to extract conversational features and analyze social phenomena in conversations.  Several large [conversational datasets](http://zissou.infosci.cornell.edu/socialkit/datasets/) are included together with scripts exemplifying the use of the toolkit on these datasets.

The toolkit currently implements features for:

- [Linguistic coordination](http://localhost:8000/Coordination_README.html), a measure of relative power between individuals or groups based on their use of function words (see the [Echoes of Power](https://www.cs.cornell.edu/~cristian/Echoes_of_power.html) paper)
  
- [Question typology](http://localhost:8000/QuestionTypology_README.html), a method for extracting surface motifs that recur in questions, and for grouping them according to their latent rhetorical role (see the [Asking too much](http://www.cs.cornell.edu/~cristian/Asking_too_much.html) paper)

- Coming soon: Politeness, currently available here: [Politeness API](https://github.com/sudhof/politeness)

- Coming soon: Basic message and turn features, currently available here [Constructive conversations](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/tree/constructive/cornellversation/constructive)

## Code
The code for the toolkit can be found [here](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit).

## Usage

**Installation**
1. Install or use `Python 3`.
2. Run `python3 setup.py install` to install the package.
3. Run `python -m spacy download en`

**Use**

Use `import convokit` to import it into your project.

Detailed installation and usage examples are also provided on the specific pages dedicated to each function of this toolkit.

## Documentation
Documentation is hosted [here](http://zissou.infosci.cornell.edu/socialkit/documentation/).

The documentation is built with [Sphinx](http://www.sphinx-doc.org/en/1.5.1/) (`pip3 install sphinx`). To build it yourself, navigate to `doc/` and run `make html`. 

## Acknowledgements

Andrew Wang (azw7@cornell.edu)  wrote the Coordination code and the respective example script, wrote the heper functions and designed the structure of the toolkit.

Ishaan Jhaveri (iaj8@cornell.edu) refactored the Question Typology code and wrote the respective example scripts.
