ConvoKit AI Evaluation
======================

ConvoKit AI Evaluation (the ``convokitai_evaluation`` package) provides transformers that evaluate
mediated multi-party conversations, such as discussions moderated by an AI facilitator:

* **CIG** (conversational information gain): an LLM rates each utterance from 1 to 4 for how much it
  adds to what the group already knew.
* **ClaimTracker**: tracks the atomic claims each speaker holds through the conversation, then
  measures how far the participants converge and whether their final claims came from the mediator
  or from another participant.
* **MediatorRedirection**: for each mediator turn, a language model estimates how much it changed the
  participant reply that followed, using :doc:`Redirection <redirectionAndUtteranceLikelihood>`.

Example usage:
`CIG demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convokitai_evaluation/examples/cig_demo.ipynb>`_,
`ClaimTracker demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convokitai_evaluation/examples/claims_demo.ipynb>`_,
`MediatorRedirection demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convokitai_evaluation/examples/redirection_demo.ipynb>`_.

Installation
------------

.. important::

   ConvoKit AI Evaluation requires **Python 3.12 or above**. This is newer than core ConvoKit, which
   supports Python 3.10 and above, so an environment that works for ConvoKit may be too old for
   this package. Check your version with ``python --version``, and if it is below 3.12, create a new
   environment first, for example::

       python3.12 -m venv .venv
       source .venv/bin/activate

   or with conda::

       conda create -n convokitai python=3.12
       conda activate convokitai

   Installing into an older Python fails with an error that the package
   ``requires a different Python``.

Then, from the repository root, install it together with ConvoKit::

    python -m pip install -e . -e ./convokit/convokitai_evaluation

or from GitHub::

    pip install "git+https://github.com/CornellNLP/ConvoKit.git@master#subdirectory=convokit/convokitai_evaluation"

CIG and ClaimTracker call OpenAI models through :doc:`convokit.genai <genai>`, so they need an
OpenAI API key (``GPT_API_KEY``, or set with ``GenAIConfigManager``). ClaimTracker also downloads a
sentence embedding model and an NLI model from Hugging Face on first use. MediatorRedirection needs
ConvoKit's optional LLM dependencies (``pip install convokit[llm]``) and a Hugging Face token for the
Gemma model.

Input format
------------

The transformers read the :doc:`AI fields <convokitai>` of the corpus:

* **The mediator** is the speaker whose ``ai_meta["role"]`` is ``"public assistant"``. All other
  speakers are participants.
* **Speaker names** come from ``conversation.alias``, so the prompts show messages under the names
  participants used in the chat.
* **The topic** shown to the models is ``conversation.meta["topic"]``, followed by
  ``conversation.meta["statement"]`` if there is one.
* **Starting positions** for ClaimTracker come from ``conversation.meta["pre_survey"]``, a dict from
  speaker ID to the text of that participant's position before the conversation.

Corpora produced by :doc:`ConvoKit AI Simulation <convokitAISimulation>` already have the mediator
role and names. For other corpora, each transformer takes functions that replace these defaults:
``is_mediator`` (Speaker → bool), ``name_func`` ((Conversation, Speaker) → str), ``topic_func``
(Conversation → str) and, for ClaimTracker, ``seed_text_func`` ((Conversation, Speaker) → str).
For example, to treat both AI and human facilitators as mediators:

.. code-block:: python

    from convokitai_evaluation import CIG

    cig = CIG(
        model="gpt-5",
        is_mediator=lambda speaker: speaker.meta.get("game_role") == "facilitator",
        topic_func=lambda convo: "Choosing a host city for a sporting event",
    )
    cig.transform(corpus)
    cig.summarize(corpus)  # mean CIG per conversation, mediator vs participants

CIG
---

.. automodule:: convokitai_evaluation.cig
    :members:

ClaimTracker
------------

.. automodule:: convokitai_evaluation.claims
    :members:

MediatorRedirection
-------------------

.. automodule:: convokitai_evaluation.redirection
    :members:

LLM client
----------

.. automodule:: convokitai_evaluation.llm
    :members:

Defaults for reading the ConvoKit AI format
-------------------------------------------

.. automodule:: convokitai_evaluation.util
    :members:
