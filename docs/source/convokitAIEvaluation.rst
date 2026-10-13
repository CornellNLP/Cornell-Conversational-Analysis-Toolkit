ConvoKit AI Evaluation
======================

ConvoKit AI Evaluation (the ``convokit.ai_evaluation`` package) provides transformers that evaluate
mediated multi-party conversations, such as discussions moderated by an AI facilitator:

* **CIG** (conversational information gain): an LLM rates each utterance from 1 to 4 for how much it
  adds to what the group already knew.
* **ClaimTracker**: tracks the atomic claims each speaker holds through the conversation, then
  measures how far the participants converge and whether their final claims came from the mediator
  or from another participant.
* **MediatorRedirection**: for each mediator turn, a language model estimates how much it changed the
  participant reply that followed, using :doc:`Redirection <redirectionAndUtteranceLikelihood>`.

Example usage:
`CIG demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/ai_evaluation/examples/cig_demo.ipynb>`_,
`ClaimTracker demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/ai_evaluation/examples/claims_demo.ipynb>`_,
`MediatorRedirection demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/ai_evaluation/examples/redirection_demo.ipynb>`_.

Installation
------------

ConvoKit AI Evaluation is part of ConvoKit. CIG and ClaimTracker need ConvoKit's GenAI
dependencies::

    pip install "convokit[genai]"

Then import the transformers from ``convokit.ai_evaluation``::

    from convokit.ai_evaluation import CIG, ClaimTracker, MediatorRedirection

CIG and ClaimTracker call an OpenAI or a Gemini model through :doc:`convokit.genai <genai>`, chosen
by the ``model`` argument: ``gpt-*`` models (the default, ``gpt-5`` for CIG and ``gpt-5-mini`` for
ClaimTracker) need an OpenAI API key, and ``gemini-*`` models need a Gemini API key. Set the key with
the ``GPT_API_KEY`` / ``GEMINI_API_KEY`` environment variable or with ``GenAIConfigManager``
(``config_manager.set_api_key("gemini", "...")``); Gemini can also run through Vertex AI (see
:doc:`GenAI <genai>`). Scores from different models are not directly comparable: the prompts were
written and checked with GPT-5 models, so use one model throughout a study. ClaimTracker also
downloads a sentence embedding model and an NLI model from Hugging Face on first use. MediatorRedirection needs
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

    from convokit.ai_evaluation import CIG

    cig = CIG(
        model="gpt-5",
        is_mediator=lambda speaker: speaker.meta.get("game_role") == "facilitator",
        topic_func=lambda convo: "Choosing a host city for a sporting event",
    )
    cig.transform(corpus)
    cig.summarize(corpus)  # mean CIG per conversation, mediator vs participants

To use Gemini instead, pass a Gemini model name (the provider is inferred from it, or can be given
with ``provider="gemini"``):

.. code-block:: python

    from convokit.ai_evaluation import CIG, ClaimTracker

    cig = CIG(model="gemini-2.5-flash")
    tracker = ClaimTracker(model="gemini-2.5-flash")

CIG
---

.. automodule:: convokit.ai_evaluation.cig
    :members:

ClaimTracker
------------

.. automodule:: convokit.ai_evaluation.claims
    :members:

MediatorRedirection
-------------------

.. automodule:: convokit.ai_evaluation.redirection
    :members:

LLM client
----------

.. automodule:: convokit.ai_evaluation.llm
    :members:

Defaults for reading the ConvoKit AI format
-------------------------------------------

.. automodule:: convokit.ai_evaluation.util
    :members:
