ConvoKit AI
===========

The ConvoKit data model can represent conversations that involve AI agents. On top of the regular
:doc:`Corpus model <model>`, it has:

* **AI speakers**: speakers marked with ``is_ai``, together with the configuration (prompt and
  model) that generates their messages.
* **Private assistants** (:class:`~convokit.model.privateAssistant.PrivateAssistant`): AI assistants
  that privately help one or more speakers, outside the conversation itself.
* **Supports** (:class:`~convokit.model.support.Support`): the messages a private assistant sends to
  the speaker(s) it helps.

These are part of the regular ``Corpus``, ``Speaker``, ``Utterance`` and ``Conversation`` classes,
so a corpus with AI agents works with all ConvoKit transformers. Corpora without AI agents are
unaffected: their speakers have ``is_ai == False`` and empty ``ai_meta``.

Two companion packages build on this format:

* :doc:`ConvoKit AI Simulation <convokitAISimulation>` runs AI agent conversations on
  ConvoArena (built on `Deliberate Lab <https://github.com/PAIR-code/deliberate-lab>`_) and returns
  them as a ConvoKit corpus.
* :doc:`ConvoKit AI Evaluation <convokitAIEvaluation>` provides transformers that evaluate
  AI-mediated conversations.

Installation
------------

The AI fields are part of ConvoKit (version 5.0.0 and later), so installing ConvoKit is enough::

    pip install convokit

.. note::

    Earlier development versions shipped these classes in a separate ``convokitai`` package,
    which has been removed. Replace imports such as ``from convokitai import Corpus`` with
    ``from convokit import Corpus``.

Quick start
-----------

.. code-block:: python

    from convokit import Corpus, download

    corpus = Corpus(filename=download("llm-facilitation-corpus"))
    corpus.print_summary_stats()  # also counts AI speakers, private assistants and supports

    print(corpus.has_ai)  # whether any speaker is an AI agent
    convo = next(corpus.iter_conversations())

    # AI speakers, and the names speakers go by in the conversation
    for speaker in convo.iter_speakers():
        print(convo.alias.get(speaker.id, speaker.id), speaker.is_ai, speaker.ai_meta.get("role"))

    # the conversation, with each private assistant's supports shown under the message they reply to
    print(convo.get_transcript(supports=True))

    # all supports across the corpus
    for support in corpus.iter_supports():
        print(support.private_assistant_id, support.text)

The AI fields can be set when building a corpus:

.. code-block:: python

    from convokit import Corpus, PrivateAssistant, Speaker, Support, Utterance

    bot = Speaker(id="bot", is_ai=True, ai_meta={"role": "public assistant"})
    corpus = Corpus(utterances=[...])
    convo = corpus.get_conversation("c1")
    convo.alias = {"alice": "Bear", "bot": "Mediator"}
    convo.private_assistants = [PrivateAssistant(id="pa", speakers=["alice"], conversation_id="c1")]

AI speakers and private assistants with a :ref:`generation config <convokitai-generation-config>`
can also produce new messages with :doc:`convokit.genai <genai>` (the API key for the model's
provider must be configured; see :doc:`GenAI <genai>`):

.. code-block:: python

    # the next message from an AI speaker, given the conversation so far
    utterance = speaker.generate(conversation=convo, append=True)

    # a private assistant's support for the speaker it helps, given what they have drafted
    support = assistant.generate(conversation=convo, draft="I think we should")

Data format
-----------

The AI fields are added to the standard ConvoKit data format. In memory, they are attributes of
each object (``speaker.is_ai``, ``speaker.ai_meta``, ``conversation.alias``, ...). On disk, they are
stored in the regular metadata (``meta["is_ai"]``, ``meta["ai_meta"]``, ...), and they are moved out
of ``meta`` into the attributes when the corpus is loaded. See
:ref:`ConvoKit AI extensions <convokitai-data-format>` in :doc:`Data Format <data_format>`.

With the ``db`` :doc:`storage backend <storage_options>`, the AI fields are also written to the
database, so ``Corpus.reconnect_to_db`` restores them. As with metadata in DB mode, a change is
saved when a field is assigned (e.g. ``convo.alias = {...}``, ``speaker.ai_meta = {...}``,
``utt.supports = [...]``); changing a returned value in place (e.g.
``convo.private_assistants[0].supports.append(...)``) only changes the copy in memory.

API reference
-------------

The AI fields and methods are documented with the classes they belong to:

* :doc:`Corpus <corpus>`: ``has_ai``, ``ai_meta``, ``iter_private_assistants``,
  ``get_private_assistant``, ``iter_supports``
* :doc:`Conversation <conversation>`: ``ai_meta``, ``alias``, ``private_assistants``,
  ``supports``, ``get_private_assistant``, ``get_ai_speakers``, ``get_transcript``
* :doc:`Speaker <speaker>`: ``is_ai``, ``ai_meta``, ``generate``
* :doc:`Utterance <utterance>`: ``ai_meta``, ``supports``, ``get_transcript``
* :doc:`PrivateAssistant <privateAssistant>`
* :doc:`Support <aiSupport>`
