MathDial Dataset
================

A dataset of 2,861 one-to-one tutoring dialogues where expert human teachers tutor a "student" (simulated by an LLM) through math word problems. The AI student intentionally makes conceptual errors, allowing researchers to study pedagogical strategies and linguistic accommodation.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers include expert human teachers and a central simulated student (``student_ai``).

For each speaker, the following metadata is provided:

* ``is_ai``: Boolean indicating whether the speaker is an AI.
* ``ai_meta``: Additional AI-specific metadata indicating the role.

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conversation flows sequentially.

Metadata for utterances include:

* ``pedagogical_move``: Pedagogical strategies extracted from the teacher's inline annotations (e.g., ``probing``, ``telling``, ``generic``).
* ``ai_meta``: AI-specific metadata.

Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversation corresponds to a single tutoring session on a specific math problem.

* ``topic``: The math problem being solved.
* ``split``: The dataset split (``train`` or ``test``).
* ``student_profile``: The simulated student's incorrect solution profile.
* ``success_state``: Whether the tutoring session was ultimately successful.

Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("mathdial-corpus"))

Statistics about the dataset:

* Number of speakers: 1,112
* Number of conversations: 2,861
* Number of utterances: 35,348

Additional note
---------------

The original MathDial dataset was created by researchers at ETH Zurich.

Contact
^^^^^^^

Please refer to the ConvoKit repository or the original dataset release for more information.
