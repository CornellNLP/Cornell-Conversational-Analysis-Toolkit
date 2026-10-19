Habermas Machine Dataset
========================

A dataset of group deliberations where human participants interact with an AI mediator (the Habermas Machine) to reach a consensus. The AI mediator helps small groups of participants find common ground on various topics by synthesizing their initial opinions and subsequent critiques.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers include human participants, the ``ai_mediator``, and a ``system_moderator``.

For each speaker, the following metadata is provided:

* ``is_ai``: Boolean indicating whether the speaker is an AI.
* ``ai_meta``: Additional AI-specific metadata indicating the role.

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance corresponds to a step in the deliberation process. The conversation tree is structured as a Directed Acyclic Graph (DAG) using ConvoKit's ``reply_to`` and ``meta`` fields to preserve many-to-one relationships (e.g. the AI synthesizing multiple opinions).

Metadata for utterances include:

* ``phase``: The phase of the deliberation (e.g., ``initial_opinion``, ``ai_synthesis_1``, ``critique``, ``ai_synthesis_2``, ``final_synthesis``).
* ``ai_meta``: AI-specific metadata.
* ``synthesizes``: (For AI synthesis utterances) A list of utterance IDs that this statement synthesizes.
* ``synthesizes_critiques``: (For revised AI synthesis utterances) A list of critique utterance IDs that this statement incorporates.
* ``approval_ratings``: (For final synthesis utterances) A dictionary mapping participant IDs to a boolean approval rating of the final consensus statement.

Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversation corresponds to a single deliberation group's round.

* ``topic``: The topic of deliberation.
* ``split``: The dataset split.
* ``participant_stances``: A dictionary mapping participant IDs to their pre- and post-deliberation stance scores on the topic.
* ``participant_rankings``: A dictionary mapping participant IDs to the list of candidate IDs they saw and their assigned rankings.
* ``ai_meta``: Additional AI-specific metadata for the conversation.

Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("habermas-machine-corpus"))

Statistics about the dataset:

* Number of speakers: 7,191
* Number of conversations: 4,492
* Number of utterances: 46,083

Additional note
---------------

Contact
^^^^^^^

Please refer to the ConvoKit repository or the original dataset release for more information.
