ConvoKit AI Simulation
======================

ConvoKit AI Simulation runs conversations between AI agents on `ConvoArena <https://github.com/CornellNLP/ConvoArena>`_, a platform for
online conversation experiments with humans and AI agents built on, and returns them as a ConvoKit
corpus with :doc:`AI fields <convokitai>`. The conversations can include an AI mediator that speaks in the chat and private assistants
that privately help individual participants.

A simulation is described by a *simulation YAML* that lists the agents, mediators and assistants,
the topics, and who talks to whom. This *YAML* is generated from the `TrAuSt <https://traust.infosci.cornell.edu/>`_ team's `Simulation Factory <https://github.com/CornellNLP/simulation-factory>`_. To make your own *simulation YAML* check out this `page <https://traust.infosci.cornell.edu/simulation-factory/login>`_. Each pairing of agents and each topic runs as its own ConvoArena conversation. The package adds the agents to the chat, waits for the conversations to finish, and
converts the experiment exports into a corpus.

Simulations run either:

* **locally** (the default): a ConvoArena backend runs on the Firebase emulators in a container
  on your machine, so no Firebase project is needed; or
* **on your own ConvoArena deployment**, using an API key from its web interface. To set up your own ConvoArena deployement check out `<https://cornellnlp.github.io/ConvoArena/developers/firebase-setup>`

Example usage: `simulation demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/simulation/simulation_demo.ipynb>`_.

Installation
------------

ConvoKit AI Simulation is part of ConvoKit, in the ``convokit.simulation`` package. Install ConvoKit
with its simulation dependencies::

    pip install "convokit[simulation]"

or, from the repository root::

    python -m pip install -e ".[simulation]"

Then import it from ``convokit.simulation``::

    from convokit.simulation import simulate

The package also includes ``convokit.simulation.deliberate_lab``, the Python client for the
ConvoArena REST API (from Deliberate Lab).

Running locally also requires:

* A container runtime to run the backend: `Docker <https://docs.docker.com/get-docker/>`_ or
  `Podman <https://podman.io/docs/installation>`_. On Linux without either (e.g. Google Colab), the
  backend runs with `udocker <https://github.com/indigo-dc/udocker>`_ instead, which is downloaded
  automatically. See `Setting up your computer`_ for the steps on each operating system.
* A `Gemini API key <https://ai.google.dev/gemini-api/docs/api-key>`_ for the agents' model calls.

Setting up your computer
------------------------

This section is only needed to run simulations locally. Running on your own ConvoArena deployment
needs no container runtime.

``simulate`` picks a runtime automatically, using the first one that works:

1. **Docker**, if ``docker info`` succeeds.
2. **Podman**, if ``podman info`` succeeds.
3. **udocker**, on Linux only.

If none of them works, ``simulate`` raises a ``BackendContainerError`` explaining what to install.
All three run the same backend image, which is built for both Intel/AMD (amd64) and ARM (arm64)
processors, so Apple Silicon Macs and ARM Windows laptops work too.

The image takes about 1.5 GB of disk space. The first run downloads it, and later runs reuse it.

Windows
^^^^^^^

Docker and Podman both run containers in a small Linux virtual machine on top of
`WSL 2 <https://learn.microsoft.com/en-us/windows/wsl/install>`_. Their installers set WSL 2 up if it
is missing. This can require administrator rights, a restart, and virtualization enabled in the
BIOS/UEFI settings (it usually is by default). udocker does not work on Windows. Use one of these
options:

**Option A: Docker Desktop**

1. Install `Docker Desktop for Windows <https://docs.docker.com/desktop/setup/install/windows-install/>`_,
   keeping the default **Use WSL 2** setting.
2. Start Docker Desktop and wait until it reports that the engine is running.
3. Check it in a new terminal (PowerShell or Anaconda Prompt)::

       docker info

   This should print information about the server, not an error such as
   ``error during connect``.

Docker Desktop must be running whenever you run a simulation. Docker Desktop requires a paid
subscription for use in larger companies (see its license terms). Podman is a free alternative.

**Option B: Podman**

1. Install Podman for Windows from the `Podman releases page <https://github.com/containers/podman/releases>`_
   (the ``podman-*-setup.exe`` installer), or install
   `Podman Desktop <https://podman-desktop.io/docs/installation/windows-install>`_, which includes it.
2. In a new terminal, create and start the Podman machine (the Linux VM). ``init`` is needed only once::

       podman machine init
       podman machine start

3. Check it::

       podman info

After restarting your computer, run ``podman machine start`` again before running a simulation.

**Option C: WSL 2 only**

If you already use a Linux distribution under WSL 2 (e.g. Ubuntu), you can install Python and this
package inside it and follow the `Linux`_ steps there instead.

macOS
^^^^^

Use either:

* **Docker Desktop:** install `Docker Desktop for Mac <https://docs.docker.com/desktop/setup/install/mac-install/>`_
  (choose the Apple Silicon or Intel build to match your Mac), start it, and check that ``docker info``
  succeeds. Other tools that provide a working ``docker`` command, such as
  `OrbStack <https://orbstack.dev/>`_ or `Colima <https://github.com/abiosoft/colima>`_, work too.
* **Podman:** install it with `Homebrew <https://brew.sh/>`_ (or the installer from
  `podman.io <https://podman.io/docs/installation>`_), then create and start its machine::

      brew install podman
      podman machine init
      podman machine start
      podman info

  After restarting your Mac, run ``podman machine start`` again.

udocker does not work on macOS.

Linux
^^^^^

Use any of:

* **Docker Engine:** install it for your distribution
  (`instructions <https://docs.docker.com/engine/install/>`_). ``simulate`` calls ``docker`` without
  ``sudo``, so add yourself to the ``docker`` group, then log out and back in::

      sudo usermod -aG docker $USER
      docker info

* **Podman:** install it from your distribution's packages, e.g.
  ``sudo apt install podman`` or ``sudo dnf install podman``. Podman runs without a daemon or a
  machine on Linux, so ``podman info`` should work right away.
* **Neither:** nothing to install. udocker is downloaded the first time you run a simulation. Its
  first start also unpacks the image, which takes several minutes, and the backend is slower under
  udocker than under Docker or Podman.

Google Colab
^^^^^^^^^^^^

Nothing to install. Colab has no Docker, so the backend runs under udocker automatically. Expect
the first start in a new Colab session to take several minutes.

Troubleshooting
^^^^^^^^^^^^^^^

* **"No container runtime is available"**: neither ``docker info`` nor ``podman info`` succeeded.
  Start Docker Desktop or run ``podman machine start``, then try again.
* **"Port(s) [...] already in use"**: a backend from an earlier run is still running, or another
  program uses port 5001, 8085 or 9099. Stop the old backend with ``docker stop dl-backend`` (or
  ``podman stop dl-backend``).
* **The emulators time out while starting**: check the container's output with
  ``docker logs dl-backend`` (or ``podman logs dl-backend``). Under udocker, the error message
  includes the path of the log file.

Quick start
-----------

Run a simulation locally:

.. code-block:: python

    from convokit.simulation import simulate

    corpus = simulate(gemini_api_key="...", sim_yaml="simulation.yaml")
    corpus.print_summary_stats()

    for convo in corpus.iter_conversations():
        print(convo.meta["pairing_id"], convo.meta["completed"])
        print(convo.get_transcript(supports=True))

``simulate`` starts the backend container, runs every conversation, and stops the backend when it is
done. The first run downloads the backend image and can take a few minutes to start.

To run on your own ConvoArena deployment instead, create a ConvoArena API key in its web
interface (**Settings → API Keys**) and save your Gemini API key in the same **Settings** page. Then
pass the Firebase project ID instead of the Gemini key:

.. code-block:: python

    corpus = simulate(sim_yaml="simulation.yaml", project_id="my-convoarena", dl_api_key="dlb_live_...")

For a deployment whose Cloud Functions run outside ``us-central1``, build the backend yourself:

.. code-block:: python

    from convokit.simulation import FirebaseBackend, create_simulation

    backend = FirebaseBackend(project_id="my-convoarena", api_key="dlb_live_...", region="europe-west1")
    corpus = create_simulation(backend, "simulation.yaml")

The simulation YAML
-------------------

``sim_yaml`` can be a path to a YAML file, the YAML text itself, or an already-parsed ``dict``.
For example, a mediator and two agents discussing one topic:

.. code-block:: yaml

    description: Two people discuss a question with the help of a mediator.
    blocks:
      - name: Debate topic
        descriptions:
          - Should cities ban cars from their downtown areas?
    max_utterance: 10
    max_time: 3
    pairings:
      - id: mediated-debate
        members:
          - participant: mediator:mediator
          - participant: skeptic
            assistant: tone-checker
          - participant: enthusiast
    definitions:
      agents:
        skeptic:
          content:
            persona: {id: skeptic, name: Skeptic, avatar: "🦉", character: cautious retiree}
            model: {apiType: GEMINI, modelName: gemini-3-flash-preview}
            chatSettings:
              promptMap:
                message:
                  order: 1
                  prompt:
                    - {type: TEXT, text: "You are a "}
                    - {type: CHARACTER_CONTEXT}
                    - {type: TEXT, text: " discussing: {topic_statement}"}
                    - {type: CONTEXT, context: current}
        enthusiast: ...
      mediators:
        mediator: ...
      assistants:
        tone-checker: ...

See the `simulation demo <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/simulation/simulation_demo.ipynb>`_
for a complete example with a mediator and a private assistant.

Top-level fields
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``description``
     - Text shown with the chat.
   * - ``blocks``
     - Text blocks shown with the chat, each with a ``name`` and a list of ``descriptions``. The first
       block fills the ``{topic_name}`` and ``{topic_statement}`` placeholders in prompts.
   * - ``max_utterance``
     - The chat ends after this many messages.
   * - ``max_time``
     - The chat ends after this many minutes.
   * - ``pairings``
     - The conversations to run. Each pairing has an ``id`` and a list of ``members``. A member's
       ``participant`` is ``mediator:<key>`` for a mediator or ``<key>`` for an agent, with an
       optional ``assistant`` (a key in ``definitions.assistants``) and ``role``.
   * - ``definitions``
     - The ``agents``, ``mediators`` and ``assistants`` that pairings refer to, keyed by name. Each
       entry's ``content`` has a ``persona`` (``id``, ``name``, ``avatar``, and for agents an optional
       ``character``), a ``model`` (``apiType``, ``modelName``), ``generation`` settings and a prompt.

One conversation runs for each pairing and each combination of block descriptions. For example,
two pairings and a block with three descriptions run six conversations.

Prompt items
^^^^^^^^^^^^

Prompts are lists of items, which are translated into ConvoArena prompt items:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item ``type``
     - Inserts
   * - ``TEXT``
     - ``text``, with ``{topic_name}``, ``{topic_statement}``, ``{post_title}`` and
       ``{post_description}`` filled in.
   * - ``CONTEXT``
     - The chat so far (``context: current``), the stages before it (``before``), or both (``all``).
   * - ``CHARACTER_CONTEXT``
     - The agent's ``persona.character``.
   * - ``PARTICIPANT_ROLE``
     - The member's ``role`` from the pairing.
   * - ``PROFILE_INFO``, ``PARTICIPANT_INFO``, ``PROFILE_CONTEXT``
     - Information about the agent itself, or (for an assistant) the participant it helps.
   * - ``PARTICIPANT_CHAT_INPUT``
     - The message the participant is drafting (for assistants).
   * - ``INITIALIZATION_CONTEXT`` (or ``PRELOADED_CONTEXT``)
     - The output of the mediator's ``initialization_context_prompt``, run once before the chat.
   * - ``POST_TITLE``, ``POST_DESCRIPTION``, ``ARTICLE_PAGE``
     - The post title and description (``post_title`` / ``post_description``, defaulting to the
       first topic and the ``description``).
   * - ``RULE``
     - A rule of the r/ChangeMyView subreddit, given by ``rule`` (``A``–``E`` or ``1``–``5``).

An agent prompt without any ``TEXT`` instructions gets a default instruction to reply in character
with a short message, so that the agent always has something to do.

The resulting corpus
--------------------

The corpus has one conversation per pairing and topic combination:

* Agents and mediators are AI speakers. Mediators have ``ai_meta["role"] == "public assistant"`` and
  agents have ``"participant"``. ``ai_meta["config"]`` holds the model and prompt, and the full
  ConvoArena agent template under ``"deliberate_lab"``.
* ``convo.alias`` maps speaker IDs to the persona names used in the chat.
* Utterances keep the model's stated reason for the message in ``ai_meta["explanation"]`` when the
  agent's structured output includes one.
* Private assistants and their supports are stored as
  :class:`~convokit.model.privateAssistant.PrivateAssistant` and :class:`~convokit.model.support.Support`
  objects. Each support keeps the participant's draft at the time.
* ``convo.meta`` has the ``pairing_id``, whether the conversation ``completed`` before the timeout,
  the topic ``blocks`` shown, and the ConvoArena ``experiment_id``, ``experiment_name`` and
  ``cohort_name``.
* ``corpus.ai_meta["simulation_config"]`` holds the parsed simulation YAML.

Because mediators are marked as ``"public assistant"`` and names are in ``convo.alias``, the corpus
works directly with :doc:`ConvoKit AI Evaluation <convokitAIEvaluation>`.

API reference
-------------

Running simulations
^^^^^^^^^^^^^^^^^^^

.. automodule:: convokit.simulation.simulation
    :members:

.. automodule:: convokit.simulation.create_simulation
    :members: create_simulation, FirebaseBackend, load_simulation_config

Converting ConvoArena exports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: convokit.simulation.export_to_corpus
    :members:

Local backend
^^^^^^^^^^^^^

.. automodule:: convokit.simulation.dl_container
    :members: BackendContainer, BackendContainerError

.. automodule:: convokit.simulation.local_backend
    :members: LocalBackend, LocalBackendError
