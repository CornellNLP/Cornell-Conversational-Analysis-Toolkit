"""Run a ConvoArena agent simulation and return it as a ConvoKit Corpus."""

from pathlib import Path

from convokit import Corpus
from dl_container import BackendContainer, FIRESTORE_PORT
from local_backend import LocalBackend
from create_simulation import (
    FirebaseBackend,
    _seed_gemini_api_key,
    create_simulation,
    load_simulation_config,
)


def simulate(
    gemini_api_key: str | None = None,
    sim_yaml: str | Path | dict | None = None,
    *,
    project_id: str | None = None,
    dl_api_key: str | None = None,
    backend: FirebaseBackend | None = None,
    wait_timeout: float | None = None,
    max_concurrent: int = 4,
    start_interval: float = 5.0,
) -> Corpus:
    """Run a simulation and return its conversations as a ConvoKit Corpus.

    By default the simulation runs on a local ConvoArena backend: the
    published backend container is started (see
    :class:`dl_container.BackendContainer`), the Gemini key is stored in it, and
    the container is stopped when the simulation finishes. To run on your own
    ConvoArena deployment instead, pass its Firebase ``project_id`` and a
    ``dl_api_key``, or a prebuilt ``backend``.

    Example::

        from simulation import simulate

        corpus = simulate(gemini_api_key="...", sim_yaml="simulation.yaml")
        corpus.print_summary_stats()

    :param gemini_api_key: Gemini API key the agents use. Required for the local
        backend; must not be passed with a deployed backend, which uses the
        Gemini key saved in its web UI (Settings) by the account that owns the
        ConvoArena API key.
    :param sim_yaml: the simulation YAML, as a file path, the YAML text, or the
        parsed dict (see :mod:`create_simulation` for the format).
    :param project_id: Firebase project ID of a deployed ConvoArena to run
        on instead of the local backend. Requires ``dl_api_key``; cannot be
        combined with ``backend``.
    :param dl_api_key: ConvoArena API key, created in the deployment's web
        UI under Settings -> API Keys. Used only with ``project_id``.
    :param backend: a :class:`create_simulation.FirebaseBackend` to run on, as
        an alternative to ``project_id`` and ``dl_api_key``.
    :param wait_timeout: seconds to wait for the conversations to finish before
        keeping them as they stand. Defaults to the YAML's ``max_time`` plus 5
        minutes.
    :param max_concurrent: how many conversations to set up at the same time
        (see :func:`create_simulation.create_simulation`).
    :param start_interval: minimum seconds between the starts of two
        conversations' setup; raise it if Gemini rate limits keep
        conversations from starting.
    :return: a Corpus with one Conversation per pairing and block combination
        (see :func:`create_simulation.create_simulation`).
    :raises ValueError: if ``sim_yaml`` is missing or invalid, or the arguments
        are inconsistent: both ``project_id`` and ``backend``, ``project_id``
        without ``dl_api_key``, ``gemini_api_key`` with a deployed backend, or
        no ``gemini_api_key`` for the local backend.
    """
    # check the YAML before starting anything, since the local backend takes minutes to start
    config = load_simulation_config(sim_yaml)

    if project_id is not None:
        if backend is not None:
            raise ValueError("pass either project_id or backend, not both")
        if not dl_api_key:
            raise ValueError(
                "dl_api_key is required with project_id; create one in the "
                "deployment's web UI under Settings -> API Keys"
            )
        backend = FirebaseBackend(project_id=project_id, api_key=dl_api_key)

    if backend is not None:
        if gemini_api_key:
            raise ValueError(
                "gemini_api_key is only used with the local backend; on your own "
                "deployment, save the Gemini key in the web UI's Settings instead"
            )
        return create_simulation(backend, config, wait_timeout, max_concurrent, start_interval)

    if not gemini_api_key:
        raise ValueError("gemini_api_key is required for the local backend")
    with BackendContainer():
        with LocalBackend(
            ".",
            reuse_running=True,
            firestore_port=FIRESTORE_PORT,
            # the functions can take minutes to load under udocker (e.g. on Colab)
            startup_timeout=600,
        ) as local:
            _seed_gemini_api_key(local, gemini_api_key)
            return create_simulation(local, config, wait_timeout, max_concurrent, start_interval)
