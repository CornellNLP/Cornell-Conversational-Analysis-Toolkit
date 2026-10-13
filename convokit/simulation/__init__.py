"""
ConvoKit AI Simulation: run AI agent conversations on ConvoArena and get them back as a ConvoKit
Corpus.

Install the extra dependencies with ``pip install "convokit[simulation]"``, then::

    from convokit.simulation import simulate

    corpus = simulate(gemini_api_key="...", sim_yaml="simulation.yaml")

See ``convokit/simulation/simulation_demo.ipynb`` and the ConvoKit AI Simulation documentation.
"""

try:
    from .create_simulation import FirebaseBackend, create_simulation, load_simulation_config
    from .dl_container import (
        DEFAULT_IMAGE,
        FIRESTORE_PORT,
        BackendContainer,
        BackendContainerError,
    )
    from .export_to_corpus import export_to_corpus
    from .local_backend import LocalBackend, LocalBackendError
    from .simulation import simulate
except ImportError as e:
    raise ImportError(
        f"convokit.simulation could not be imported ({e}). Install its dependencies with "
        '`pip install "convokit[simulation]"`.'
    ) from e

__all__ = [
    "simulate",
    "create_simulation",
    "load_simulation_config",
    "export_to_corpus",
    "FirebaseBackend",
    "LocalBackend",
    "LocalBackendError",
    "BackendContainer",
    "BackendContainerError",
    "DEFAULT_IMAGE",
    "FIRESTORE_PORT",
]
