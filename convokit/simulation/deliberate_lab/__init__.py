"""
Deliberate Lab Python Client

Simple Python client for interacting with the Deliberate Lab API.

Usage:
    import deliberate_lab as dl

    client = dl.Client()  # Uses DL_API_KEY environment variable
    experiments = client.list_experiments()
    data = client.export_experiment("experiment-id")
"""

from .client import Client, APIError
from .types import *  # noqa: F401, F403

__all__ = ["Client", "APIError"]
