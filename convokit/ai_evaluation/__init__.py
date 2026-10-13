"""
ConvoKit AI evaluation: Transformers that evaluate AI-mediated conversations stored in the
ConvoKit AI data format (see convokit/model/AI_FORMAT.md). Import them from
``convokit.ai_evaluation``, e.g. ``from convokit.ai_evaluation import CIG``. CIG and ClaimTracker
need ConvoKit's GenAI dependencies (``pip install "convokit[genai]"``).

- ``CIG``: per-utterance conversational information gain, rated by an LLM;
- ``ClaimTracker``: claim-level consensus between participants, and where their claims came from;
- ``MediatorRedirection``: how much each mediator turn redirected the conversation, computed
  with ``convokit.redirection``.

``MediatorRedirection`` and its context selectors are ``None`` if ``convokit.redirection`` (and
its dependencies) cannot be imported.
"""

from .cig import CIG
from .claims import ClaimTracker
from .llm import GPT5Client, JSONClient, get_eval_llm_client
from .util import (
    default_is_mediator,
    default_speaker_name,
    default_topic,
    pre_survey_text,
)

MediatorRedirection = None
mediator_future_context_selector = mediator_previous_context_selector = None
try:
    from .redirection import (
        MediatorRedirection,
        mediator_future_context_selector,
        mediator_previous_context_selector,
    )
except ImportError:
    pass

__all__ = [
    "CIG",
    "ClaimTracker",
    "MediatorRedirection",
    "GPT5Client",
    "JSONClient",
    "get_eval_llm_client",
    "default_is_mediator",
    "default_speaker_name",
    "default_topic",
    "pre_survey_text",
    "mediator_future_context_selector",
    "mediator_previous_context_selector",
]
