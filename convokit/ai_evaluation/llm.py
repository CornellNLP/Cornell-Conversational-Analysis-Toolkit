"""
LLM access for the evaluation Transformers.

The Transformers call ``generate_json(prompt)`` on their client. :func:`get_eval_llm_client`
builds one from a model name:

- OpenAI models (``gpt-*``, ``o1``/``o3``/``o4``): ``convokit.genai``'s ``GPTClient`` always
  sends ``temperature`` and ``max_tokens``, which OpenAI's reasoning models (gpt-5,
  gpt-5-mini) reject, so :class:`GPT5Client` is used instead. It keeps the ``convokit.genai``
  ``LLMClient`` interface and adds ``reasoning_effort``, an optional fixed ``seed`` and JSON
  mode.
- Gemini models (``gemini-*``): ``convokit.genai``'s ``GeminiClient``, wrapped in
  :class:`JSONClient`.
"""

import json
import re
import time
from typing import Optional

from convokit.genai import GenAIConfigManager, LLMClient, LLMResponse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

RETRY_AFTER = 10  # seconds


def parse_json(text: str):
    """
    Parse the JSON object or array in a model reply, tolerating Markdown code fences and control
    characters.

    :param text: the model reply
    :return: the parsed JSON (dict or list)
    :raises json.JSONDecodeError: if the reply is not valid JSON
    """
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text, strict=False)


def _retry(attempt, max_retries: int):
    """Call ``attempt()``, retrying failed calls after a 10-second wait."""
    for n in range(max_retries + 1):
        try:
            return attempt()
        except Exception as e:
            if n == max_retries:
                raise
            print(f"{type(e).__name__}: {e}. Retrying in {RETRY_AFTER}s...")
            time.sleep(RETRY_AFTER)


class JSONClient(LLMClient):
    """
    Wraps a ``convokit.genai`` ``LLMClient`` with the ``generate_json`` method the evaluation
    Transformers call.

    Unparseable replies are retried after a 10-second wait.

    :param client: the ``convokit.genai`` LLMClient, e.g. a ``GeminiClient``
    :param max_retries: how many times to retry an unparseable JSON reply
    :param generate_kwargs: keyword arguments passed to every ``client.generate`` call, e.g.
        ``json_mode=True`` and ``seed=0`` for a ``GeminiClient``
    """

    def __init__(self, client: LLMClient, max_retries: int = 1, **generate_kwargs):
        self.client = client
        self.max_retries = max_retries
        self.generate_kwargs = generate_kwargs

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a reply to a prompt with the wrapped client.

        :param prompt: the prompt
        :param kwargs: passed to the wrapped client's ``generate``, overriding
            ``generate_kwargs``
        :return: the wrapped client's LLMResponse
        """
        return self.client.generate(prompt, **{**self.generate_kwargs, **kwargs})

    def generate_json(self, prompt: str):
        """
        Generate a reply to a prompt and parse it as JSON.

        :param prompt: the prompt
        :return: the parsed JSON (dict or list)
        """
        return _retry(lambda: parse_json(self.generate(prompt).text), self.max_retries)


def infer_provider(model: str) -> str:
    """
    Infer the ``convokit.genai`` provider of a model from its name.

    :param model: the model name
    :return: ``"gpt"`` for OpenAI models (``gpt*``, ``o1*``, ``o3*``, ``o4*``) or ``"gemini"``
        for Gemini models (``gemini*``)
    :raises ValueError: if the provider cannot be inferred
    """
    name = model.lower()
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return "gpt"
    if name.startswith("gemini"):
        return "gemini"
    raise ValueError(
        f"Cannot infer the provider for model {model!r}; pass provider='gpt' or "
        "provider='gemini'."
    )


def get_eval_llm_client(
    model: str,
    provider: Optional[str] = None,
    config_manager: GenAIConfigManager = None,
    reasoning_effort: str = "minimal",
    seed: Optional[int] = None,
    json_mode: bool = False,
) -> LLMClient:
    """
    Build the LLM client the evaluation Transformers use by default.

    The API key is read by ``convokit.genai`` (``GenAIConfigManager``): the ``"gpt"`` or
    ``"gemini"`` key, from the ``GPT_API_KEY`` / ``GEMINI_API_KEY`` environment variable or
    ``~/.convokit/config.yml``. Gemini can also use Vertex AI (see ``convokit.genai``'s
    ``GeminiClient``).

    :param model: model name, e.g. ``"gpt-5"`` or ``"gemini-2.5-flash"``
    :param provider: ``"gpt"`` or ``"gemini"``; inferred from ``model`` if not given
    :param config_manager: GenAIConfigManager holding the API key (default: a
        GenAIConfigManager reading ``~/.convokit/config.yml``)
    :param reasoning_effort: OpenAI ``reasoning_effort``; not used for Gemini
    :param seed: seed for best-effort determinism (not sent if None)
    :param json_mode: for OpenAI, request ``response_format={"type": "json_object"}`` (which only
        allows a JSON object, not an array). Gemini clients always use JSON mode, since Gemini's
        also allows arrays.
    :return: a client with ``generate`` and ``generate_json`` methods
    :raises ValueError: if the provider is not supported or cannot be inferred
    """
    provider = (provider or infer_provider(model)).lower()
    if provider == "gpt":
        return GPT5Client(
            model,
            config_manager=config_manager,
            reasoning_effort=reasoning_effort,
            seed=seed,
            json_mode=json_mode,
        )
    if provider == "gemini":
        from convokit.genai import get_llm_client

        client = get_llm_client("gemini", config_manager or GenAIConfigManager(), model=model)
        return JSONClient(client, json_mode=True, seed=seed)
    raise ValueError(f"Unsupported provider {provider!r}; use 'gpt' or 'gemini'.")


class GPT5Client(LLMClient):
    """
    Client for OpenAI reasoning models (the gpt-5 family) with the ``convokit.genai``
    ``LLMClient`` interface.

    Failed calls (and, in ``generate_json``, unparseable replies) are retried after a
    10-second wait.

    :param model: model name, e.g. ``"gpt-5"`` or ``"gpt-5-mini"``
    :param config_manager: GenAIConfigManager holding the ``"gpt"`` API key (default: a
        GenAIConfigManager reading ``~/.convokit/config.yml``)
    :param reasoning_effort: OpenAI ``reasoning_effort`` (``"minimal"``, ``"low"``,
        ``"medium"`` or ``"high"``)
    :param seed: OpenAI ``seed`` for best-effort determinism (not sent if None)
    :param json_mode: request ``response_format={"type": "json_object"}``
    :param max_retries: how many times to retry a failed call or an unparseable JSON reply
    :raises ImportError: if the ``openai`` package is not installed
    :raises ValueError: if no OpenAI API key is configured
    """

    def __init__(
        self,
        model: str,
        config_manager: GenAIConfigManager = None,
        reasoning_effort: str = "minimal",
        seed: Optional[int] = None,
        json_mode: bool = False,
        max_retries: int = 1,
    ):
        if OpenAI is None:
            raise ImportError(
                "OpenAI client not available. Please install via `pip install convokit[genai]`."
            )
        if config_manager is None:
            config_manager = GenAIConfigManager()
        api_key = config_manager.get_api_key("gpt")
        if not api_key:
            raise ValueError("OpenAI API key is required. ")

        self.config_manager = config_manager
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.json_mode = json_mode
        self.max_retries = max_retries

    def _call(self, prompt: str):
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": self.reasoning_effort,
        }
        if self.seed is not None:
            request["seed"] = self.seed
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**request)

    def _retry(self, attempt):
        return _retry(attempt, self.max_retries)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a reply to a prompt.

        :param prompt: the prompt
        :param kwargs: ignored; accepted for compatibility with the ``LLMClient`` interface
        :return: an LLMResponse with the reply text, total tokens used (-1 if unknown), latency
            in seconds and the raw OpenAI response
        """
        start = time.time()
        raw = self._retry(lambda: self._call(prompt))
        tokens = raw.usage.total_tokens if raw.usage else -1
        return LLMResponse(
            text=raw.choices[0].message.content,
            tokens=tokens,
            latency=time.time() - start,
            raw=raw,
        )

    def generate_json(self, prompt: str):
        """
        Generate a reply to a prompt and parse it as JSON.

        :param prompt: the prompt
        :return: the parsed JSON (dict or list)
        """
        return self._retry(lambda: parse_json(self._call(prompt).choices[0].message.content))
