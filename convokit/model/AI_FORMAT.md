# ConvoKit AI Data Format

The ConvoKit data model ([data format](https://convokit.cornell.edu/documentation/data_format.html)) can
describe conversations that involve AI agents, with these fields:

- **AI speakers**: speakers marked with `is_ai`, together with the configuration that generates their messages.
- **Private assistants** (`PrivateAssistant`): AI assistants that help one or more speakers privately, outside the
  conversation itself.
- **Supports** (`Support`): the messages a private assistant sends to the speaker(s) it helps.

On disk, the AI fields are stored inside the regular ConvoKit metadata (`meta["is_ai"]`, `meta["ai_meta"]`, ...).
When `convokit.Corpus` loads a corpus, those fields are moved out of `meta` and exposed as attributes of the
Speakers, Utterances, Conversations and the Corpus. Corpora without AI agents don't have any of these fields, and
their objects get the defaults (`is_ai == False`, empty `ai_meta`).

With the `db` storage backend, the AI fields are also stored in the database (as fields of each component's entry,
and, for the corpus, in an `ai_corpus_<corpus id>` entry of the meta collection), so `Corpus.reconnect_to_db`
restores them.

The full documentation, including the API reference, is at
<https://convokit.cornell.edu/documentation/convokitai.html>.

## Speaker

| Field     | Type   | Description                                                              |
| --------- | ------ | ------------------------------------------------------------------------ |
| `id`      | `str`  | Unique ID of the speaker.                                                |
| `is_ai`   | `bool` | Whether the speaker is known to be an AI agent.                          |
| `ai_meta` | `dict` | ConvoKit AI metadata (see below). Empty for speakers that aren't AI.     |

`Speaker.ai_meta`:

| Key      | Type   | Description                                                                                       |
| -------- | ------ | ------------------------------------------------------------------------------------------------- |
| `config` | `dict` | The [generation config](#generation-config) that produces this speaker's messages (used by `Speaker.generate`). |
| `role`   | `str`  | The speaker's role in the conversation, e.g. `"participant"` or `"public assistant"` (a mediator). |

## Utterance

Utterances have the usual ConvoKit fields (`id`, `speaker`, `conversation_id`, `reply_to`, `timestamp`, `text`,
`meta`), plus:

| Field     | Type   | Description                                                                                   |
| --------- | ------ | --------------------------------------------------------------------------------------------- |
| `ai_meta` | `dict` | Additional outputs of the model call that produced this message. Empty for human utterances. |

`Utterance.ai_meta`:

| Key        | Type            | Description                                                         |
| ---------- | --------------- | ------------------------------------------------------------------- |
| `config`   | `dict`          | The generation config used to produce this message.                 |
| `supports` | `list[Support]` | The supports sent in reply to this utterance (`Utterance.supports`). |

Other model outputs, such as a mediator's stated reason for a message (`explanation`), can be stored here as well.

## Conversation

| Field     | Type   | Description                                  |
| --------- | ------ | -------------------------------------------- |
| `id`      | `str`  | Unique ID of the conversation.               |
| `ai_meta` | `dict` | ConvoKit AI metadata (see below).            |

`Conversation.ai_meta`:

| Key                  | Type                     | Description                                                                                       |
| -------------------- | ------------------------ | ------------------------------------------------------------------------------------------------- |
| `alias`              | `dict[str, str]`         | The name each speaker goes by in the conversation (e.g. `"Bear"`), keyed by speaker ID (`Conversation.alias`). |
| `private_assistants` | `list[PrivateAssistant]` | The private assistants in this conversation (`Conversation.private_assistants`).                  |
| `supports`           | `list[Support]`          | Optional. Supports stored at the conversation level. `Conversation.supports` returns these together with the supports of the private assistants and of the utterances. |

## Corpus

| Field     | Type   | Description                                                                                          |
| --------- | ------ | ---------------------------------------------------------------------------------------------------- |
| `has_ai`  | `bool` | Whether any speaker in the corpus is AI. Computed from the speakers if not given.                     |
| `ai_meta` | `dict` | Free-form corpus-level ConvoKit AI metadata. For example, `convokit.simulation` stores the simulation configuration under `simulation_config`. |

## PrivateAssistant

| Field             | Type            | Description                                                                          |
| ----------------- | --------------- | ------------------------------------------------------------------------------------ |
| `id`              | `str`           | Unique ID of the private assistant.                                                  |
| `config`          | `dict`          | The [generation config](#generation-config) that produces its supports (used by `PrivateAssistant.generate`). |
| `supports`        | `list[Support]` | The supports this assistant sent.                                                    |
| `speakers`        | `list[str]`     | IDs of the speakers who can see this assistant's supports.                           |
| `conversation_id` | `str`           | ID of the conversation the assistant belongs to.                                     |

## Support

| Field                  | Type         | Description                                                                         |
| ---------------------- | ------------ | ----------------------------------------------------------------------------------- |
| `id`                   | `str`        | Unique ID of the support.                                                           |
| `text`                 | `str`        | The message the assistant sent.                                                     |
| `reply_to`             | `str`        | ID of the utterance the assisted speaker was replying to.                           |
| `draft`                | `str`        | What the assisted speaker had drafted so far (`""` if nothing).                     |
| `private_assistant_id` | `str`        | ID of the private assistant that sent the support.                                  |
| `timestamp`            | `int`/`str`  | When the support was sent.                                                          |

## Generation config

Speakers and private assistants that can generate new messages carry a generation config:

| Key           | Type    | Description                                                                                  |
| ------------- | ------- | -------------------------------------------------------------------------------------------- |
| `prompt`      | `str`   | The prompt that produces a message (required).                                               |
| `model`       | `str`   | The model name, e.g. `"gemini-2.5-flash"` or `"gpt-4o-mini"`.                                |
| `provider`    | `str`   | Optional. The `convokit.genai` provider: `"gemini"`, `"gpt"` or `"local"`. Inferred from `model` if missing. |
| `temperature` | `float` | Optional. Sampling temperature.                                                              |

For example:

```python
{
    "prompt": "You help Goose phrase their replies politely.",
    "model": "gemini-2.5-flash",
    "temperature": 0.7,
}
```
