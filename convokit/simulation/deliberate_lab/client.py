# pylint: disable=line-too-long
"""
Deliberate Lab REST API Client

Simple Python client for interacting with the Deliberate Lab API.

Usage:
    Set the DL_API_KEY environment variable with your API key, then:

    import deliberate_lab as dl

    client = dl.Client()              # defaults to dev
    client = dl.Client(env="prod")     # use production
    client = dl.Client(base_url="...") # custom URL

    # List experiments
    experiments = client.list_experiments()

    # Create experiment with typed stage configs
    stage = dl.SurveyStageConfig(
        id="survey1",
        kind="survey",
        name="Demographics",
        descriptions=dl.StageTextConfig(primaryText="Please answer", infoText="", helpText=""),
        progress=dl.StageProgressConfig(minParticipants=1, waitForAllParticipants=False, showParticipantProgress=False),
        questions=[
            dl.TextSurveyQuestion(id="q1", kind="text", questionTitle="What is your name?")
        ]
    )
    client.create_experiment(name="My Study", stages=[stage])
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import os
import requests

# Re-export all types so users can do: dl.SurveyStageConfig, dl.TextSurveyQuestion, etc.
from .types import *  # pylint: disable=wildcard-import,unused-wildcard-import

if TYPE_CHECKING:
    from pydantic import BaseModel


class APIError(requests.HTTPError):
    """Exception raised for API errors with parsed error message."""

    def __init__(self, response: requests.Response, message: str):
        self.message = message
        super().__init__(f"API Error ({response.status_code}): {message}", response=response)


class Client:
    """Client for the Deliberate Lab REST API."""

    PROD_URL = "https://us-central1-deliberate-lab.cloudfunctions.net/api/v1"
    DEV_URL = "http://127.0.0.1:5001/demo-deliberate-lab/us-central1/api/v1"

    def __init__(
        self,
        env: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize the client.

        Args:
            env: Environment to use: "prod" or "dev". Defaults to "dev".
                 Ignored if base_url is provided.
            base_url: Custom base URL for the API. Overrides env if provided.
            api_key: API key for authentication. If not provided, reads from
                     DL_API_KEY environment variable.
            timeout: Request timeout in seconds. Defaults to 60, longer for exports.
        """
        if base_url is not None:
            self.base_url = base_url
        elif env == "prod":
            self.base_url = self.PROD_URL
        elif env is None or env == "dev":
            self.base_url = self.DEV_URL
        else:
            raise ValueError(f"Unknown env '{env}'. Use 'prod', 'dev', or provide a base_url.")
        self.api_key = api_key or os.environ.get("DL_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("API key required. Pass api_key parameter or set DL_API_KEY env var.")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _handle_response(self, response: requests.Response):
        """Handle API response and raise errors if needed."""
        if response.status_code >= 400:
            try:
                error_msg = response.json().get("error", response.text)
            except ValueError:
                error_msg = response.text
            raise APIError(response, error_msg)
        return response.json()

    def health_check(self) -> dict:
        """Check API health status."""
        response = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        return self._handle_response(response)

    # =========================================================================
    # Experiment Methods
    # =========================================================================

    def list_experiments(self) -> dict:
        """
        List all experiments for the authenticated user.

        Returns:
            dict with 'experiments' list and 'total' count
        """
        response = self._session.get(f"{self.base_url}/experiments", timeout=self.timeout)
        return self._handle_response(response)

    def get_experiment(self, experiment_id: str) -> dict:
        """
        Get a specific experiment by ID.

        Args:
            experiment_id: The experiment ID

        Returns:
            dict with 'experiment', 'stageMap', 'agentMediatorMap', 'agentParticipantMap'
        """
        response = self._session.get(
            f"{self.base_url}/experiments/{experiment_id}", timeout=self.timeout
        )
        return self._handle_response(response)

    def create_experiment(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        stages: Optional[list[BaseModel]] = None,
        prolific_config: Optional[ProlificConfig] = None,
        agent_mediators: Optional[list[BaseModel]] = None,
        agent_participants: Optional[list[BaseModel]] = None,
        agent_assistants: Optional[list[BaseModel]] = None,
        unlock_time_ms: Optional[int] = None,
        unlock_duration_ms: Optional[int] = None,
        template: Optional[BaseModel] = None,
    ) -> dict:
        """
        Create a new experiment.

        Supports two modes:

        1. **Simple creation**: Provide name (required), plus optional fields
        2. **Full template creation**: Provide a complete ExperimentTemplate

        When `template` is provided, creates experiment with all stages and agents.
        This uses the same logic as the UI's experiment creation.

        Args:
            name: Experiment name (required for simple creation)
            description: Optional description (simple creation)
            stages: Optional list of stage configurations (simple creation)
            prolific_config: Optional Prolific integration config (simple creation)
            agent_mediators: Optional list of AgentMediatorTemplate (simple creation)
            agent_participants: Optional list of AgentParticipantTemplate (simple creation)
            agent_assistants: Optional list of AgentAssistantTemplate (simple creation)
            template: Optional full ExperimentTemplate for complete creation.
                      When provided, all other fields are ignored.

        Returns:
            dict with 'experiment' containing created experiment config (including 'id')

        Example (simple creation):
            stage = dl.SurveyStageConfig(...)
            client.create_experiment(name="My Study", stages=[stage])

        Example (simple creation with agents):
            client.create_experiment(
                name="My Study",
                stages=[stage],
                agent_mediators=[mediator]
            )

        Example (full template creation):
            template = dl.ExperimentTemplate(...)
            client.create_experiment(template=template)
        """
        data: dict = {}

        # Full template creation takes precedence
        if template is not None:
            data["template"] = template.model_dump(mode="json", by_alias=True, exclude_none=True)
        else:
            # Simple creation mode
            if name is None:
                raise ValueError("name is required for simple creation (or provide template)")
            data["name"] = name
            if description is not None:
                data["description"] = description
            if stages is not None:
                # Convert Pydantic models to dicts for JSON serialization
                data["stages"] = [
                    s.model_dump(mode="json", by_alias=True, exclude_none=True) for s in stages
                ]
            if prolific_config is not None:
                data["prolificConfig"] = prolific_config.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )
            if agent_mediators is not None:
                data["agentMediators"] = [
                    a.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for a in agent_mediators
                ]
            if agent_participants is not None:
                data["agentParticipants"] = [
                    a.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for a in agent_participants
                ]
            if agent_assistants is not None:
                data["agentAssistants"] = [
                    a.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for a in agent_assistants
                ]
            if unlock_time_ms is not None:
                data["unlockTimeMs"] = unlock_time_ms

            if unlock_duration_ms is not None:
                data["unlockDurationMs"] = unlock_duration_ms

        response = self._session.post(
            f"{self.base_url}/experiments", json=data, timeout=self.timeout
        )
        return self._handle_response(response)

    def update_experiment(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        stages: Optional[list[BaseModel]] = None,
        prolific_config: Optional[ProlificConfig] = None,
        agent_mediators: Optional[list[BaseModel]] = None,
        agent_participants: Optional[list[BaseModel]] = None,
        agent_assistants: Optional[list[BaseModel]] = None,
        template: Optional[BaseModel] = None,
    ) -> dict:
        """
        Update an existing experiment.

        Supports two modes:

        1. **Partial update**: Provide individual fields to update only those fields.
           Omit fields you don't want to change.

        2. **Full template update**: Provide a complete ExperimentTemplate in the `template` field.
           This replaces the entire experiment including all stages and agents.
           Other fields are ignored when template is provided.

        Args:
            experiment_id: The experiment ID to update
            name: Optional new name (partial update)
            description: Optional new description (partial update)
            stages: Optional new list of stage configurations (partial update, replaces all stages)
            prolific_config: Optional new Prolific integration config (partial update)
            agent_mediators: Optional list of AgentMediatorTemplate (partial update, replaces all mediators)
            agent_participants: Optional list of AgentParticipantTemplate (partial update, replaces all participants)
            agent_assistants: Optional list of AgentAssistantTemplate (partial update, replaces all assistants)
            template: Optional full ExperimentTemplate for complete replacement.
                      When provided, replaces entire experiment. Other fields are ignored.

        Returns:
            dict with 'updated' bool and 'id'

        Example (partial update - just name):
            client.update_experiment("exp123", name="New Name")

        Example (partial update - just agents):
            mediator = dl.AgentMediatorTemplate(...)
            client.update_experiment("exp123", agent_mediators=[mediator])

        Example (full template update):
            template = dl.ExperimentTemplate(...)
            client.update_experiment("exp123", template=template)
        """
        data: dict = {}

        # Full template update takes precedence
        if template is not None:
            data["template"] = template.model_dump(mode="json", by_alias=True, exclude_none=True)
        else:
            # Partial update mode
            if name is not None:
                data["name"] = name
            if description is not None:
                data["description"] = description
            if stages is not None:
                data["stages"] = [
                    s.model_dump(mode="json", by_alias=True, exclude_none=True) for s in stages
                ]
            if prolific_config is not None:
                data["prolificConfig"] = prolific_config.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )
            if agent_mediators is not None:
                data["agentMediators"] = [
                    a.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for a in agent_mediators
                ]
            if agent_participants is not None:
                data["agentParticipants"] = [
                    a.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for a in agent_participants
                ]
            if agent_assistants is not None:
                data["agentAssistants"] = [
                    a.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for a in agent_assistants
                ]

        response = self._session.put(
            f"{self.base_url}/experiments/{experiment_id}",
            json=data,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def delete_experiment(self, experiment_id: str) -> dict:
        """
        Delete an experiment.

        Args:
            experiment_id: The experiment ID to delete

        Returns:
            dict with 'deleted' bool and 'id'
        """
        response = self._session.delete(
            f"{self.base_url}/experiments/{experiment_id}", timeout=self.timeout
        )
        return self._handle_response(response)

    def export_experiment(self, experiment_id: str) -> dict:
        """
        Export full experiment data including participants.

        Args:
            experiment_id: The experiment ID to export

        Returns:
            Full ExperimentDownload structure with experiment, stages,
            cohorts, participants, agents, and chat data
        """
        response = self._session.get(
            f"{self.base_url}/experiments/{experiment_id}/export",
            timeout=(self.timeout * 3),  # Allow more time for exports
        )
        return self._handle_response(response)

    def export_experiment_logs(self, experiment_id: str) -> list:
        """
        Export all model logs from an experiment.

        Args:
            experiment_id: The experiment ID to export logs for

        Returns:
            List of model log entries
        """
        response = self._session.get(
            f"{self.base_url}/experiments/{experiment_id}/export/logs",
            timeout=(self.timeout * 3),  # Allow more time for exports
        )
        return self._handle_response(response)

    def fork_experiment(self, experiment_id: str, name: Optional[str] = None) -> dict:
        """
        Fork an experiment, creating a copy with all stages and agents.

        Args:
            experiment_id: The experiment ID to fork
            name: Optional custom name for the forked experiment.
                  Defaults to "Copy of [original name]"

        Returns:
            dict with 'experiment' (the new experiment) and 'sourceExperimentId'

        Example:
            # Fork with default name
            result = client.fork_experiment("exp123")
            print(f"Forked to: {result['experiment']['id']}")

            # Fork with custom name
            result = client.fork_experiment("exp123", name="My New Study")
        """
        data: dict = {}
        if name is not None:
            data["name"] = name

        response = self._session.post(
            f"{self.base_url}/experiments/{experiment_id}/fork",
            json=data if data else None,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    # =========================================================================
    # Cohort Methods
    # =========================================================================

    def list_cohorts(self, experiment_id: str) -> dict:
        """
        List all cohorts for an experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            dict with 'cohorts' list and 'total' count
        """
        response = self._session.get(
            f"{self.base_url}/experiments/{experiment_id}/cohorts",
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def get_cohort(self, experiment_id: str, cohort_id: str) -> dict:
        """
        Get a specific cohort by ID.

        Args:
            experiment_id: The experiment ID
            cohort_id: The cohort ID

        Returns:
            dict with 'cohort' and 'participantCount'
        """
        response = self._session.get(
            f"{self.base_url}/experiments/{experiment_id}/cohorts/{cohort_id}",
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def create_cohort(
        self,
        experiment_id: str,
        name: str,
        description: Optional[str] = None,
        participant_config: Optional[BaseModel] = None,
    ) -> dict:
        """
        Create a new cohort in an experiment.

        Args:
            experiment_id: The experiment ID
            name: Cohort name (required)
            description: Optional description
            participant_config: Optional CohortParticipantConfig with min/max participants

        Returns:
            dict with 'cohort' containing created cohort config

        Example:
            client.create_cohort(
                experiment_id="exp123",
                name="Control Group",
                description="Participants in control condition",
            )
        """
        data: dict = {"name": name}
        if description is not None:
            data["description"] = description
        if participant_config is not None:
            data["participantConfig"] = participant_config.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )

        response = self._session.post(
            f"{self.base_url}/experiments/{experiment_id}/cohorts",
            json=data,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def create_simulation(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        stages: Optional[list[BaseModel]] = None,
        prolific_config: Optional[ProlificConfig] = None,
        agent_mediators: Optional[list[BaseModel]] = None,
        agent_participants: Optional[list[BaseModel]] = None,
        agent_assistants: Optional[list[BaseModel]] = None,
        unlock_time_ms: Optional[int] = None,
        unlock_duration_ms: Optional[int] = None,
        template: Optional[BaseModel | dict] = None,
        num_cohorts: int = 0,
        cohort_names: list[str] = None,
        cohort_description: Optional[list[str]] = None,
        cohort_participant_config: Optional[list[BaseModel]] = None,
    ) -> dict:

        data: dict = {}

        # Full template creation takes precedence
        if template is not None:
            # a plain dict is sent as-is, for templates the pydantic models can't express
            data["template"] = (
                template
                if isinstance(template, dict)
                else template.model_dump(mode="json", by_alias=True, exclude_none=True)
            )
        else:
            if name is None:
                raise ValueError("name is required for simple creation (or provide template)")

            data["name"] = name

            if description is not None:
                data["description"] = description

            if stages is not None:
                data["stages"] = [
                    s.model_dump(mode="json", by_alias=True, exclude_none=True) for s in stages
                ]

            if prolific_config is not None:
                data["prolificConfig"] = prolific_config.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )

            if agent_mediators is not None:
                data["agentMediators"] = [
                    a.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for a in agent_mediators
                ]

            if agent_participants is not None:
                data["agentParticipants"] = [
                    a.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for a in agent_participants
                ]

            if agent_assistants is not None:
                data["agentAssistants"] = [
                    a.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for a in agent_assistants
                ]

            if unlock_time_ms is not None:
                data["unlockTimeMs"] = unlock_time_ms

            if unlock_duration_ms is not None:
                data["unlockDurationMs"] = unlock_duration_ms

        # 1) Create experiment
        response = self._session.post(
            f"{self.base_url}/experiments",
            json=data,
            timeout=self.timeout,
        )

        experiment = self._handle_response(response)
        experiment_id = experiment["experiment"]["id"]

        # 2) Build batch cohorts payload (NEW)
        cohort_payloads = []

        for i in range(num_cohorts):
            c_name = cohort_names[i] if cohort_names else f"Cohort {i+1}"
            desc = cohort_description[i] if cohort_description else None
            config = cohort_participant_config[i] if cohort_participant_config else None

            cohort_payload = {
                "name": c_name,
                "description": desc,
            }

            if config is not None:
                cohort_payload["participantConfig"] = config.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )

            cohort_payloads.append(cohort_payload)

        # 3) Single POST for ALL cohorts (BATCH CALL)
        cohort_results = []

        if cohort_payloads:
            cohort_response = self._session.post(
                f"{self.base_url}/experiments/{experiment_id}/cohorts/batch",
                json={"cohorts": cohort_payloads},
                timeout=None,
            )

            cohort_results = self._handle_response(cohort_response).get("cohorts", [])

        return {
            "experiment": experiment,
            "cohorts": cohort_results,
        }

    def update_cohort(
        self,
        experiment_id: str,
        cohort_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        participant_config: Optional[BaseModel] = None,
    ) -> dict:
        """
        Update an existing cohort.

        Args:
            experiment_id: The experiment ID
            cohort_id: The cohort ID to update
            name: Optional new name
            description: Optional new description
            participant_config: Optional new CohortParticipantConfig

        Returns:
            dict with 'updated' bool and 'id'
        """
        data: dict = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if participant_config is not None:
            data["participantConfig"] = participant_config.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )

        response = self._session.put(
            f"{self.base_url}/experiments/{experiment_id}/cohorts/{cohort_id}",
            json=data,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def delete_cohort(self, experiment_id: str, cohort_id: str) -> dict:
        """
        Delete a cohort.

        Note: This marks all participants in the cohort as DELETED and
        recursively deletes all cohort data.

        Args:
            experiment_id: The experiment ID
            cohort_id: The cohort ID to delete

        Returns:
            dict with 'deleted' bool and 'id'
        """
        response = self._session.delete(
            f"{self.base_url}/experiments/{experiment_id}/cohorts/{cohort_id}",
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def get_completed_experiment_data(self, experiment_id: str) -> dict:
        export = self.export_experiment(experiment_id)

        TERMINAL_STATUSES = {
            "SUCCESS",
            "TRANSFER_TIMEOUT",
            "TRANSFER_FAILED",
            "TRANSFER_DECLINED",
            "ATTENTION_TIMEOUT",
            "BOOTED_OUT",
            "DELETED",
        }

        for participant in export["participantMap"].values():
            status = participant.get("profile", {}).get("currentStatus")

            if status not in TERMINAL_STATUSES:
                raise RuntimeError(f"Experiment is still running (participant status: {status}).")

        return export


if __name__ == "__main__":
    # Quick test - requires DL_API_KEY to be set
    try:
        client = Client()

        print("Checking API health...")
        health = client.health_check()
        print(f"API Status: {health['status']}")

        print("\nListing experiments...")
        result = client.list_experiments()
        print(f"Found {result['total']} experiments")

        for exp in result["experiments"][:5]:  # Show first 5
            print(f"  - {exp['metadata']['name']} ({exp['id']})")

    except APIError as e:
        print(f"API Error: {e}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
