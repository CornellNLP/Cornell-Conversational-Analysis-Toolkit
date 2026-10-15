# pyright: reportInvalidTypeForm=false
# pytype: disable=invalid-function-definition
# pylint: disable=missing-module-docstring,missing-class-docstring,invalid-name,too-few-public-methods

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11 (added for ConvoKit, which supports Python 3.10)
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)


from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel


class CollectionName(StrEnum):
    experimentTemplates = "experimentTemplates"
    experiments = "experiments"


class Metadata(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    name: str
    publicName: str
    description: str
    tags: list[str]


class Visibility(StrEnum):
    public = "public"
    private = "private"


class Permissions(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    visibility: Visibility
    readers: list[str]


class ProlificConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    enableProlificIntegration: bool
    defaultRedirectCode: str
    attentionFailRedirectCode: str
    bootedRedirectCode: str


class CohortDefinition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    alias: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    description: str | None = None
    generatedCohortId: str | None = None
    maxParticipantsPerCohort: Annotated[int | None, Field(ge=1)] = None


class MinParticipantsPerCohort(RootModel[int]):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    root: Annotated[int, Field(ge=0)]


class MaxParticipantsPerCohort(RootModel[int]):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    root: Annotated[int, Field(ge=1)]


class CohortParticipantConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    minParticipantsPerCohort: MinParticipantsPerCohort | None = None
    maxParticipantsPerCohort: MaxParticipantsPerCohort | None = None
    includeAllParticipantsInCohortCount: bool
    botProtection: bool


class Scope(StrEnum):
    experiment = "experiment"
    cohort = "cohort"
    participant = "participant"


class String(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )
    type: Literal["string"] = "string"


class Number(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )
    type: Literal["number"] = "number"


class Integer(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )
    type: Literal["integer"] = "integer"


class Boolean(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )
    type: Literal["boolean"] = "boolean"


class Seed(StrEnum):
    experiment = "experiment"
    cohort = "cohort"
    participant = "participant"
    custom = "custom"


class ShuffleConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    shuffle: bool
    seed: Seed
    customSeed: str


class Weight(RootModel[float]):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    root: Annotated[float, Field(ge=1.0)]


class BalanceStrategy(StrEnum):
    round_robin = "round_robin"
    random = "random"


class BalanceAcross(StrEnum):
    experiment = "experiment"
    cohort = "cohort"


class StageTextConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    primaryText: str
    infoText: str
    helpText: str


class StageProgressConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    minParticipants: int
    waitForAllParticipants: bool
    showParticipantProgress: bool
    agentJoinTimeMs: int | None = None
    maxWaitTimeMs: int | None = None
    lockCohortAtTwoParticipants: bool | None = None
    waitStageHasTimer: bool | None = None
    pIdToAgentId: Annotated[dict[str, str] | None, Field(title="PIdToAgentId")] = None


class ParsedItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    date: Annotated[str, Field(min_length=1)]
    close: float


class CustomCard(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    title: str
    value: str
    subtext: str
    enabled: bool
    description: str | None = None


class Stock(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    description: str
    csvData: str
    parsedData: list[ParsedItem]
    customCards: list[CustomCard]


class MultiAssetAllocationStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["multiAssetAllocation"] = "multiAssetAllocation"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    stockOptions: list[Stock]
    stockInfoStageId: str
    attributeRatingStageId: str | None = None
    attributeGroupId: str | None = None
    attributeGroupAsAttributeIds: list[str] | None = None
    randomizeStockOrder: bool | None = None
    randomizeAttributeOrder: bool | None = None
    useRatingUtility: bool | None = None
    showUtilityScore: bool | None = None
    visibleRanks: list[float] | None = None
    anonymizeAssets: bool | None = None
    anonymizeLeadupWord: str | None = None
    groupAsAttrDescriptions: Annotated[
        dict[str, str] | None, Field(title="GroupAsAttrDescriptions")
    ] = None


class AttributeGroup(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    name: str
    description: str
    attributeIds: list[str]


class AttributeRatingStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["attributeRating"] = "attributeRating"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    attributeList: list[Stock]
    attributeInfoStageId: str
    randomizeAttributeOrder: bool | None = None
    attributeGroups: list[AttributeGroup] | None = None


class DefaultChatDiscussion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: str
    type: Literal["DEFAULT"] = "DEFAULT"
    description: str


class DiscussionItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: str
    imageId: str
    name: str


class ChipItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: str
    name: str
    avatar: str
    canBuy: bool
    canSell: bool
    startingQuantity: float
    lowerValue: float
    upperValue: float


class TextQuestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["text"] = "text"
    questionTitle: str
    correctAnswer: str


class MultipleChoiceItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    imageId: str
    text: str


class FlipCard(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    title: str
    frontContent: Annotated[str, Field(min_length=1)]
    backContent: Annotated[str, Field(min_length=1)]


class InfoStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["info"] = "info"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    infoLines: list[str]
    youtubeVideoId: str | None = None


class Currency(StrEnum):
    EUR = "EUR"
    GBP = "GBP"
    USD = "USD"


class DefaultPayoutItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    type: Literal["DEFAULT"] = "DEFAULT"
    name: str
    description: str
    isActive: bool
    stageId: str
    baseCurrencyAmount: float
    randomSelectionId: str


class ChipPayoutItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    type: Literal["CHIP"] = "CHIP"
    name: str
    description: str
    isActive: bool
    stageId: str
    baseCurrencyAmount: float


class SurveyPayoutItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    type: Literal["SURVEY"] = "SURVEY"
    name: str
    description: str
    isActive: bool
    stageId: str
    baseCurrencyAmount: float
    rankingStageId: str | None = None
    questionMap: Annotated[dict[str, float | None], Field(title="QuestionMap")]


class PrivateChatStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["privateChat"] = "privateChat"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    timeLimitInMinutes: float | None = None
    numUtterances: float | None = None
    requireFullTime: bool | None = None
    isTurnBasedChat: bool | None = None
    minNumberOfTurns: float | None = None
    maxNumberOfTurns: float | None = None
    preventCancellation: bool | None = None


class ProfileType(StrEnum):
    DEFAULT = "DEFAULT"
    DEFAULT_GENDERED = "DEFAULT_GENDERED"
    ANONYMOUS_ANIMAL = "ANONYMOUS_ANIMAL"
    ANONYMOUS_PARTICIPANT = "ANONYMOUS_PARTICIPANT"


class ProfileStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["profile"] = "profile"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    profileType: ProfileType


class Strategy(StrEnum):
    none = "none"
    condorcet = "condorcet"


RankingItem = MultipleChoiceItem


class ParticipantRankingStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["ranking"] = "ranking"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    rankingType: Literal["participants"] = "participants"
    strategy: Strategy
    enableSelfVoting: bool


class RevealAudience(StrEnum):
    CURRENT = "CURRENT"
    ALL = "ALL"


class ChipRevealItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["chip"] = "chip"
    revealAudience: RevealAudience


class RankingRevealItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["ranking"] = "ranking"
    revealAudience: RevealAudience


class SurveyRevealItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["survey"] = "survey"
    revealAudience: RevealAudience
    revealScorableOnly: bool


class DisplayMode(StrEnum):
    full = "full"
    scoreOnly = "scoreOnly"


class MultiAssetAllocationRevealItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["multiAssetAllocation"] = "multiAssetAllocation"
    revealAudience: RevealAudience
    displayMode: DisplayMode


class Role(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    name: str
    displayLines: list[str]
    minParticipants: int
    maxParticipants: int | None = None


class RoleStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["role"] = "role"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    roles: list[Role]


class SalespersonStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["salesperson"] = "salesperson"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig


class StockInfoStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["stockinfo"] = "stockinfo"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    stocks: list[Stock]
    visibleStockIds: list[str] | None = None
    showBestYearCard: bool
    showWorstYearCard: bool
    requireViewAllStocks: bool
    useQuarterlyMarkers: bool
    showInvestmentGrowth: bool
    useSharedYAxis: bool
    initialInvestment: Annotated[float | None, Field(ge=1.0)] = 1000
    currency: str | None = "USD"
    introText: str | None = None


class ComparisonOperator(StrEnum):
    equals = "equals"
    not_equals = "not_equals"
    greater_than = "greater_than"
    greater_than_or_equal = "greater_than_or_equal"
    less_than = "less_than"
    less_than_or_equal = "less_than_or_equal"
    contains = "contains"
    not_contains = "not_contains"


class ConditionTargetReference(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    stageId: Annotated[str, Field(min_length=1)]
    questionId: Annotated[str, Field(min_length=1)]


class ConditionOperator(StrEnum):
    and_ = "and"
    or_ = "or"


class TOSStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["tos"] = "tos"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    tosLines: list[str]


class DefaultAutoTransferConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["default"] = "default"
    autoCohortParticipantConfig: CohortParticipantConfig
    minParticipants: Annotated[int, Field(ge=1)]
    maxParticipants: Annotated[int, Field(ge=1)]


class SurveyAutoTransferConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["survey"] = "survey"
    autoCohortParticipantConfig: CohortParticipantConfig
    surveyStageId: Annotated[str, Field(min_length=1)]
    surveyQuestionId: Annotated[str, Field(min_length=1)]
    participantCounts: Annotated[dict[str, int], Field(title="ParticipantCounts")]


class ParticipantProfileBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    pronouns: str | None = None
    avatar: str | None = None
    name: str | None = None


class ApiKeyType(StrEnum):
    GEMINI = "GEMINI"
    OPENAI = "OPENAI"
    CLAUDE = "CLAUDE"
    OLLAMA = "OLLAMA"


class AgentModelSettings(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    apiType: Annotated[ApiKeyType, Field(title="ApiKeyType")]
    modelName: str


class ChatStageType(StrEnum):
    chat = "chat"
    privateChat = "privateChat"


class ReasoningLevel(StrEnum):
    off = "off"
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"


class CustomRequestBodyField(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    name: str
    value: str


class GoogleThinkingLevel(StrEnum):
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"


class GoogleThinkingConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    thinkingBudget: int | None = None
    includeThoughts: bool | None = None
    thinkingLevel: Annotated[GoogleThinkingLevel | None, Field(title="GoogleThinkingLevel")] = None


class GoogleSafetyCategory(StrEnum):
    HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
    HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    HARM_CATEGORY_CIVIC_INTEGRITY = "HARM_CATEGORY_CIVIC_INTEGRITY"


class GoogleSafetyThreshold(StrEnum):
    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
    HARM_BLOCK_THRESHOLD_UNSPECIFIED = "HARM_BLOCK_THRESHOLD_UNSPECIFIED"


class GoogleSafetySetting(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    category: Annotated[GoogleSafetyCategory, Field(title="GoogleSafetyCategory")]
    threshold: Annotated[GoogleSafetyThreshold, Field(title="GoogleSafetyThreshold")]


class ReasoningEffort(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class AnthropicThinkingType(StrEnum):
    enabled = "enabled"
    disabled = "disabled"


class AnthropicThinkingConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Annotated[AnthropicThinkingType, Field(title="AnthropicThinkingType")]
    budgetTokens: int | None = None


class AnthropicCacheTtl(StrEnum):
    field_5m = "5m"
    field_1h = "1h"


class AnthropicCacheControl(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["ephemeral"] = "ephemeral"
    ttl: Annotated[AnthropicCacheTtl | None, Field(title="AnthropicCacheTtl")] = None


class OpenAIProviderOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    reasoningEffort: Annotated[ReasoningEffort | None, Field(title="ReasoningEffort")] = None
    parallelToolCalls: bool | None = None


class OllamaProviderOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    numCtx: int | None = None
    numPredict: int | None = None


class StructuredOutputType(StrEnum):
    NONE = "NONE"
    JSON_FORMAT = "JSON_FORMAT"
    JSON_SCHEMA = "JSON_SCHEMA"


class StructuredOutputDataType(StrEnum):
    STRING = "STRING"
    NUMBER = "NUMBER"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"
    ARRAY = "ARRAY"
    OBJECT = "OBJECT"
    ENUM = "ENUM"


class StageKind(StrEnum):
    info = "info"
    tos = "tos"
    profile = "profile"
    chat = "chat"
    chip = "chip"
    comprehension = "comprehension"
    flipcard = "flipcard"
    ranking = "ranking"
    payout = "payout"
    privateChat = "privateChat"
    reveal = "reveal"
    salesperson = "salesperson"
    stockinfo = "stockinfo"
    assetAllocation = "assetAllocation"
    multiAssetAllocation = "multiAssetAllocation"
    attributeRating = "attributeRating"
    role = "role"
    survey = "survey"
    surveyPerParticipant = "surveyPerParticipant"
    transfer = "transfer"


class ParticipantPersona(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    name: str
    description: str | None = None
    isDefaultAddToCohort: bool | None = None
    defaultProfile: ParticipantProfileBase | None = None
    defaultModelSettings: AgentModelSettings | None = None
    type: Literal["participant"] = "participant"
    character: Annotated[dict[str, str] | None, Field(title="Character")] = None
    assistantId: str | None = None


class AssistantPersona(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    name: str
    description: str | None = None
    isDefaultAddToCohort: bool | None = None
    defaultProfile: ParticipantProfileBase | None = None
    defaultModelSettings: AgentModelSettings | None = None
    type: Literal["assistant"] = "assistant"
    minCallIntervalMs: float | None = None


class StockConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    stockInfoStageId: Annotated[str | None, Field(min_length=1)] = None
    stockAId: str | None = None
    stockBId: str | None = None
    stockA: Stock | None = None
    stockB: Stock | None = None


class AssetAllocationStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["assetAllocation"] = "assetAllocation"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    stockConfig: Annotated[StockConfig, Field(title="StockConfig")]


class CompareChatDiscussion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: str
    type: Literal["COMPARE"] = "COMPARE"
    description: str
    items: list[DiscussionItem]


class ChipStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["chip"] = "chip"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    enableChat: bool
    numRounds: float
    chips: list[ChipItem]


class McQuestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["mc"] = "mc"
    questionTitle: str
    options: list[MultipleChoiceItem]
    correctAnswerId: str


class ComprehensionStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["comprehension"] = "comprehension"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    questions: list[TextQuestion | McQuestion]


class FlipCardStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["flipcard"] = "flipcard"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    cards: list[FlipCard]
    enableSelection: bool
    allowMultipleSelections: bool
    requireConfirmation: bool
    minUniqueCardsFlippedRequirement: float
    shuffleCards: bool


class PayoutStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["payout"] = "payout"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    currency: Currency
    payoutItems: list[DefaultPayoutItem | ChipPayoutItem | SurveyPayoutItem]
    averageAllPayoutItems: bool


class ItemRankingStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["ranking"] = "ranking"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    rankingType: Literal["items"] = "items"
    strategy: Strategy
    rankingItems: list[RankingItem]


class RevealStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["reveal"] = "reveal"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    items: list[
        ChipRevealItem | RankingRevealItem | SurveyRevealItem | MultiAssetAllocationRevealItem
    ]


class ComparisonCondition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    type: Literal["comparison"] = "comparison"
    target: ConditionTargetReference
    operator: Annotated[ComparisonOperator, Field(title="ComparisonOperator")]
    value: str | float | bool


class MediatorPersona(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    name: str
    description: str | None = None
    isDefaultAddToCohort: bool | None = None
    defaultProfile: ParticipantProfileBase | None = None
    defaultModelSettings: AgentModelSettings | None = None
    type: Literal["mediator"] = "mediator"


class GoogleProviderOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    thinkingConfig: GoogleThinkingConfig | None = None
    safetySettings: list[GoogleSafetySetting] | None = None


class AnthropicProviderOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    thinking: AnthropicThinkingConfig | None = None
    effort: Annotated[ReasoningEffort | None, Field(title="ReasoningEffort")] = None
    cacheControl: AnthropicCacheControl | None = None
    sendReasoning: bool | None = None


class ChatStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["chat"] = "chat"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    timeLimitInMinutes: float | None = None
    numUtterances: float | None = None
    requireFullTime: bool | None = None
    discussions: list[DefaultChatDiscussion | CompareChatDiscussion]


class RankingStageConfig(RootModel[ItemRankingStageConfig | ParticipantRankingStageConfig]):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    root: Annotated[
        ItemRankingStageConfig | ParticipantRankingStageConfig,
        Field(title="RankingStageConfig"),
    ]


class ProviderOptionsMap(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    google: GoogleProviderOptions | None = None
    anthropic: AnthropicProviderOptions | None = None
    openai: OpenAIProviderOptions | None = None
    ollama: OllamaProviderOptions | None = None


class ModelGenerationConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    maxTokens: int | None = None
    stopSequences: list[str] | None = None
    temperature: float | None = None
    topP: float | None = None
    frequencyPenalty: float | None = None
    presencePenalty: float | None = None
    reasoningLevel: Annotated[ReasoningLevel | None, Field(title="ReasoningLevel")] = None
    reasoningBudget: int | None = None
    includeReasoning: bool | None = None
    disableSafetyFilters: bool | None = None
    providerOptions: ProviderOptionsMap | None = None
    customRequestBodyFields: list[CustomRequestBodyField] | None = None


class Experiment(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: str
    versionId: float
    metadata: Annotated[Metadata, Field(title="Metadata")]
    permissions: Annotated[Permissions, Field(title="Permissions")]
    defaultCohortConfig: CohortParticipantConfig
    prolificConfig: Annotated[ProlificConfig, Field(title="ProlificConfig")]
    stageIds: list[str]
    cohortLockMap: Annotated[dict[str, bool], Field(title="CohortLockMap")]
    variableConfigs: (
        list[
            StaticVariableConfig
            | RandomPermutationVariableConfig
            | BalancedAssignmentVariableConfig
        ]
        | None
    ) = None
    variableMap: Annotated[dict[str, str] | None, Field(title="VariableMap")] = None
    cohortDefinitions: list[CohortDefinition] | None = None


class ExperimentTemplate(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    id: str
    experiment: Annotated[Experiment, Field(title="Experiment")]
    stageConfigs: list[
        AssetAllocationStageConfig
        | MultiAssetAllocationStageConfig
        | AttributeRatingStageConfig
        | ChatStageConfig
        | ChipStageConfig
        | ComprehensionStageConfig
        | FlipCardStageConfig
        | InfoStageConfig
        | PayoutStageConfig
        | PrivateChatStageConfig
        | ProfileStageConfig
        | ItemRankingStageConfig
        | ParticipantRankingStageConfig
        | RevealStageConfig
        | RoleStageConfig
        | SalespersonStageConfig
        | StockInfoStageConfig
        | SurveyPerParticipantStageConfig
        | SurveyStageConfig
        | TOSStageConfig
        | TransferStageConfig
    ]
    agentMediators: list[AgentMediatorTemplate]
    agentParticipants: list[AgentParticipantTemplate]
    agentAssistants: list[AgentAssistantTemplate]


class ExperimentCreation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    collectionName: CollectionName
    experimentTemplate: Annotated[ExperimentTemplate, Field(title="ExperimentTemplate")]


class DeliberateLabAPISchemas(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    experimentCreation: Annotated[ExperimentCreation, Field(title="ExperimentCreation")]


class StaticVariableConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    type: Literal["static"] = "static"
    scope: Scope
    definition: VariableDefinition
    value: str
    cohortValues: Annotated[dict[str, str] | None, Field(title="CohortValues")] = None


class Object(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )
    type: Literal["object"] = "object"
    properties: Annotated[
        dict[str, String | Number | Integer | Boolean | Object | Array] | None,
        Field(title="Properties"),
    ] = None


class Array(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )
    type: Literal["array"] = "array"
    items: Annotated[
        String | Number | Integer | Boolean | Object | Array | None,
        Field(title="JSONSchemaDefinition"),
    ] = None


class VariableDefinition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    name: Annotated[str, Field(min_length=1)]
    description: str
    schema_: Annotated[
        String | Number | Integer | Boolean | Object | Array,
        Field(alias="schema", title="JSONSchemaDefinition"),
    ]


class RandomPermutationVariableConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    type: Literal["random_permutation"] = "random_permutation"
    scope: Scope
    definition: VariableDefinition
    shuffleConfig: ShuffleConfig
    values: list[str]
    numToSelect: Annotated[float | None, Field(ge=1.0)] = None
    expandListToSeparateVariables: bool | None = None


class BalancedAssignmentVariableConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    type: Literal["balanced_assignment"] = "balanced_assignment"
    scope: Scope
    definition: VariableDefinition
    values: list[str]
    weights: list[Weight] | None = None
    balanceStrategy: BalanceStrategy
    balanceAcross: BalanceAcross


class SurveyPerParticipantStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["surveyPerParticipant"] = "surveyPerParticipant"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    questions: list[
        TextSurveyQuestion
        | CheckSurveyQuestion
        | MultipleChoiceSurveyQuestion
        | ScaleSurveyQuestion
    ]
    enableSelfSurvey: bool


class TextSurveyQuestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["text"] = "text"
    questionTitle: str
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )
    minCharCount: float | None = None
    maxCharCount: float | None = None


class ConditionGroup(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    type: Literal["group"] = "group"
    operator: Annotated[ConditionOperator, Field(title="ConditionOperator")]
    conditions: list[ComparisonCondition | ConditionGroup]


class CheckSurveyQuestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["check"] = "check"
    questionTitle: str
    isRequired: bool
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class MultipleChoiceSurveyQuestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["mc"] = "mc"
    questionTitle: str
    options: list[MultipleChoiceItem]
    correctAnswerId: str | None = None
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class ScaleSurveyQuestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["scale"] = "scale"
    questionTitle: str
    upperValue: float
    upperText: str
    lowerValue: float
    lowerText: str
    middleText: str | None = None
    useSlider: bool | None = None
    stepSize: Annotated[float | None, Field(ge=1.0)] = None
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class SurveyStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["survey"] = "survey"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    questions: list[
        TextSurveyQuestion
        | CheckSurveyQuestion
        | MultipleChoiceSurveyQuestion
        | ScaleSurveyQuestion
    ]


class TransferStageConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    kind: Literal["transfer"] = "transfer"
    name: Annotated[str, Field(min_length=1)]
    descriptions: StageTextConfig
    progress: StageProgressConfig
    enableTimeout: bool
    timeoutSeconds: float
    autoTransferConfig: (
        DefaultAutoTransferConfig | SurveyAutoTransferConfig | ConditionAutoTransferConfig | None
    ) = None


class ConditionAutoTransferConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["condition"] = "condition"
    autoCohortParticipantConfig: CohortParticipantConfig
    transferGroups: Annotated[list[TransferGroup], Field(min_length=1)]
    enableGroupBalancing: bool | None = None


class TransferGroup(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    composition: Annotated[list[GroupComposition], Field(min_length=1)]
    targetCohortAlias: Annotated[str | None, Field(min_length=1)] = None


class GroupComposition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    condition: Annotated[ComparisonCondition | ConditionGroup, Field(title="Condition")]
    minCount: Annotated[int, Field(ge=1)]
    maxCount: Annotated[int, Field(ge=1)]


class AgentMediatorTemplate(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    persona: MediatorPersona
    promptMap: Annotated[
        dict[str, ChatPromptConfig | GenericPromptConfig], Field(title="PromptMap")
    ]


class ChatPromptConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    type: Annotated[ChatStageType, Field(title="ChatStageType")]
    prompt: dict[
        str,
        list[
            TextPromptItem
            | ProfileInfoPromptItem
            | ParticipantInfoPromptItem
            | ParticipantChatInputPromptItem
            | ProfileContextPromptItem
            | StageContextPromptItem
            | PromptItemGroup
            | InitializationContextPromptItem
        ],
    ]
    order: dict[int, list[str]]
    addTo: dict[str, list[str]]
    includeScaffoldingInPrompt: bool | None = None
    includeConcessionInPrompt: bool | None = None
    includeThoughtHistory: list[str] | None = None
    includePersona: list[str] | None = None
    numRetries: int | None = None
    generationConfig: ModelGenerationConfig | None = None
    structuredOutputConfig: (
        StructuredOutputConfig
        | ChatMediatorStructuredOutputConfig
        | AgentParticipantStructuredOutputConfig
        | AssistantStructuredOutputConfig
        | None
    ) = None
    chatSettings: AgentChatSettings | None = None
    initializationContextPrompt: (
        list[
            TextPromptItem
            | ProfileInfoPromptItem
            | ParticipantInfoPromptItem
            | ParticipantChatInputPromptItem
            | ProfileContextPromptItem
            | StageContextPromptItem
            | PromptItemGroup
            | InitializationContextPromptItem
        ]
        | None
    ) = None
    shouldRespondPrompt: (
        list[
            TextPromptItem
            | ProfileInfoPromptItem
            | ParticipantInfoPromptItem
            | ParticipantChatInputPromptItem
            | ProfileContextPromptItem
            | StageContextPromptItem
            | PromptItemGroup
            | InitializationContextPromptItem
        ]
        | None
    ) = None
    concedeStrength: int | None = None
    shouldConcedePrompt: (
        list[
            TextPromptItem
            | ProfileInfoPromptItem
            | ParticipantInfoPromptItem
            | ParticipantChatInputPromptItem
            | ProfileContextPromptItem
            | StageContextPromptItem
            | PromptItemGroup
            | InitializationContextPromptItem
        ]
        | None
    ) = None
    thoughtPrompt: (
        list[
            TextPromptItem
            | ProfileInfoPromptItem
            | ParticipantInfoPromptItem
            | ParticipantChatInputPromptItem
            | ProfileContextPromptItem
            | StageContextPromptItem
            | PromptItemGroup
            | InitializationContextPromptItem
        ]
        | None
    ) = None
    characterPrompt: (
        list[
            TextPromptItem
            | ProfileInfoPromptItem
            | ParticipantInfoPromptItem
            | ParticipantChatInputPromptItem
            | ProfileContextPromptItem
            | StageContextPromptItem
            | PromptItemGroup
            | InitializationContextPromptItem
        ]
        | None
    ) = None
    initializationInformation: str | None = None
    initializationContext: Annotated[
        dict[str, str] | None, Field(title="InitializationContext")
    ] = None
    minParticipantMessagesBeforeResponding: int | None = None


class TextPromptItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["TEXT"] = "TEXT"
    text: str
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class ProfileInfoPromptItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["PROFILE_INFO"] = "PROFILE_INFO"
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class ParticipantInfoPromptItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["PARTICIPANT_INFO"] = "PARTICIPANT_INFO"
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class ParticipantChatInputPromptItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["PARTICIPANT_CHAT_INPUT"] = "PARTICIPANT_CHAT_INPUT"
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class ProfileContextPromptItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["PROFILE_CONTEXT"] = "PROFILE_CONTEXT"
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class StageContextPromptItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["STAGE_CONTEXT"] = "STAGE_CONTEXT"
    stageId: str
    includePrimaryText: bool
    includeInfoText: bool
    includeHelpText: bool
    includeStageDisplay: bool
    includeParticipantAnswers: bool
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class PromptItemGroup(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["GROUP"] = "GROUP"
    title: str
    items: list[
        TextPromptItem
        | ProfileInfoPromptItem
        | ParticipantInfoPromptItem
        | ParticipantChatInputPromptItem
        | ProfileContextPromptItem
        | StageContextPromptItem
        | PromptItemGroup
    ]
    shuffleConfig: ShuffleConfig | None = None
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class InitializationContextPromptItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Literal["INITIALIZATION_CONTEXT"] = "INITIALIZATION_CONTEXT"
    condition: Annotated[ComparisonCondition | ConditionGroup | None, Field(title="Condition")] = (
        None
    )


class StructuredOutputConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    enabled: bool
    type: Annotated[StructuredOutputType, Field(title="StructuredOutputType")]
    schema_: Annotated[StructuredOutputSchema | None, Field(alias="schema")] = None
    appendToPrompt: bool


class StructuredOutputSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    type: Annotated[StructuredOutputDataType, Field(title="StructuredOutputDataType")]
    description: str | None = None
    properties: list[StructuredOutputSchemaProperty] | None = None
    arrayItems: StructuredOutputSchema | None = None
    enumItems: list[str] | None = None


class StructuredOutputSchemaProperty(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    name: str
    schema_: Annotated[StructuredOutputSchema, Field(alias="schema")]


class ChatMediatorStructuredOutputConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    enabled: bool
    type: Annotated[StructuredOutputType, Field(title="StructuredOutputType")]
    schema_: Annotated[StructuredOutputSchema | None, Field(alias="schema")] = None
    appendToPrompt: bool
    messageField: str
    explanationField: str
    readyToEndField: str


class AgentParticipantStructuredOutputConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    enabled: bool
    type: Annotated[StructuredOutputType, Field(title="StructuredOutputType")]
    schema_: Annotated[StructuredOutputSchema | None, Field(alias="schema")] = None
    appendToPrompt: bool
    shouldRespondField: str | None = None
    messageField: str | None = None
    explanationField: str | None = None
    readyToEndField: str | None = None
    shouldConcedeField: str | None = None


class AssistantStructuredOutputConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    enabled: bool
    type: Annotated[StructuredOutputType, Field(title="StructuredOutputType")]
    schema_: Annotated[StructuredOutputSchema | None, Field(alias="schema")] = None
    appendToPrompt: bool
    messageField: str | None = None


class AgentChatSettings(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    wordsPerMinute: float | None = None
    minMessagesBeforeResponding: int
    canSelfTriggerCalls: bool
    maxResponses: int | None = None
    initialMessage: str
    shouldRespondPrompt: (
        list[
            TextPromptItem
            | ProfileInfoPromptItem
            | ParticipantInfoPromptItem
            | ParticipantChatInputPromptItem
            | ProfileContextPromptItem
            | StageContextPromptItem
            | PromptItemGroup
            | InitializationContextPromptItem
        ]
        | None
    ) = None
    minParticipantMessagesBeforeResponding: float | None = None
    shouldConcedePrompt: (
        TextPromptItem
        | ProfileInfoPromptItem
        | ParticipantInfoPromptItem
        | ParticipantChatInputPromptItem
        | ProfileContextPromptItem
        | StageContextPromptItem
        | PromptItemGroup
        | InitializationContextPromptItem
        | None
    ) = None
    concedeStrength: int
    thoughtPrompt: (
        TextPromptItem
        | ProfileInfoPromptItem
        | ParticipantInfoPromptItem
        | ParticipantChatInputPromptItem
        | ProfileContextPromptItem
        | StageContextPromptItem
        | PromptItemGroup
        | InitializationContextPromptItem
        | None
    ) = None


class GenericPromptConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )
    id: Annotated[str, Field(min_length=1)]
    type: Annotated[StageKind, Field(title="StageKind")]
    prompt: dict[
        str,
        list[
            TextPromptItem
            | ProfileInfoPromptItem
            | ParticipantInfoPromptItem
            | ParticipantChatInputPromptItem
            | ProfileContextPromptItem
            | StageContextPromptItem
            | PromptItemGroup
            | InitializationContextPromptItem
        ],
    ]
    order: dict[int, list[str]]
    addTo: dict[str, list[str]]
    includeScaffoldingInPrompt: bool | None = None
    includeConcessionInPrompt: bool | None = None
    includeThoughtHistory: list[str] | None = None
    includePersona: list[str] | None = None
    numRetries: int | None = None
    generationConfig: ModelGenerationConfig | None = None
    structuredOutputConfig: StructuredOutputConfig | None = None


class AgentParticipantTemplate(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    persona: ParticipantPersona
    promptMap: Annotated[
        dict[str, ChatPromptConfig | GenericPromptConfig], Field(title="PromptMap")
    ]


class AgentAssistantTemplate(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    persona: AssistantPersona
    promptMap: Annotated[
        dict[str, ChatPromptConfig | GenericPromptConfig], Field(title="PromptMap")
    ]


class JSONSchemaDefinition(RootModel[String | Number | Integer | Boolean | Object | Array]):
    model_config = ConfigDict(
        populate_by_name=True,
    )
    root: Annotated[
        String | Number | Integer | Boolean | Object | Array,
        Field(title="JSONSchemaDefinition"),
    ]


Experiment.model_rebuild()
ExperimentTemplate.model_rebuild()
StaticVariableConfig.model_rebuild()
Object.model_rebuild()
Array.model_rebuild()
SurveyPerParticipantStageConfig.model_rebuild()
TextSurveyQuestion.model_rebuild()
ConditionGroup.model_rebuild()
TransferStageConfig.model_rebuild()
ConditionAutoTransferConfig.model_rebuild()
TransferGroup.model_rebuild()
AgentMediatorTemplate.model_rebuild()
AgentAssistantTemplate.model_rebuild()
ChatPromptConfig.model_rebuild()
StructuredOutputConfig.model_rebuild()
StructuredOutputSchema.model_rebuild()
