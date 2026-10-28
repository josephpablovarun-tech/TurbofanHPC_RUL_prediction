"""Typed contracts: immutable request context, tool inputs/outputs, findings and the final response.

Tool outputs repeat the six context identifiers so the harness can detect a mismatched result
(roadmap: "Shared immutable context"). Python 3.9 compatible: typing generics, no ``X | Y``.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field

DatasetSplit = Literal["train", "test"]
Status = Literal["ok", "partial", "insufficient_evidence", "error"]
DataRole = Literal["train_in_sample", "validation_holdout", "official_test"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequestContext(Strict):
    """Bound by the harness to every tool call; neither the user text nor the LLM can change it."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    engine_id: int = Field(ge=1)
    dataset_split: DatasetSplit
    cutoff_cycle: int = Field(ge=1)
    question: str
    model_version: str
    corpus_version: str

    def selection_key(self) -> Tuple[str, int, int]:
        return (self.dataset_split, self.engine_id, self.cutoff_cycle)


# ---------------------------------------------------------------- tool inputs (no context fields allowed)
class PredictRULInput(Strict):
    pass


class SensorTrendsInput(Strict):
    sensors: Optional[List[str]] = Field(default=None, description="Optional subset of the 12 model sensors, e.g. ['sensor_11'].")
    window: int = Field(default=30, ge=5, le=100, description="Number of most recent observed cycles to summarise.")
    chart_last_n: Optional[int] = Field(default=None, ge=5, le=100, description="If set, the chart shows only the last N observed cycles.")
    top_n: int = Field(default=4, ge=1, le=12, description="How many sensors to summarise when none are named (ranked by movement relative to the engine's own spread).")
    chart_mode: Literal["raw", "change"] = Field(
        default="raw", description="'raw' plots sensor values. 'change' plots each sensor's change since the first plotted cycle in units of its own spread, "
                                   "so sensors with different units share one axis.")


class ModelMetadataInput(Strict):
    pass


class ScreenFleetInput(Strict):
    threshold_cycles: Optional[float] = Field(default=None, ge=0, le=1000, description="Optional user screening threshold: list engines whose predicted RUL is below it.")
    top_n: int = Field(default=10, ge=1, le=20, description="How many lowest-RUL engines to return.")


class SearchDocumentsInput(Strict):
    query: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=4, ge=1, le=4)


# ---------------------------------------------------------------- tool outputs
class ToolOutput(Strict):
    request_id: str
    engine_id: int
    dataset_split: str
    cutoff_cycle: int
    model_version: str
    corpus_version: str


class PredictRULOutput(ToolOutput):
    observed_cycles: int
    predicted_rul_cycles: float
    support_rul_cycles: float
    support_model_version: str
    model_agreement_gap_cycles: float
    model_agreement: Literal["close", "moderate", "wide"]
    demo_band: Literal["Normal", "Watch", "Review"]
    feature_schema_version: str
    data_role: DataRole
    warnings: List[str] = []
    cache_status: Literal["hit", "miss"] = "miss"
    feature_vector: Dict[str, float] = {}


class SensorSummary(Strict):
    sensor: str
    unit: Optional[str] = None  # units are not verified in this project
    first: float
    last: float
    min: float
    max: float
    window_mean: float
    change_over_window: float
    slope_per_cycle: Optional[float] = None
    recent_slope_per_cycle: Optional[float] = None  # least-squares slope over the most recent third of the window (at least 5 cycles)
    recent_window: Optional[int] = None
    normalised_trend: Optional[float] = None  # |slope| x (window-1) / the engine's own std over its observed history (unitless)
    evidence_id: str


class ChartData(Strict):
    cycles: List[int]
    series: Dict[str, List[float]]
    mode: Literal["raw", "change"] = "raw"


class SensorTrendsOutput(ToolOutput):
    observed_cycles: int
    window_used: int
    selection_rule: str
    sensors: List[SensorSummary]
    chart: ChartData
    quality_warnings: List[str] = []
    data_quality: Dict[str, Any] = {}
    range_check: Dict[str, Any] = {}


class ModelMetadataOutput(ToolOutput):
    metadata: Dict[str, Any]


class Passage(Strict):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    revision: str
    text: str
    score: float


class SearchDocumentsOutput(ToolOutput):
    query: str
    results: List[Passage]


class FleetRow(Strict):
    rank: int
    engine_id: int
    dataset_split: str
    observed_cycle: int
    predicted_rul_cycles: float
    demo_band: str
    flags: List[str] = []


class ScreenFleetOutput(ToolOutput):
    """Latest-observed-cycle predictions for every engine of the selected split (test split only)."""
    n_engines: int
    threshold_cycles: Optional[float] = None
    n_below_threshold: Optional[int] = None
    rows: List[FleetRow]
    ranking_rule: str


class ToolSpec(Strict):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    name: str
    description: str
    input_model: Any
    output_model: Any


TOOL_SPECS: Dict[str, ToolSpec] = {
    "predict_rul": ToolSpec(
        name="predict_rul",
        description="Remaining useful life (operating cycles) for the selected engine at the selected cutoff, from the saved models. Takes no arguments.",
        input_model=PredictRULInput, output_model=PredictRULOutput),
    "get_sensor_trends": ToolSpec(
        name="get_sensor_trends",
        description="Summaries and chart data of recent sensor readings for the selected engine up to the cutoff. Optional sensors and window.",
        input_model=SensorTrendsInput, output_model=SensorTrendsOutput),
    "get_model_metadata": ToolSpec(
        name="get_model_metadata",
        description="Training domain, feature schema, evaluation metrics (with their dataset split), target policy and limitations of the models. Takes no arguments.",
        input_model=ModelMetadataInput, output_model=ModelMetadataOutput),
    "screen_fleet": ToolSpec(
        name="screen_fleet",
        description="Rank test-split engines by predicted RUL at each engine's own latest observed cycle, optionally with a user screening threshold. Test split only.",
        input_model=ScreenFleetInput, output_model=ScreenFleetOutput),
    "search_documents": ToolSpec(
        name="search_documents",
        description="Search the dataset guide, sensor glossary and model card. Returns passages with chunk ids.",
        input_model=SearchDocumentsInput, output_model=SearchDocumentsOutput),
}


# ---------------------------------------------------------------- specialist findings (LLM or deterministic)
class SensorFinding(Strict):
    sensor: str
    statement: str
    evidence_id: str


class EngineFindings(Strict):
    summary: str = Field(default="", max_length=700)
    sensor_findings: List[SensorFinding] = Field(default=[], max_length=6)


class DocStatement(Strict):
    statement: str = Field(max_length=900)
    source_ids: List[str]


class DocFindings(Strict):
    statements: List[DocStatement] = Field(default=[], max_length=6)


# ---------------------------------------------------------------- final response
class PredictionBlock(Strict):
    predicted_rul_cycles: float
    support_rul_cycles: float
    model_agreement: str
    model_agreement_gap_cycles: float
    demo_band: str
    model_version: str
    support_model_version: str
    feature_schema_version: str
    data_role: str
    observed_cycles: int
    warnings: List[str] = []


class Citation(Strict):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    revision: str


class TraceEvent(Strict):
    request_id: str
    seq: int
    agent: str
    tool: str
    args: Dict[str, Any] = {}
    started_utc: str
    duration_ms: float
    status: Literal["ok", "error", "rejected", "timeout", "budget_exceeded", "context_mismatch", "deadline"]
    model_version: str
    corpus_version: str
    result_ref: Optional[str] = None
    cache_status: Optional[str] = None
    error: Optional[str] = None


class LifeSensorRow(Strict):
    sensor: str
    direction: Literal["rises", "falls"]
    spearman_rho: float  # rank correlation of the sensor with the cycle number over the engine's full life
    late_minus_early: float  # (mean of the last 20% of cycles - mean of the first 20%) / the sensor's own std over the full life; unitless


class LifeView(Strict):
    """Retrospective summary of a TRAIN engine's complete run to failure (never produced for the test split)."""
    dataset_split: DatasetSplit
    engine_id: int
    total_cycles: int
    data_role: DataRole
    model_version: str
    ranking: List[LifeSensorRow] = []
    notes: List[str] = []


class LifeChart(Strict):
    title: str
    cycles: List[int]
    series: Dict[str, List[float]]
    caption: str = ""
    zero_baseline: bool = True  # False: the y-axis fits the data (sensor values sit far from zero, so a zero baseline hides the trend)


class AssessmentResponse(Strict):
    context: RequestContext
    route: str
    agent_mode: str
    status: Status
    prediction: Optional[PredictionBlock] = None
    prediction_unavailable_reason: Optional[str] = None
    sensor_findings: List[SensorFinding] = []
    explanation: str = ""
    doc_statements: List[DocStatement] = []
    citations: List[Citation] = []
    limitations: List[str] = []
    warnings: List[str] = []
    answer_markdown: str = ""
    chart: Optional[ChartData] = None
    life: Optional[LifeView] = None
    life_charts: List[LifeChart] = []
    intent: str = "assess"
    response_mode: str = "answer"  # answer | clarify | decline | reject | not_implemented
    table: List[Dict[str, Any]] = []
    evidence: List[str] = []
    trace_id: str
    trace: List[TraceEvent] = []
    tool_calls: int = 0
    latency_s: float = 0.0
