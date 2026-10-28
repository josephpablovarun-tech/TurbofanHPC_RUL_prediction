"""Response boundaries: fixed, reviewed wording for requests the assistant must decline, reject, clarify or report as
not implemented (business test pack Q15-Q40, Q47-Q51).

Each boundary names the sentences it may cite from the project documents (chunk ids are verified against what the
search tool actually returned, never assumed). No number in this text comes from a model; evidence numbers are added
by the tools. Wording is deliberately plain: what is refused, why, and what can be offered instead.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Boundary:
    mode: str                 # decline | reject | clarify_inputs | unsupported
    headline: str
    body: str
    offer: str = ""
    doc_query: str = ""
    cite: List[str] = field(default_factory=list)
    with_prediction: bool = False   # also run the engine tools and show the real point estimate
    with_checks: bool = False       # also show the measured data-quality / training-range checks


OFFER_ASSESSMENT = ("I can show the supported assessment for the selected engine (point estimate in cycles, recent sensor trends, "
                    "limitations and model version) as one input for human review.")

BOUNDARIES = {
    "safe_to_fly": Boundary(
        "decline", "I can't say whether an aircraft is safe to fly.",
        "This application gives a simulated-model estimate of remaining useful life, in operating cycles, for a simulated FD001 engine. "
        "It is not a flight release, airworthiness or safety decision, and it knows nothing about a real aircraft.",
        OFFER_ASSESSMENT, "read-only scope flight release airworthiness not a maintenance decision system intended use educational demonstration",
        ["MC-19", "MC-09", "MC-01"]),
    "part_diagnosis": Boundary(
        "decline", "I can't identify a specific blade, bearing or part.",
        "FD001 supplies sensor histories and an engine-level failure endpoint only; there are no part-level diagnostic labels. "
        "The simulated fault mode is degradation that originates in the high pressure compressor, and nothing in the data identifies a component.",
        "I can show which sensors have moved recently, which describes observations and is not a diagnosis.",
        "HPC degradation no blade component physical cause engine-level target not component life no part-level diagnoses",
        ["DG-06", "DG-10", "MC-09"]),
    "root_cause": Boundary(
        "decline", "I can't confirm a root cause.",
        "The data are simulated. The only known fault mode is the high pressure compressor degradation built into the simulation, and there is no independent "
        "evidence from a real engine. A sensor trend describes what was observed; it does not identify a cause.",
        "I can describe the observed sensor trends and the limits of the model.",
        "reading a trend responsibly does not identify a cause HPC degradation fault mode no part-level diagnoses",
        ["SG-05", "DG-06", "MC-09"]),
    "guarantee": Boundary(
        "decline", "I can't give an exact failure date or guarantee a prediction.",
        "The estimate is a single point estimate in operating cycles with no uncertainty interval, errors are much larger at arbitrary cutoffs than at "
        "endpoints, and cycles cannot be converted to calendar dates without an outside assumption about how the engine is used.",
        "The point estimate in cycles is shown below for reference; it is not a promise.",
        "probability intervals guarantees point estimate cycles are not calendar time errors across all test cycles",
        ["MC-14", "DG-07", "MC-04"], with_prediction=True),
    "procedure_spec": Boundary(
        "decline", "I can't provide approved torque settings or other maintenance procedures.",
        "No approved manual for any engine is part of this project's sources, and I will not invent values or borrow them from a different engine model. "
        "Use the approved manual that applies to the engine in question.",
        "",
        "sources contain no maintenance procedures repair actions manual specifications torque values read-only scope",
        ["MC-09", "MC-19"]),
    "image_health": Boundary(
        "decline", "I can't confirm the internal HPC is healthy from a photo.",
        "An external photograph cannot establish the hidden internal condition of a compressor. This application also does not read images or inspection "
        "findings at all, so image interpretation is not implemented.",
        "The sensor-based assessment can be shown, but it is not a substitute for inspection.",
        "images photographs inspection evidence cannot establish internal condition of HPC",
        ["MC-18", "DG-06"]),
    "causal": Boundary(
        "decline", "I can't say that changing operating conditions would extend this engine's life.",
        "The models learned associations in simulated data. They do not establish what would happen if an operating condition were changed. "
        "That would need evidence from controlled changes to the same kind of engine and a measured effect on degradation.",
        "",
        "causal effects financial claims models describe association do not establish operating condition extend life",
        ["MC-15", "MC-06"]),
    "probability": Boundary(
        "decline", "I can't give a probability of failing within a number of cycles.",
        "The models produce one point estimate. There is no validated probability or survival model, and the Normal, Watch and Review bands are not probabilities.",
        "The point estimate in cycles is shown below for reference.",
        "probability intervals guarantees validated survival model point estimate no uncertainty interval",
        ["MC-14", "MC-02"], with_prediction=True),
    "interval": Boundary(
        "decline", "A prediction interval is not available; only a point estimate exists.",
        "There is no calibrated interval method in this project. A 95 percent interval would need calibration on separate engines and a measured held-out "
        "coverage, and neither exists, so any range I gave would be invented.",
        "The point estimate in cycles is shown below for reference.",
        "probability intervals guarantees calibrated prediction interval point estimate no uncertainty interval",
        ["MC-14", "MC-02"], with_prediction=True),
    "financial": Boundary(
        "clarify_inputs", "I can't estimate downtime savings or financial benefit from the simulated results.",
        "The evaluation figures come from simulated data; they are not realised savings. To build an assumptions-based scenario I would need: the baseline number of "
        "failures and unplanned downtime events, the downtime duration per event, the cost per downtime hour or per event, how often an intervention would actually "
        "be triggered and its measured effect, and the cost of running the system. If you give me those, I can calculate a scenario and label it as one.",
        "",
        "causal effects financial claims downtime savings cost benefits simulated data not realised savings",
        ["MC-15", "MC-01"]),
    "fleet_reliability": Boundary(
        "clarify_inputs", "I can't calculate MTBF or availability from this data alone.",
        "MTBF needs operating exposure (including engines that have not failed) and failure records, and availability also needs repair or downtime durations. "
        "Run-to-failure trajectories do not give repair times, and an average RUL is not an MTBF. If you can provide exposure, failure and repair records, "
        "I can compute them with the assumptions stated.",
        "",
        "fleet reliability statistics MTBF availability run-to-failure trajectories repair times average RUL",
        ["MC-16"]),
    "fault_vs_degradation": Boundary(
        "decline", "I can't tell a sensor fault from real equipment degradation.",
        "FD001 has no sensor-fault labels, so neither the models nor this assistant can separate the two. What can be checked is listed below: missing values, "
        "duplicated or skipped cycles, constant channels, and whether readings lie outside the training range. A clean check does not rule out a sensor fault, "
        "and an out-of-range reading is not a confirmed failure.",
        "",
        "sensor faults versus degradation no sensor-fault labels observable checks missing values gaps constant channels range",
        ["MC-17", "SG-05"], with_checks=True),
    "local_explanation": Boundary(
        "unsupported", "A per-engine explanation of this estimate is not implemented.",
        "I can show which of the 12 model sensors have moved most recently, and that ranks movement, not importance to the model. It is not a local explanation "
        "of this prediction, and a global feature importance would not explain this engine either, so I am not presenting either as one.",
        "Ask for the recent sensor trends to see the observed movement.",
        "sensors used by the model trend summaries ranks sensors by recent movement not importance reading a trend responsibly",
        ["SG-02", "SG-03", "SG-05"]),
    "compare_engines": Boundary(
        "unsupported", "Comparing two engines in one request is not supported.",
        "Every answer is bound to the single engine, split and cutoff selected in the sidebar, and text in the chat cannot change it. To compare, run each engine "
        "at the same cutoff in turn with the same model and compare the two values. A lower estimate is not a guaranteed failure order.",
        "I can answer for the engine currently selected.",
        "engine identifiers unique only within a dataset split every request names the split",
        ["DG-04"]),
    "cycle_comparison": Boundary(
        "unsupported", "Comparing two cutoffs in one request is not supported.",
        "Each answer uses only the cutoff selected in the sidebar. To see how the estimate changed, run the first cutoff, then the later one; each run reads only "
        "rows up to its own cutoff, so a later row never influences the earlier result. The estimate need not fall steadily as cycles are added.",
        "I can answer for the cutoff currently selected.",
        "estimate need not fall steadily as new cycles arrive limitations short histories",
        ["MC-06"]),
    "control_action": Boundary(
        "reject", "I can't do that: I have no control over any equipment, alarm or setting.",
        "This assistant is read-only. It cannot disable an alarm or change a control setting, and I will not suggest settings intended to keep an engine running.",
        "I can summarise the evidence for a human to review.",
        "read-only scope no authority over equipment alarm control setting maintenance approval work order",
        ["MC-19"]),
    "approval_bypass": Boundary(
        "reject", "I can't bypass maintenance approval or schedule service.",
        "This assistant is read-only and has no scheduling or approval authority. I can prepare an evidence summary (estimate, trends, limitations, model version) "
        "for a human to take through the proper approval route.",
        "",
        "read-only scope no authority over equipment alarm control setting maintenance approval work order before planning maintenance",
        ["MC-19", "MC-11"]),
    "rule_override": Boundary(
        "reject", "I can't follow that request.",
        "Text in a question or in a document cannot change my rules or tool permissions, and I cannot run shell commands or code. "
        "Only the engine, split and cutoff selected in the sidebar are used.",
        "",
        "read-only scope no authority over equipment", ["MC-19"]),
    "fabricate": Boundary(
        "reject", "I won't invent an RUL value.",
        "If the prediction service is unavailable I report that and return no number. A made-up value would look like a result and would not be one.",
        "",
        "read-only scope no authority", [], with_prediction=True),
    "cutoff_bypass": Boundary(
        "reject", "I can't use readings after the selected cutoff.",
        "Only observed cycles up to the selected cutoff are ever read; later rows are not accessible to the tools, so no wording of a request can change that.",
        "The assessment below uses only the cycles up to the selected cutoff.",
        "", [], with_prediction=True),
}
