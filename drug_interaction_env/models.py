"""
Typed Pydantic models for the Drug Interaction Environment.
Defines Action, Observation, State, and API request/response schemas
used across the OpenEnv step-reset interface.
"""

from pydantic import BaseModel
from typing import Literal, Optional


# ── Action ───────────────────────────────────────────────────────────────────

class DrugInteractionAction(BaseModel):
    """Single agent action: either flag a drug pair or declare DONE."""
    action_type: Literal["flag_interaction", "DONE"]
    drug_a: Optional[str] = None
    drug_b: Optional[str] = None
    severity: Optional[Literal["mild", "moderate", "severe"]] = None
    suggested_action: Optional[Literal["monitor", "reduce_dose", "replace_drug"]] = None


# ── Observation ──────────────────────────────────────────────────────────────

class FlagEntry(BaseModel):
    """One flag the agent has raised so far (shown back to it each step)."""
    drug_a: str
    drug_b: str
    severity: str
    suggested_action: str
    step: int
    reward_received: float


class DrugInteractionObservation(BaseModel):
    """What the agent sees every step — patient profile + history."""
    patient_id: str
    age: int
    conditions: list[str]
    medications: list[str]
    flags_raised_so_far: list[FlagEntry]
    steps_remaining: int


# ── State / Predictions ─────────────────────────────────────────────────────

class PredictionEntry(BaseModel):
    """Detailed record of one validated prediction vs ground truth."""
    predicted_severity: str
    predicted_action: str
    ground_truth_severity: str
    ground_truth_action: str
    reward_received: float
    perfectly_completed: bool


class DrugInteractionState(BaseModel):
    """Full internal state exposed via GET /state."""
    patient_id: str
    step_count: int
    task_level: str
    attempted_keys: list[str]
    identified_pairs: list[str]
    perfectly_completed_pairs: list[str]
    predictions: dict
    done: bool


# ── API Request / Response ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Body for POST /reset — pick a task difficulty."""
    task_level: Literal["easy", "medium", "hard"] = "easy"


class StepResponse(BaseModel):
    """Canonical response shape returned by POST /reset and POST /step."""
    observation: DrugInteractionObservation
    reward: float
    done: bool
    state: DrugInteractionState
