from server.drug_database import DRUG_INTERACTIONS
from server.patients import PATIENTS
from itertools import combinations
import ast

TASK_TRUE_PAIRS = {
    "easy":   1,
    "medium": 3,
    "hard":   5
}

def grade_episode(task_level: str, final_state: dict) -> float:
    """
    Takes the final state dict from the environment at episode end.
    Computes normalized episode score in [0.0, 1.0].
    """
    n_true_pairs = TASK_TRUE_PAIRS[task_level]
    max_possible = n_true_pairs * 0.8

    predictions = final_state.get("predictions", {})
    identified_strs = set(final_state.get("identified_pairs", []))
    
    meds = PATIENTS[task_level]["medications"]
    ground_truth_keys = [tuple(sorted([a.lower(), b.lower()])) for a, b in combinations(meds, 2)]
    ground_truth_keys = [k for k in ground_truth_keys if k in DRUG_INTERACTIONS]
    ground_truth_strs = set(str(k) for k in ground_truth_keys)

    # Sum rewards from all predictions
    total_reward = sum(p["reward_received"] for p in predictions.values())

    # Apply severity-weighted termination penalties for unidentified pairs
    unidentified = ground_truth_strs - identified_strs
    for key_str in unidentified:
        # key_str is like "('drug_a', 'drug_b')"
        key = ast.literal_eval(key_str)
        gt = DRUG_INTERACTIONS.get(key, {})
        severity = gt.get("severity", "mild")
        if severity == "severe":
            total_reward -= 0.4
        elif severity == "moderate":
            total_reward -= 0.3
        else:
            total_reward -= 0.2

    # Normalize
    if max_possible == 0:
        return 1.0
    return max(0.0, min(1.0, total_reward / max_possible))

def verify_score_range(score: float) -> bool:
    return 0.0 <= score <= 1.0
