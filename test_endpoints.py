"""Quick endpoint smoke-test for the Drug Interaction Environment server."""
import requests, json, sys

BASE = "http://localhost:8000"

def pp(label, resp):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

try:
    # 1. POST /reset
    r1 = requests.post(f"{BASE}/reset", json={"task_level": "easy"})
    pp("TEST 1: POST /reset  (task_level=easy)", r1)
    assert r1.status_code == 200, f"Expected 200, got {r1.status_code}"
    body = r1.json()
    assert body["done"] == False
    assert body["reward"] == 0.0
    assert body["observation"]["patient_id"] == "P001"
    print("  ✓ PASS")

    # 2. GET /state
    r2 = requests.get(f"{BASE}/state")
    pp("TEST 2: GET /state", r2)
    assert r2.status_code == 200
    state = r2.json()
    assert state["task_level"] == "easy"
    assert state["step_count"] == 0
    print("  ✓ PASS")

    # 3. POST /step
    r3 = requests.post(f"{BASE}/step", json={
        "action_type": "flag_interaction",
        "drug_a": "warfarin",
        "drug_b": "aspirin",
        "severity": "severe",
        "suggested_action": "replace_drug",
    })
    pp("TEST 3: POST /step  (flag_interaction)", r3)
    assert r3.status_code == 200
    step_body = r3.json()
    assert step_body["reward"] > 0
    assert len(step_body["observation"]["flags_raised_so_far"]) == 1
    print("  ✓ PASS")

    # 4. GET /health (bonus)
    r4 = requests.get(f"{BASE}/health")
    pp("TEST 4: GET /health", r4)
    assert r4.status_code == 200
    print("  ✓ PASS")

    print("\n" + "="*60)
    print("  ALL ENDPOINT TESTS PASSED ✓")
    print("="*60)

except Exception as e:
    print(f"\n  ✗ FAIL: {e}")
    sys.exit(1)
