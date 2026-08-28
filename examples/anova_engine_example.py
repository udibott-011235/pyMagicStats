"""Minimal one-way inference-engine example; no post-hoc comparisons."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from pyMagicStat.assumptions import InferenceValidator, OneWayRobustness
from pyMagicStat.inference import MethodSelector, OneWayANOVA, WelchANOVA


rng = np.random.default_rng(20260827)
groups = (
    rng.normal(loc=100.0, scale=8.0, size=35),
    rng.normal(loc=105.0, scale=12.0, size=45),
    rng.normal(loc=110.0, scale=20.0, size=55),
)

validation = InferenceValidator().validate_one_way(
    *groups,
    independence="assumed",
)
decision = MethodSelector().select(validation.report)

print("Policy:", OneWayRobustness.POLICY_VERSION)
print("Decision:", decision.to_dict())

# Welch is the calibrated default for equal_var=None.
if decision.selected_method == "welch_anova":
    result = WelchANOVA(*groups, independence="assumed").run_test()
    print(result["method"], result["statistic"], result["p_value"])

# Classical is never selected automatically from a non-significant variance test.
# It must be requested explicitly and the joint variance diagnostics must support it.
classical_decision = MethodSelector().select(validation.report, equal_var=True)
if classical_decision.selected_method == "classical_anova":
    classical = OneWayANOVA(*groups, independence="assumed").run_test()
    print(classical["method"], classical["p_value"])

# A global rejection only states that at least one mean differs. This release
# intentionally does not implement Tukey, Games-Howell or other post-hoc tests.
