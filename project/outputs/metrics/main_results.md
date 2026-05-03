# Main Results

| Method | Accuracy | Macro-F1 | F1-Fake | F1-True | JSON Parse Rate |
|---|---:|---:|---:|---:|---:|
| BasePack only | 0.8605 | 0.8494 | 0.8085 | 0.8902 | — |
| Heuristic Pre | 0.8798 | 0.8681 | 0.8287 | 0.9075 | 0.9660 |
| SFT Hint Pre | 0.8992 | 0.8922 | 0.8646 | 0.9198 | 0.9859 |
| DPO Hint Pre | 0.8837 | 0.8750 | 0.8421 | 0.9080 | 0.9830 |

# Key Comparisons (pp = percentage points)

| Comparison | ΔAcc (pp) | ΔMacro-F1 (pp) | ΔF1-Fake (pp) |
|---|---:|---:|---:|
| Heuristic - BasePack | +1.93 | +1.87 | +2.02 |
| SFT Hint - Heuristic | +1.94 | +2.41 | +3.59 |
| DPO Hint - SFT Hint | -1.55 | -1.72 | -2.25 |
| DPO Hint - Heuristic | +0.39 | +0.69 | +1.34 |

# Notes

- DPO > SFT > Heuristic: preference training effective.
- SFT > Heuristic but DPO flat: check DPO reward design.
- Heuristic best: check SFT template quality and scoring reward.
