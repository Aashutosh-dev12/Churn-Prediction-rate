# Churn Prediction Model — Bug Summary

## Problem
The model **always predicted "Yes"** (churn) regardless of input values. Even for long-tenure, low-charge customers who clearly should not churn.

---

## Bugs Found

### Bug 1: Typo in `app.py`
| | |
|---|---|
| **File** | `app.py`, line 5 |
| **Wrong** | `import stramlit as st` |
| **Fixed** | `import streamlit as st` |
| **Impact** | App wouldn't start at all |

---

### Bug 2: Incorrect scaler usage in notebook
| | |
|---|---|
| **File** | `app.ipynb` (scaler cell) |
| **Wrong** | `X_test = scaler.fit_transform(X_test)` |
| **Fixed** | `X_test = scaler.transform(X_test)` |
| **Reason** | `fit_transform` re-fits the scaler on test data, which leaks test data info and gives wrong evaluation. You should **only** `transform` the test set using the scaler fitted on training data. |
| **Impact** | Test accuracy was inaccurate (misleading evaluation) |

---

### Bug 3: Class imbalance — the **main** bug  ⚠️
| | |
|---|---|
| **File** | `app.ipynb` (SVC model cell) |
| **Wrong** | `SVC(C=0.01, kernel='linear')` |
| **Fixed** | `SVC(C=0.01, kernel='linear', class_weight='balanced')` |
| **Impact** | Model always predicted "Yes" for every input |

**Why this happened:**

```
Dataset distribution:
  - Churn = Yes  →  ~76% (766 samples)
  - Churn = No   →  ~24% (234 samples)
```

Without `class_weight='balanced'`, the model learned that predicting **"Yes" every time** gives 76% accuracy — so it never bothered learning to distinguish the two classes.

**Proof — classification report of the old model:**
```
              precision    recall   support
  No  (0)       0.00      0.00        15
  Yes (1)       0.93      1.00       185
  Accuracy:     93%
```
- Recall for "No" = **0.00** → never predicted "No"
- 93% accuracy was **fake** — same as always printing "Yes"

**After fix with `class_weight='balanced'`:**
- Model now correctly predicts **both** "Yes" and "No"
- Overall accuracy dropped from 93% → ~69%, but the model is actually **useful** now

---

### Bug 4: Unused variable in `app.py`
| | |
|---|---|
| **File** | `app.py`, line 41 |
| **Code** | `X1 = np.array(X)` |
| **Issue** | Created but never used (dead code) |
| **Impact** | No functional impact, just unnecessary code |
| **Status** | Removed by user |

---

## Key Takeaway

> **High accuracy on imbalanced data ≠ good model.**
> Always check the **classification report** (precision, recall, f1 for each class).
> A model with 69% balanced accuracy is far more useful than one with 93% accuracy that only predicts one class.
