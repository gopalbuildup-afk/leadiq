## Module 1: Data Traps & Integrity Log

| Trap Category | Specific Issue Discovered | Evidence / Count in Raw Data | Resolution Applied |
| :--- | :--- | :--- | :--- |
| **Identity** | Duplicate enquiries & returning users | 121 duplicate person records | Grouped by unique identity; counts restricted to events strictly before $t$. |
| **Time** | Unaligned timezones across tables | Mixed UTC and naive timestamps | Enforced explicit `Asia/Kolkata` conversion on load in `contract.py`. |
| **Time** | Future event leakage after $t$ | Events present after `created_at + 24h` | Implemented point-in-time filter (`timestamp < t`) in `features.py`. |
| **Labels** | Immature leads (right-censored) | 907 leads created within 30 days of export | Excluded immature leads from training/eval sets in `contract.py`. |
| **Labels** | Post-facto leakage columns | 5 forbidden columns in raw CSV | Enforced strict validation in `features.py` raising `ValueError`. |

## Temporal Train / Validation Split Dates
* **Training Window:** [e.g., 2025-01-01 to 2025-10-31]
* **Validation Window:** [e.g., 2025-11-01 to 2025-12-15]
* **Calibration Window:** [e.g., 2025-12-16 to 2025-12-31]