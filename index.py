import pandas as pd

# Load raw datasets
leads = pd.read_csv("data/leads.csv")
messages = pd.read_csv("data/messages.csv")
calls = pd.read_csv("data/calls.csv")

print("=== DATA TRAP AUDIT REPORT ===")

# Trap 1: Identity & Duplicates
if "phone" in leads.columns or "email" in leads.columns:
    id_col = "phone" if "phone" in leads.columns else "email"
    dup_counts = leads[id_col].duplicated().sum()
    print(f"[Trap 1 - Identity] Duplicate person entries found: {dup_counts}")

# Trap 2 & 3: Telemetry Anomalies
negative_calls = (calls["duration_sec"] < 0).sum() if "duration_sec" in calls.columns else 0
print(f"[Trap 2 - Telemetry] Calls with negative duration: {negative_calls}")

# Trap 4: Immature Leads
leads["created_at"] = pd.to_datetime(leads["created_at"])
max_date = leads["created_at"].max()
immature_cutoff = max_date - pd.Timedelta(days=30)
immature_leads = (leads["created_at"] > immature_cutoff).sum()
print(f"[Trap 5 - Labels] Immature leads (< 30 days old before export): {immature_leads}")

# Trap 5: Forbidden Columns Present
forbidden = ["current_stage", "legacy_score", "legacy_score_version", "updated_at", "last_contacted_at"]
found_forbidden = [c for c in forbidden if c in leads.columns]
print(f"[Trap 6 - Leakage] Forbidden columns present in raw CSV: {found_forbidden}")