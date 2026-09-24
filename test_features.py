import pandas as pd
from leadiq.contract import (
    load_and_validate_leads,
    load_and_validate_messages,
    load_and_validate_calls,
    load_and_validate_localities,
)
from leadiq.features import build_feature_matrix, FORBIDDEN_COLUMNS

leads = load_and_validate_leads("data/leads.csv")
msgs = load_and_validate_messages("data/messages.csv")
calls = load_and_validate_calls("data/calls.csv")
locs = load_and_validate_localities("data/localities.csv")

clean_leads = leads.drop(
    columns=[
        c for c in FORBIDDEN_COLUMNS
        if c in leads.columns
    ]
)

X = build_feature_matrix(
    clean_leads,
    msgs,
    calls,
    locs,
)

print("Dataset Shape:", X.shape)

print("\nTop 5 preprocessed rows:")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(X.head(5))

print("\nTop 5 rows transposed:")
print(X.head(5).T)