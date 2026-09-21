import pandas as pd
from leadiq.features import build_feature_matrix, FORBIDDEN_COLUMNS

leads = pd.read_csv("data/leads.csv")
msgs = pd.read_csv("data/messages.csv")
calls = pd.read_csv("data/calls.csv")
locs = pd.read_csv("data/localities.csv")

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