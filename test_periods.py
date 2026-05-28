import pandas as pd
import numpy as np
import os

DATA_DIR = "./data"
campaigns = pd.read_csv(os.path.join(DATA_DIR, "campaign_data.csv"))
transactions = pd.read_csv(os.path.join(DATA_DIR, "customer_transaction_data.csv"))
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

# Filter to only train campaigns
train_campaigns = train["campaign_id"].unique()
campaigns = campaigns[campaigns["campaign_id"].isin(train_campaigns)].copy()

campaigns["start_date"] = pd.to_datetime(campaigns["start_date"], format="%d/%m/%y").dt.normalize()
campaigns["end_date"] = pd.to_datetime(campaigns["end_date"], format="%d/%m/%y").dt.normalize()

boundaries = pd.concat([
    campaigns["start_date"],
    campaigns["end_date"] + pd.Timedelta(days=1),
]).dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)

periods = []
for i in range(len(boundaries) - 1):
    s = boundaries.iloc[i]
    e = boundaries.iloc[i + 1] - pd.Timedelta(days=1)
    if e < s: continue
    periods.append({"start": s, "end": e})

print("Artificial periods from campaigns:", len(periods))

transactions["date"] = pd.to_datetime(transactions["date"])
min_tx = transactions["date"].min()
min_camp = campaigns["start_date"].min()
print("Min transaction date:", min_tx)
print("Min campaign start:", min_camp)

