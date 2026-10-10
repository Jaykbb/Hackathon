# We will win that shit
import numpy as np
import pandas as pd

def dataframe(x):
    df = pd.read_csv(f"{x}")
    y = df["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy()
    print(df.head())

dataframe("american_bankruptcy_dataset.csv")