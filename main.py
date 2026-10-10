import numpy as np
import pandas as pd

def dataframe(x):
    df = pd.read_csv(f"{x}")
    x = df["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy() #si compagnie est alive =1 si fail = 0
    y = df.drop(columns = ["status_label", "fyear", "company_name","Division","MajorGroup"]) # Enlever les colonnes inutile
    y = y.to_numpy(dtype=float) #Mettre les 18 caractéristique en array numpy
    print()

dataframe("american_bankruptcy_dataset.csv")