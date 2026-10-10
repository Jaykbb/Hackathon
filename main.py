import numpy as np
import pandas as pd
from qiskit.circuit.library import n_local, zz_feature_map
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.datasets import ad_hoc_data

def dataframe(x):
    df = pd.read_csv(f"{x}")
    x = df["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy() #si compagnie est alive =1 si fail = 0
    y = df.drop(columns = ["status_label", "fyear", "company_name","Division","MajorGroup"]) # Enlever les colonnes inutile
    y = y.to_numpy(dtype=float) #Mettre les 18 caractéristique en array numpy
    

dataframe("american_bankruptcy_dataset.csv")

def VQC(x):
    seed = 1376
algorithm_globals.random_seed = seed

# Use ad hoc data set for training and test data
feature_dim = 2  # dimension of each data point
training_size = 20
test_size = 10

# training features, training labels, test features, test labels as np.ndarray,
# one hot encoding for labels
training_features, training_labels, test_features, test_labels = ad_hoc_data(
    training_size=training_size, test_size=test_size, n=feature_dim, gap=0.3
)

feature_map = zz_feature_map(feature_dimension=feature_dim, reps=2, entanglement="linear")
ansatz = n_local(feature_map.num_qubits, ["ry", "rz"], "cz", reps=3)
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=100),
)
vqc.fit(training_features, training_labels)

score = vqc.score(test_features, test_labels)
print(f"Testing accuracy: {score:0.2f}")