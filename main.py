import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Imports Qiskit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC
from tqdm.auto import tqdm

df = pd.read_csv("dataset_paper/dataset_paper/financial_train.csv")

#y (minuscule) = la cible (0 = faillite, 1 = saine)
y = df["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy()
#X (majuscule) = la matrice des 18 ratios financiers
X_df = df.drop(columns=["status_label", "fyear", "cik"])
X = X_df.to_numpy(dtype=float)



np.random.seed(123)

idx_0 = np.where(y == 0)[0]  # indices des faillites
idx_1 = np.where(y == 1)[0]  # indices des saines

n_samples = min(1000, len(idx_0), len(idx_1))

chosen_0 = np.random.choice(idx_0, size=n_samples, replace=False)
chosen_1 = np.random.choice(idx_1, size=n_samples, replace=False)

indices = np.concatenate([chosen_0, chosen_1])
np.random.shuffle(indices)

X_balanced = X[indices]
y_balanced = y[indices]

print(f"Jeu équilibré : {len(y_balanced)} entreprises au total ({n_samples} faillites, {n_samples} saines).")


pca = PCA(n_components=6)
X_5d = pca.fit_transform(X_balanced)

# Conversion des valeurs en angles (radians) pour les portes quantiques
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_5d)


train_features, test_features, train_labels, test_labels = train_test_split(
    X_scaled,
    y_balanced,
    train_size=0.8,
    random_state=123,
    stratify=y_balanced
)


algorithm_globals.random_seed = 123
n_qubits = 6

feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement="linear")
ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement="linear")

maxiter = 1000


def callback(weights, loss):
    progress.update(1)

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=maxiter),
    callback=callback
)

print("Entraînement du VQC en cours sur vos données...")

progress = tqdm(
    total=maxiter,
    desc="Entraînement VQC",
    bar_format="{desc}: {percentage:3.0f}% {bar} | Temps écoulé: {elapsed}, Temps restant: {remaining}",
)

try:
    vqc.fit(train_features, train_labels)
finally:
    progress.close()


score = vqc.score(test_features, test_labels)
print(f"Précision (Accuracy) sur le jeu de test : {score * 100:.2f} %")