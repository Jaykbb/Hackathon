import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# Imports Qiskit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC


df_train = pd.read_csv("dataset_paper/dataset_paper/financial_train.csv")

# Cible y_train (0 = faillite, 1 = saine)
y_train = df_train["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy()

# Matrice des ratios financiers X_train
X_train_df = df_train.drop(columns=["status_label", "fyear", "cik"])
X_train = X_train_df.to_numpy(dtype=float)


# Équilibrage des classes (200 de chaque pour le train)
np.random.seed(123)

idx_0 = np.where(y_train == 0)[0]
idx_1 = np.where(y_train == 1)[0]

n_samples = min(1000, len(idx_0), len(idx_1))

chosen_0 = np.random.choice(idx_0, size=n_samples, replace=False)
chosen_1 = np.random.choice(idx_1, size=n_samples, replace=False)

indices = np.concatenate([chosen_0, chosen_1])
np.random.shuffle(indices)

X_balanced = X_train[indices]
y_balanced = y_train[indices]

print(f"Jeu d'entraînement équilibré : {len(y_balanced)} entreprises ({n_samples} faillites, {n_samples} saines).")

# Compression PCA 
pca = PCA(n_components=6)
X_train_pca = pca.fit_transform(X_balanced)

# Normalisation angulaire [0, pi]
scaler = MinMaxScaler(feature_range=(0, np.pi))
train_features = scaler.fit_transform(X_train_pca)
train_labels = y_balanced


df_test = pd.read_csv("dataset_paper/dataset_paper/financial_test.csv")

# Cible y_test
test_labels = df_test["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy()

# Matrice X_test
X_test_df = df_test.drop(columns=["status_label", "fyear", "cik"])
X_test = X_test_df.to_numpy(dtype=float)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Projection PCA et normalisation avec les estimateurs déjà ajustés
X_test_pca = pca.transform(X_test)
test_features = scaler.transform(X_test_pca)

print(f"Jeu de test prêt : {len(test_labels)} entreprises.")


algorithm_globals.random_seed = 123
n_qubits = 6

feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement="linear")
ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement="linear")

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=200)
)

print("Entraînement du VQC en cours...")
vqc.fit(train_features, train_labels)


score = vqc.score(test_features, test_labels)
print(f"Précision (Accuracy) sur le jeu de test : {score * 100:.2f} %")

# Prédictions sur le jeu de test
y_pred = vqc.predict(test_features)

print("Matrice de confusion :")
print(confusion_matrix(test_labels, y_pred))

print("\nRapport détaillé :")
print(classification_report(test_labels, y_pred, target_names=["Faillite (0)", "Saine (1)"]))