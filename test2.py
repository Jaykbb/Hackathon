import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Imports Qiskit
from qiskit.circuit.library import PauliFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# 1. Données
df = pd.read_csv("dataset_paper/dataset_paper/financial_train.csv")
y = df["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy()
X_df = df.drop(columns=["status_label", "fyear", "cik"])
X = X_df.to_numpy(dtype=float)

# 2. Échantillonnage (400 samples)
np.random.seed(123)
idx_0 = np.where(y == 0)[0]
idx_1 = np.where(y == 1)[0]
n_samples = min(400, len(idx_0), len(idx_1)) 

indices = np.concatenate([np.random.choice(idx_0, size=n_samples, replace=False),
                          np.random.choice(idx_1, size=n_samples, replace=False)])
np.random.shuffle(indices)

X_balanced = X[indices]
y_balanced = y[indices]

# 3. PCA 10 composantes
n_qubits = 10
pca = PCA(n_components=n_qubits)
X_pca = pca.fit_transform(X_balanced)

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_pca)

train_features, test_features, train_labels, test_labels = train_test_split(
    X_scaled, y_balanced, train_size=0.8, random_state=123, stratify=y_balanced
)

algorithm_globals.random_seed = 123

# 4. Modèle Pauli & Kernel (Version standard sans boucles manuelles)
feature_map = PauliFeatureMap(feature_dimension=n_qubits, reps=1, paulis=['Z', 'ZZ'], entanglement="linear")
q_kernel = FidelityQuantumKernel(feature_map=feature_map)

qsvc = QSVC(quantum_kernel=q_kernel)

print("Calcul du Kernel Quantique et Entraînement en cours (sans boucle manuelle)...")
# On passe directement les features et NON la matrice précalculée !
qsvc.fit(train_features, train_labels)

# 5. Évaluation
score = qsvc.score(test_features, test_labels)
print(f"\nPrécision globale (Accuracy) corrigée : {score * 100:.2f} %")

predictions = qsvc.predict(test_features)
print("\nRapport détaillé :")
print(classification_report(test_labels, predictions, target_names=["Faillite", "Saine"]))