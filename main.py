import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif  # Ajout pour remplacer la PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Imports Qiskit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import PauliFeatureMap
from tqdm.auto import tqdm

# Import Classique (Gradient Boosting)
from xgboost import XGBClassifier

# =============================================================
# 1. DONNÉES D'ENTRAÎNEMENT (df_train)
# =============================================================
df_train = pd.read_csv("dataset_paper/dataset_paper/financial_train.csv")

#y (minuscule) = la cible (0 = faillite, 1 = saine)
y = df["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy()
#X (majuscule) = la matrice des 18 ratios financiers
X_df = df.drop(columns=["status_label", "fyear", "cik"])
X = X_df.to_numpy(dtype=float)


np.random.seed(123)

idx_0 = np.where(y_train == 0)[0]
idx_1 = np.where(y_train == 1)[0]

n_samples = min(500, len(idx_0), len(idx_1))

chosen_0 = np.random.choice(idx_0, size=n_samples, replace=False)
chosen_1 = np.random.choice(idx_1, size=n_samples, replace=False)

indices = np.concatenate([chosen_0, chosen_1])
np.random.shuffle(indices)

X_train_balanced = X_train[indices]
y_train_balanced = y_train[indices]

print(f"Jeu d'entraînement équilibré : {len(y_train_balanced)} entreprises ({n_samples} faillites, {n_samples} saines).")

n_qubits = 8

# REMPLACEMENT DE LA PCA PAR LE SELECTKBEST (Garde votre nom de variable X_5d pour ne rien casser)
selector = SelectKBest(score_func=f_classif, k=n_qubits)
X_5d = selector.fit_transform(X_balanced, y_balanced)

# Optionnel : Affiche dans votre terminal quels ratios financiers ont été jugés les plus cruciaux
selected_features = X_df.columns[selector.get_support()]
print(f"Ratios financiers sélectionnés pour les qubits : {list(selected_features)}")

scaler = MinMaxScaler(feature_range=(0, np.pi))
train_features_vqc = scaler.fit_transform(X_train_pca)
train_labels = y_train_balanced

# =============================================================
# 2. DONNÉES DE TEST (df_test)
# =============================================================
df_test = pd.read_csv("dataset_paper/dataset_paper/financial_test.csv")

test_labels = df_test["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy()

X_test_df = df_test.drop(columns=[c for c in colonnes_a_retirer if c in df_test.columns])
X_test = X_test_df.to_numpy(dtype=float)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Projection PCA et normalisation pour le VQC
X_test_pca = pca.transform(X_test)
test_features_vqc = scaler.transform(X_test_pca)

print(f"Jeu de test prêt : {len(test_labels)} entreprises.\n")

# =============================================================
# 3. MODÈLE QUANTIQUE (VQC - 6 QUBITS)
# =============================================================
algorithm_globals.random_seed = 123

# Votre feature map Pauli configurée à reps=2
feature_map = PauliFeatureMap(feature_dimension=n_qubits, reps=2, paulis=['Z', 'ZZ'], entanglement="linear")
ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement="linear")

maxiter = 300


def callback(weights, loss):
    progress.update(1)

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=200)
)

print("Entraînement du VQC en cours sur vos données...")

progress = tqdm(
    total=maxiter,
    desc="Entraînement VQC",
    bar_format="{desc}: {percentage:3.0f}% {bar} | Temps écoulé: {elapsed}, Temps restant: ~{remaining}",
)

try:
    vqc.fit(train_features, train_labels)
finally:
    progress.close()


score = vqc.score(test_features, test_labels)
print(f"Précision (Accuracy) sur le jeu de test : {score * 100:.2f} %")