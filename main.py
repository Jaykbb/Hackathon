import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Imports Qiskit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC
from tqdm.auto import tqdm

# Import Classique (Gradient Boosting)
from xgboost import XGBClassifier

# =============================================================
# 1. DONNÉES D'ENTRAÎNEMENT (df_train)
# =============================================================
df_train = pd.read_csv("dataset_paper/dataset_paper/financial_train.csv")

y_train = df_train["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy()

colonnes_a_retirer = ["status_label", "fyear", "cik"]
X_train_df = df_train.drop(columns=[c for c in colonnes_a_retirer if c in df_train.columns])
X_train = X_train_df.to_numpy(dtype=float)
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

# Équilibrage des classes (200 de chaque pour l'entraînement)
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

# Compression PCA et normalisation [0, pi] pour le VQC
pca = PCA(n_components=6)
X_train_pca = pca.fit_transform(X_train_balanced)

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
n_qubits = 6

feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement="circular")
ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement="circular")

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=200)
)

print("--- 1. Entraînement du VQC (Quantique) ---")
vqc.fit(train_features_vqc, train_labels)

# =============================================================
# 4. MODÈLE CLASSIQUE (XGBOOST SUR LES 54 RATIOS)
# =============================================================
print("\n--- 2. Entraînement de XGBoost (Classique SOTA) ---")
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=123,
    eval_metric="logloss"
)

xgb.fit(X_train_balanced, y_train_balanced)


print("\n--- 3. Évaluation du modèle hybride combiné ---")

# A) VQC natif (utilise la règle de décision interne)
y_pred_vqc = vqc.predict(test_features_vqc)
proba_vqc = vqc.predict_proba(test_features_vqc)[:, 1]   # P(saine = 1)

# B) XGBoost
y_pred_xgb = xgb.predict(X_test)
proba_xgb = xgb.predict_proba(X_test)[:, 1]             # P(saine = 1)

# C) Vote hybride combiné
proba_hybride = (proba_vqc + proba_xgb) / 2.0
y_pred_hybride = (proba_hybride >= 0.5).astype(int)

# D) Calcul des scores
acc_vqc = accuracy_score(test_labels, y_pred_vqc)
acc_xgb = accuracy_score(test_labels, y_pred_xgb)
acc_hybride = accuracy_score(test_labels, y_pred_hybride)

print("\n=======================================================")
print("             RÉSULTATS DU VOTE HYBRIDE                 ")
print("=======================================================")
print(f"Accuracy VQC seul (Quantique natif): {acc_vqc * 100:.2f} %")
print(f"Accuracy XGBoost seul (Classique)  : {acc_xgb * 100:.2f} %")
print(f"Accuracy Modèle Hybride Combiné    : {acc_hybride * 100:.2f} %")

print("\n--- MATRICE DE CONFUSION (MODÈLE HYBRIDE) ---")
print(confusion_matrix(test_labels, y_pred_hybride))

print("\n--- RAPPORT DÉTAILLÉ (MODÈLE HYBRIDE) ---")
print(classification_report(test_labels, y_pred_hybride, target_names=["Faillite (0)", "Saine (1)"]))
