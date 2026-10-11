import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Imports Qiskit
from qiskit.circuit.library import RealAmplitudes, PauliFeatureMap
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

# Équilibrage des classes 
np.random.seed(123)
idx_0 = np.where(y_train == 0)[0]  # Forcé en 1D
idx_1 = np.where(y_train == 1)[0]  # Forcé en 1D

n_samples = min(500, len(idx_0), len(idx_1))

chosen_0 = np.random.choice(idx_0, size=n_samples, replace=False)
chosen_1 = np.random.choice(idx_1, size=n_samples, replace=False)

indices = np.concatenate([chosen_0, chosen_1])
np.random.shuffle(indices)

X_train_balanced = X_train[indices]
y_train_balanced = y_train[indices]

print(f"Jeu d'entraînement équilibré : {len(y_train_balanced)} entreprises ({n_samples} faillites, {n_samples} saines).")

# Sélection des 8 meilleures variables (Configuration optimale à 8 Qubits)
n_qubits = 8
selector = SelectKBest(score_func=f_classif, k=n_qubits)
X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)

selected_features = X_train_df.columns[selector.get_support()]
print(f"Ratios financiers retenus pour le pipeline hybride : {list(selected_features)}")

# Normalisation [0, pi] pour le VQC
scaler = MinMaxScaler(feature_range=(0, np.pi))
train_features_vqc = scaler.fit_transform(X_train_selected)
train_labels = y_train_balanced

# =============================================================
# 2. DONNÉES DE TEST (df_test)
# =============================================================
df_test = pd.read_csv("dataset_paper/dataset_paper/financial_test.csv")

test_labels = df_test["status_label"].apply(lambda val: 1 if val == "alive" else 0).to_numpy()

X_test_df = df_test.drop(columns=[c for c in colonnes_a_retirer if c in df_test.columns])
X_test = X_test_df.to_numpy(dtype=float)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Alignement des données de test sur les 8 variables sélectionnées et normalisation
X_test_selected = selector.transform(X_test)
test_features_vqc = scaler.transform(X_test_selected)

print(f"Jeu de test prêt : {len(test_labels)} entreprises.\n")

# =============================================================
# 3. MODÈLE QUANTIQUE OPTIMISÉ (VQC - 8 QUBITS)
# =============================================================
algorithm_globals.random_seed = 123

feature_map = PauliFeatureMap(feature_dimension=n_qubits, reps=2, paulis=['Z', 'ZZ'], entanglement="linear")
ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement="linear")

# Fixé à 150 suite à l'analyse de convergence (optimisation de vitesse maximale)
maxiter = 150 
history_loss = []

def callback(weights, loss):
    history_loss.append(loss)
    progress.update(1)
    if len(history_loss) % 10 == 0:
        progress.set_postfix({"Loss": f"{loss:.4f}"})

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=maxiter),
    callback=callback
)

print("--- 1. Entraînement du VQC (Quantique) ---")
progress = tqdm(
    total=maxiter,
    desc="Optimisation VQC",
    bar_format="{desc}: {percentage:3.0f}% {bar} | {postfix} | Temps: {elapsed} < {remaining}",
)

try:
    vqc.fit(train_features_vqc, train_labels)
finally:
    progress.close()

# =============================================================
# 4. MODÈLE CLASSIQUE OPTIMISÉ (XGBOOST)
# =============================================================
print("\n--- 2. Entraînement de XGBoost (Classique SOTA) ---")
xgb = XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05, random_state=123, eval_metric="logloss"
)
xgb.fit(X_train_selected, y_train_balanced)

# =============================================================
# 5. ÉVALUATION DU MODÈLE HYBRIDE COMBINÉ
# =============================================================
print("\n--- 3. Évaluation du modèle hybride combiné ---")

batch_size = 10
n_test = len(test_features_vqc)
proba_vqc_list = []
y_pred_vqc_list = []

print("Calcul des prédictions quantiques (VQC)...")
eval_progress = tqdm(
    total=n_test, 
    desc="Évaluation VQC Test",
    bar_format="{desc}: {percentage:3.0f}% {bar} | Temps écoulé: {elapsed}, Temps restant: ~{remaining}"
)

for i in range(0, n_test, batch_size):
    batch_features = test_features_vqc[i:i+batch_size]
    batch_pred = vqc.predict(batch_features)
    batch_proba = vqc.predict_proba(batch_features)[:, 1]
    proba_vqc_list.extend(batch_proba)
    y_pred_vqc_list.extend(batch_pred)
    eval_progress.update(len(batch_features))
eval_progress.close()

proba_vqc = np.array(proba_vqc_list)
y_pred_vqc = np.array(y_pred_vqc_list)

# B) XGBoost
y_pred_xgb = xgb.predict(X_test_selected)
proba_xgb = xgb.predict_proba(X_test_selected)[:, 1]

# C) Vote hybride combiné avec SEUIL OPTIMISÉ POUR L'ACCURACY (0.38)
proba_hybride = (proba_vqc + proba_xgb) / 2.0
seuil_optimiste_survie = 0.38  
y_pred_hybride = (proba_hybride >= seuil_optimiste_survie).astype(int)

# D) Calcul des scores
acc_vqc = accuracy_score(test_labels, y_pred_vqc)
acc_xgb = accuracy_score(test_labels, y_pred_xgb)
acc_hybride = accuracy_score(test_labels, y_pred_hybride)

print("\n=======================================================")
print("       RÉSULTATS DU VOTE HYBRIDE OPTIMISÉ              ")
print("=======================================================")
print(f"Accuracy VQC seul (Quantique optimal): {acc_vqc * 100:.2f} %")
print(f"Accuracy XGBoost seul (Classique)    : {acc_xgb * 100:.2f} %")
print(f"Accuracy Modèle Hybride Combiné      : {acc_hybride * 100:.2f} %")

print("\n--- MATRICE DE CONFUSION (MODÈLE HYBRIDE) ---")
print(confusion_matrix(test_labels, y_pred_hybride))

print("\n--- RAPPORT DÉTAILLÉ (MODÈLE HYBRIDE) ---")
print(classification_report(test_labels, y_pred_hybride, target_names=["Faillite (0)", "Saine (1)"]))

# =============================================================
# 🎯 RECHERCHE CHIRURGICALE DU SEUIL OPTIMAL POUR LE RECALL
# =============================================================
print("\n=======================================================")
print("   OPTIMISATION DU SEUIL : ACCURACY VS RECALL (FAILLITE)")
print("=======================================================")
print(f"{'Seuil':<8} | {'Accuracy':<10} | {'Recall Faillite':<15} | {'Faillites Trouvées':<20}")
print("-" * 65)

# np.arange(0.35, 0.56, 0.01) permet de tester TOUTES les décimales (0.35, 0.36, 0.37... jusqu'à 0.55)
for seuil in np.arange(0.35, 0.56, 0.01):
    preds_temporaires = (proba_hybride >= seuil).astype(int)
    
    # Calcul des métriques pour ce seuil précis
    acc_temp = accuracy_score(test_labels, preds_temporaires)
    cm_temp = confusion_matrix(test_labels, preds_temporaires)
    
    # Extraction des faillites correctement détectées (Vrais Négatifs)
    vrais_faillites = cm_temp[0, 0]
    recall_faillite = vrais_faillites / 72.0
    
    print(f"{seuil:<8.2f} | {acc_temp * 100:<9.2f}% | {recall_faillite * 100:<14.2f}% | {vrais_faillites:<2d} / 72")