import time
import os
import logging
from sklearn.calibration import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib
import gc
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Limitar o número de threads para evitar sobrecarga de CPU
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desativa oneDNN para evitar erros no TensorFlow

def optimize_directories(folds_dir="../folds/", predictions_dir="../predictions/", logs_dir="../logs/"):
    os.makedirs(folds_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return folds_dir, predictions_dir, logs_dir

def optimize_logging(logs_dir, log_file_name):
    log_file = os.path.join(logs_dir, log_file_name)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    return log_file

def meta_model_template(
    X_train_resampled,
    y_train_resampled,
    data_test,
    n_splits_n=3,
    num_classes_n=8,
    verbose=True,
    model_function=None,
    model_name="model",
    model_params=None
):
    if model_function is None or model_params is None:
        raise ValueError("É necessário fornecer uma função de modelo e seus parâmetros.")

    start_time = time.time()
    folds_dir, predictions_dir, logs_dir = optimize_directories()
    log_file = optimize_logging(logs_dir, f"training_log_{model_name}.txt")

    logging.info(f"Início do treinamento do {model_name}.")

    kf = StratifiedKFold(n_splits=n_splits_n, shuffle=True, random_state=42)
    oof_predictions = np.zeros((X_train_resampled.shape[0], num_classes_n))
    test_predictions = np.zeros((data_test.shape[0], num_classes_n))

    f1_scores, validation_scores = [], []
    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_resampled, y_train_resampled)):
        fold_start = time.time()
        logging.info(f"Training Fold {fold + 1}...")

        X_train_fold = X_train_resampled.iloc[train_idx]
        X_valid_fold = X_train_resampled.iloc[valid_idx]
        y_train_fold = y_train_resampled.iloc[train_idx]
        y_valid_fold = y_train_resampled.iloc[valid_idx]

        fold_model = model_function(**model_params)
        fold_model.fit(X_train_fold, y_train_fold)

        valid_proba = fold_model.predict_proba(X_valid_fold)
        valid_preds = np.argmax(valid_proba, axis=1)

        # Métrica de F1-Score
        fold_f1 = f1_score(y_valid_fold, valid_preds, average='macro')
        f1_scores.append(fold_f1)

        # Métrica de Accuracy
        fold_val_score = accuracy_score(y_valid_fold, valid_preds)
        validation_scores.append(fold_val_score)

        oof_predictions[valid_idx] = valid_proba

        # Previsões em lote para economizar memória
        test_batch_preds = []
        batch_size = 1000
        for i in range(0, len(data_test), batch_size):
            test_batch_preds.append(fold_model.predict_proba(data_test.iloc[i:i + batch_size]))
        test_predictions += np.vstack(test_batch_preds)

        fold_time = time.time() - fold_start
        fold_results.append({
            "Fold": fold + 1,
            "F1-Score": fold_f1,
            "Validation Score": fold_val_score,
            "Time (s)": fold_time
        })

        logging.info(f"Fold {fold + 1} concluído em {fold_time:.2f}s - F1: {fold_f1:.4f}, Acc: {fold_val_score:.4f}")
        joblib.dump(fold_model, os.path.join(folds_dir, f"{model_name}_fold_{fold + 1}.joblib"))

        del X_train_fold, X_valid_fold, y_train_fold, y_valid_fold, fold_model
        gc.collect()

    test_predictions /= n_splits_n
    np.save(os.path.join(predictions_dir, f"oof_predictions_{model_name}.npy"), oof_predictions)
    np.save(os.path.join(predictions_dir, f"test_predictions_{model_name}.npy"), test_predictions)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(logs_dir, f"{model_name}_fold_results.csv"), index=False)

    logging.info(f"Treinamento concluído. F1-Score Médio: {np.mean(f1_scores):.4f}")
    print(f"F1-Score Médio: {np.mean(f1_scores):.4f}")

    return f1_scores, validation_scores, oof_predictions, test_predictions



def meta_model_rf(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    return meta_model_template(
        X_train_resampled,
        y_train_resampled,
        data_test,
        n_splits_n,
        num_classes_n,
        verbose,
        model_function=RandomForestClassifier,
        model_name="rfc",
        model_params = {
            "n_estimators": 50,          # Número de árvores maior para melhorar a estabilidade (valor típico: 50-300)
            "max_depth": 20,              # Profundidade máxima controlada para evitar overfitting
            "min_samples_split": 5,       # Evita nós muito pequenos (mais robusto)
            "min_samples_leaf": 2,        # Mínimo de amostras por folha para evitar overfitting
            "class_weight": "balanced",   # Lida com classes desbalanceadas ajustando pesos automaticamente
            "max_features": "sqrt",       # Reduz o número de features testadas em cada divisão (melhora eficiência)
            "random_state": 42           # Reprodutibilidade
        }
    )


def meta_model_et(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    return meta_model_template(
        X_train_resampled,
        y_train_resampled,
        data_test,
        n_splits_n,
        num_classes_n,
        verbose,
        model_function=ExtraTreesClassifier,
        model_name="et",
        model_params = {
            "n_estimators": 50,             # Aumenta o número de árvores para maior estabilidade
            "max_depth": None,               # Sem limite para permitir árvores completas (ou ajuste conforme necessário)
            "min_samples_split": 2,          # Padrão; controle a divisão de nós
            "min_samples_leaf": 1,           # Padrão; mínimo de amostras em cada folha
            "max_features": "sqrt",          # Melhora a eficiência, especialmente com muitas features
            "bootstrap": False,              # Extra Trees usa por padrão amostragem completa (sem bootstrap)
            "class_weight": "balanced",      # Ajusta os pesos para lidar com desbalanceamento
            "random_state": 42              # Reprodutibilidade
        }
    )


def meta_model_lr(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    return meta_model_template(
        X_train_resampled,
        y_train_resampled,
        data_test,
        n_splits_n,
        num_classes_n,
        verbose,
        model_function=LogisticRegression,
        model_name="lr",
        model_params={
            "random_state": 42,
            "solver": "saga",              # Mais eficiente e permite L1/L2
            "multi_class": "multinomial",  # Para problemas multiclasse
            "penalty": "l2",               # Regularização padrão (ou "l1" para sparsidade)
            "C": 1.0,                      # Regularização: teste 0.01, 0.1, 1, 10
            "max_iter": 200,               # Garantir convergência
            "tol": 1e-4                    # Tolerância para a convergência
        }
    )

def meta_model_gb(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    return meta_model_template(
        X_train_resampled,
        y_train_resampled,
        data_test,
        n_splits_n,
        num_classes_n,
        verbose,
        model_function=GradientBoostingClassifier,
        model_name="gb",
        model_params={
            "random_state": 42,  # Para reprodutibilidade
            "n_estimators": 100,  # Número de árvores no ensemble
            "learning_rate": 0.2,  # Taxa de aprendizado
            "max_depth": 3,  # Profundidade máxima de cada árvore
        }
    )

def meta_model_adaboost(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    return meta_model_template(
        X_train_resampled,
        y_train_resampled,
        data_test,
        n_splits_n,
        num_classes_n,
        verbose,
        model_function=AdaBoostClassifier,
        model_name="adaboost",
        model_params={
            "n_estimators": 25,  # Número de estimadores (árvores no ensemble)
            "learning_rate": 0.5,  # Taxa de aprendizado (impacto de cada árvore)
            "algorithm": "SAMME.R",  # Algoritmo padrão para problemas multiclasse
            "random_state": 42  # Reprodutibilidade
        }
    )


def meta_model_gnb(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    return meta_model_template(
        X_train_resampled,
        y_train_resampled,
        data_test,
        n_splits_n,
        num_classes_n,
        verbose,
        model_function=GaussianNB,
        model_name="gnb",
        model_params={
            "var_smoothing": 1e-9  # Fator de suavização para evitar variâncias zero
        }
    )

def meta_model_cb(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    return meta_model_template(
        X_train_resampled,
        y_train_resampled,
        data_test,
        n_splits_n,
        num_classes_n,
        verbose,
        model_function=CatBoostClassifier,
        model_name="catboost",
        model_params={
            "iterations": 800,  # Reduzido para economizar recursos
            "depth": 6,  # Profundidade padrão para evitar overfitting
            "learning_rate": 0.3,  # Taxa de aprendizado padrão
            "random_seed": 42,  # Para reprodutibilidade
            "auto_class_weights": "Balanced",  # Balanceamento automático de classes
            "verbose": verbose,  # Controle de log durante o treinamento
            "task_type": "CPU"  # Garantindo que o CatBoost use CPU
        }
    )