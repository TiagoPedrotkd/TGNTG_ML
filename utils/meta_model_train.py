import time
import os
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib
import gc
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier




def optimize_directories(folds_dir="../folds/", predictions_dir="../predictions/", logs_dir="../logs/"):
    """Cria diretórios necessários se ainda não existirem."""
    os.makedirs(folds_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return folds_dir, predictions_dir, logs_dir


def optimize_logging(logs_dir, log_file_name):
    """Configura logging para registrar eventos importantes."""
    log_file = os.path.join(logs_dir, log_file_name)
    logging.basicConfig(
        filename=log_file,
        level=logging.WARNING,  # Reduz logs para evitar excesso de informações
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
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

    models = []
    f1_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_resampled, y_train_resampled)):
        if verbose:
            print(f"Training Fold {fold + 1}...")
            logging.info(f"Training Fold {fold + 1}...")

        X_train_fold = X_train_resampled.iloc[train_idx]
        X_valid_fold = X_train_resampled.iloc[valid_idx]
        y_train_fold = y_train_resampled.iloc[train_idx]
        y_valid_fold = y_train_resampled.iloc[valid_idx]

        logging.info(f"Shapes - X_train_fold: {X_train_fold.shape}, X_valid_fold: {X_valid_fold.shape}")

        fold_model = model_function(**model_params)
        fold_model.fit(X_train_fold, y_train_fold)

        oof_predictions[valid_idx] = fold_model.predict_proba(X_valid_fold)
        test_predictions += fold_model.predict_proba(data_test)

        valid_preds = np.argmax(oof_predictions[valid_idx], axis=1)
        fold_f1 = f1_score(y_valid_fold, valid_preds, average='macro')
        f1_scores.append(fold_f1)

        if verbose:
            print(f"Macro F1-Score do Fold {fold + 1}: {fold_f1:.4f}")
            logging.info(f"Macro F1-Score do Fold {fold + 1}: {fold_f1:.4f}")

        model_filename = os.path.join(folds_dir, f"{model_name}_fold_{fold + 1}.joblib")
        joblib.dump(fold_model, model_filename)

        del X_train_fold, X_valid_fold, y_train_fold, y_valid_fold, fold_model
        gc.collect()

    test_predictions /= n_splits_n

    np.save(os.path.join(predictions_dir, f"oof_predictions_{model_name}.npy"), oof_predictions)
    np.save(os.path.join(predictions_dir, f"test_predictions_{model_name}.npy"), test_predictions)

    end_time = time.time()
    total_time = end_time - start_time
    if verbose:
        print(f"Treinamento concluído em {total_time / 60:.2f} minutos.")
        logging.info(f"Treinamento concluído em {total_time / 60:.2f} minutos.")

    return models, f1_scores, oof_predictions, test_predictions


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
        model_params={
            "n_estimators": 100,  # Reduzido para economizar recursos
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 2  # Reduzido para minimizar uso de CPU
        }
    )


def meta_model_xgbc(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    return meta_model_template(
        X_train_resampled,
        y_train_resampled,
        data_test,
        n_splits_n,
        num_classes_n,
        verbose,
        model_function=XGBClassifier,
        model_name="xgbc",
        model_params={
            "n_estimators": 500,  # Reduzido
            "eval_metric": 'mlogloss',
            "use_label_encoder": False,
            "n_jobs": 2  # Reduzido para evitar sobrecarga
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
        model_params={
            "n_estimators": 200,  # Reduzido para minimizar uso de recursos
            "random_state": 42,
            "n_jobs": 2  # Minimiza consumo de CPU
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
            "solver": "lbfgs",  # Default solver, good for multiclass
            "multi_class": "multinomial",  # Handles multiclass directly
            "max_iter": 1000,  # Ensure convergence for large datasets
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