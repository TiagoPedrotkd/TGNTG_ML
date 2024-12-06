import time  #type: ignore
import os # type: ignore
import logging #type: ignore
from sklearn.model_selection import StratifiedKFold # type: ignore
from sklearn.utils.class_weight import compute_class_weight # type: ignore
import numpy as np # type: ignore
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier # type: ignore
from sklearn.metrics import f1_score, classification_report # type: ignore
import joblib # type: ignore
from catboost import CatBoostClassifier, Pool # type: ignore
from xgboost import XGBClassifier # type: ignore


def meta_model_rf(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    start_time = time.time()

    model_dir = "../models/"
    predictions_dir = "../predictions/"
    logs_dir = "../logs/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Ajustar variáveis de ambiente para evitar problemas de threads
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desativar oneDNN se necessário
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Limitar threads do OpenBLAS
    os.environ["OMP_NUM_THREADS"] = "4"  # Limitar threads OpenMP

    log_file = os.path.join(logs_dir, "training_log_rfc.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Início do treinamento do meta_model com RandomForestClassifier.")

    kf = StratifiedKFold(n_splits=n_splits_n, shuffle=True, random_state=42)

    oof_predictions = np.zeros((X_train_resampled.shape[0], num_classes_n))
    test_predictions = np.zeros((data_test.shape[0], num_classes_n))

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_resampled),
        y=y_train_resampled
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights)}

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

        fold_model = RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weights,
            random_state=42,
            n_jobs=4
        )

        fold_model.fit(X_train_fold, y_train_fold)

        oof_predictions[valid_idx] = fold_model.predict_proba(X_valid_fold)
        test_predictions += fold_model.predict_proba(data_test)

        valid_preds = np.argmax(oof_predictions[valid_idx], axis=1)
        fold_f1 = f1_score(y_valid_fold, valid_preds, average='macro')
        f1_scores.append(fold_f1)

        if verbose:
            print(f"Macro F1-Score do Fold {fold + 1}: {fold_f1:.4f}")
            logging.info(f"Macro F1-Score do Fold {fold + 1}: {fold_f1:.4f}")

        model_filename = os.path.join(model_dir, f"rfc_fold_{fold + 1}.joblib")
        joblib.dump(fold_model, model_filename)

        models.append(fold_model)

    test_predictions /= n_splits_n

    np.save(os.path.join(predictions_dir, "oof_predictions_rfc.npy"), oof_predictions)
    np.save(os.path.join(predictions_dir, "test_predictions_rfc.npy"), test_predictions)

    end_time = time.time()
    total_time = end_time - start_time
    if verbose:
        print(f"Treinamento concluído em {total_time / 60:.2f} minutos.")
        logging.info(f"Treinamento concluído em {total_time / 60:.2f} minutos.")

    return models, f1_scores, oof_predictions, test_predictions

def meta_model(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    
    start_time = time.time()

    model_dir = "../models/"
    predictions_dir = "../predictions/"
    logs_dir = "../logs/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Ajustar variáveis de ambiente para evitar problemas de threads
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desativar oneDNN se necessário
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Limitar threads do OpenBLAS
    os.environ["OMP_NUM_THREADS"] = "4"  # Limitar threads OpenMP

    log_file = os.path.join(logs_dir, 'training_log.txt')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Início do treinamento do meta_model.")

    kf = StratifiedKFold(n_splits=n_splits_n, shuffle=True, random_state=42)

    oof_predictions = np.zeros((X_train_resampled.shape[0], num_classes_n))
    test_predictions = np.zeros((data_test.shape[0], num_classes_n))

    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train_resampled), 
        y=y_train_resampled
    )
    class_weights = class_weights.tolist()

    mapa_codificacao = {
        0: '2. NON-COMP',
        1: '4. TEMPORARY',
        2: '3. MED ONLY',
        3: '5. PPD SCH LOSS',
        4: '6. PPD NSL',
        5: '1. CANCELLED',
        6: '8. DEATH',
        7: '7. PTD'    
    }
        
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

        fold_model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.4,
            class_weights=class_weights,
            eval_metric="MultiClass",
            task_type="CPU",
            use_best_model=False,
            early_stopping_rounds=50,
            verbose=25
        )

        train_pool = Pool(X_train_fold, y_train_fold)
        valid_pool = Pool(X_valid_fold, y_valid_fold)
        fold_model.fit(train_pool, eval_set=valid_pool, verbose=100)

        oof_predictions[valid_idx] = fold_model.predict_proba(X_valid_fold)
        test_predictions += fold_model.predict_proba(data_test)

        valid_preds = np.argmax(oof_predictions[valid_idx], axis=1)
        fold_f1 = f1_score(y_valid_fold, valid_preds, average='macro')
        f1_scores.append(fold_f1)

        report = classification_report(y_valid_fold, valid_preds, target_names=list(mapa_codificacao.values()))
        logging.info(f"Relatório de métricas por classe no Fold {fold + 1}:\n{report}")

        if verbose:
            print(f"Macro F1-Score do Fold {fold + 1}: {fold_f1:.4f}")
            logging.info(f"Macro F1-Score do Fold {fold + 1}: {fold_f1:.4f}")

        model_filename = os.path.join(model_dir, f"catboost_fold_{fold + 1}.joblib")
        joblib.dump(fold_model, model_filename)

        models.append(fold_model)

    test_predictions /= n_splits_n

    oof_preds_labels = np.argmax(oof_predictions, axis=1)
    oof_f1 = f1_score(y_train_resampled, oof_preds_labels, average='macro')
    logging.info(f"Macro F1-Score do OOF: {oof_f1:.4f}")

    np.save(os.path.join(predictions_dir, "oof_predictions.npy"), oof_predictions)
    np.save(os.path.join(predictions_dir, "test_predictions.npy"), test_predictions)

    end_time = time.time()
    total_time = end_time - start_time
    if verbose:
        print(f"Treinamento concluído em {total_time / 60:.2f} minutos.")
        logging.info(f"Treinamento concluído em {total_time / 60:.2f} minutos.")

    return models, f1_scores, oof_predictions, test_predictions

def meta_model_xgbc(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):

    start_time = time.time()

    model_dir = "../models/"
    predictions_dir = "../predictions/"
    logs_dir = "../logs/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Ajustar variáveis de ambiente para evitar problemas de threads
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desativar oneDNN se necessário
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Limitar threads do OpenBLAS
    os.environ["OMP_NUM_THREADS"] = "4"  # Limitar threads OpenMP

    log_file = os.path.join(logs_dir,"training_log_xgbc.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Início do treinamento do meta_model.")

    kf = StratifiedKFold(n_splits=n_splits_n, shuffle=True, random_state=42)

    oof_predictions = np.zeros((X_train_resampled.shape[0], num_classes_n))
    test_predictions = np.zeros((data_test.shape[0], num_classes_n))

    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train_resampled), 
        y=y_train_resampled
    )
    class_weights = class_weights.tolist()

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

        fold_model = XGBClassifier(
            n_estimators=800,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=4
        )

        fold_model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            verbose=100
        )

        oof_predictions[valid_idx] = fold_model.predict_proba(X_valid_fold)
        test_predictions += fold_model.predict_proba(data_test)

        valid_preds = np.argmax(oof_predictions[valid_idx], axis=1)
        fold_f1 = f1_score(y_valid_fold, valid_preds, average='macro')
        f1_scores.append(fold_f1)

        if verbose:
            print(f"Macro F1-Score do Fold {fold + 1}: {fold_f1:.4f}")
            logging.info(f"Macro F1-Score do Fold {fold + 1}: {fold_f1:.4f}")

        model_filename =  os.path.join(model_dir, f"xgbc_fold_{fold + 1}.joblib")
        joblib.dump(fold_model, model_filename)

        models.append(fold_model)

    test_predictions /= n_splits_n

    np.save(os.path.join(predictions_dir, "oof_predictions_xgbc.npy"), oof_predictions)
    np.save(os.path.join(predictions_dir, "test_predictions_xgbc.npy"), test_predictions)

    end_time = time.time()
    total_time = end_time - start_time
    if verbose:
        print(f"Treinamento concluído em {total_time / 60:.2f} minutos.")
        logging.info(f"Treinamento concluído em {total_time / 60:.2f} minutos.")

    return models, f1_scores, oof_predictions, test_predictions

def meta_model_et(X_train_resampled, y_train_resampled, data_test, n_splits_n=3, num_classes_n=8, verbose=True):
    start_time = time.time()

    model_dir = "../models/"
    predictions_dir = "../predictions/"
    logs_dir = "../logs/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Ajustar variáveis de ambiente para evitar problemas de threads
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Desativar oneDNN se necessário
    os.environ["OPENBLAS_NUM_THREADS"] = "4"  # Limitar threads do OpenBLAS
    os.environ["OMP_NUM_THREADS"] = "4"  # Limitar threads OpenMP

    log_file = os.path.join(logs_dir,"training_log_et.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Início do treinamento do meta_model com ExtraTreesClassifier.")

    kf = StratifiedKFold(n_splits=n_splits_n, shuffle=True, random_state=42)

    oof_predictions = np.zeros((X_train_resampled.shape[0], num_classes_n))
    test_predictions = np.zeros((data_test.shape[0], num_classes_n))

    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train_resampled), 
        y=y_train_resampled
    )
    class_weights = class_weights.tolist()

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

        fold_model = ExtraTreesClassifier(
            n_estimators=100,
            random_state=42
        )

        fold_model.fit(X_train_fold, y_train_fold)

        oof_predictions[valid_idx] = fold_model.predict_proba(X_valid_fold)
        test_predictions += fold_model.predict_proba(data_test)

        valid_preds = np.argmax(oof_predictions[valid_idx], axis=1)
        fold_f1 = f1_score(y_valid_fold, valid_preds, average='macro')
        f1_scores.append(fold_f1)

        if verbose:
            print(f"Macro F1-Score do Fold {fold + 1}: {fold_f1:.4f}")
            logging.info(f"Macro F1-Score do Fold {fold + 1}: {fold_f1:.4f}")

        model_filename =  os.path.join(model_dir, f"et_fold_{fold + 1}.joblib")
        joblib.dump(fold_model, model_filename)

        models.append(fold_model)

    test_predictions /= n_splits_n

    np.save(os.path.join(predictions_dir, "oof_predictions_et.npy"), oof_predictions)
    np.save(os.path.join(predictions_dir, "test_predictions_et.npy"), test_predictions)

    end_time = time.time()
    total_time = end_time - start_time
    if verbose:
        print(f"Treinamento concluído em {total_time / 60:.2f} minutos.")
        logging.info(f"Treinamento concluído em {total_time / 60:.2f} minutos.")

    return models, f1_scores, oof_predictions, test_predictions
