import os
import logging
import pandas as pd
from sklearn.feature_selection import RFE
import numpy as np
from tqdm import tqdm


def monitor_model_rfe(
    model, X_train_scaled, y_train, X_val_scaled, y_val, 
    patience=8, nof_list=np.arange(1, 31), nof=None, 
    log_file="rfe_model.txt", output_csv="feature_selection_results.csv"
):
    base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
    log_folder = os.path.join(base_dir, "logs")
    results_folder = os.path.join(base_dir, "results")

    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file_path = os.path.join(log_folder, log_file)
    logging.basicConfig(
        filename=log_file_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
    )

    high_score = 0
    results = []
    no_improvement_count = 0

    logging.info("Starting feature selection process...")
    for n in tqdm(nof_list, desc="Feature Selection Progress", unit="features"):
        rfe = RFE(estimator=model, n_features_to_select=n)
        X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
        X_val_rfe = rfe.transform(X_val_scaled)

        model.fit(X_train_rfe, y_train)

        train_score = model.score(X_train_rfe, y_train)
        val_score = model.score(X_val_rfe, y_val)

        selected_features = list(X_train_scaled.columns[rfe.support_])
        discarded_features = list(X_train_scaled.columns[~rfe.support_])

        logging.info(f"Features: {n}, Train score: {train_score:.4f}, Val score: {val_score:.4f}")
        logging.info(f"Selected features: {selected_features}")
        logging.info(f"Discarded features: {discarded_features}")

        results.append({
            "nof_features": n,
            "train_score": train_score,
            "val_score": val_score,
            "selected_features": selected_features,
            "discarded_features": discarded_features
        })

        if val_score > high_score:
            high_score = val_score
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            logging.info("Stopping early due to no improvement.")
            break

    results_df = pd.DataFrame(results)

    results_df["selected_features"] = results_df["selected_features"].apply(lambda x: ", ".join(x))
    results_df["discarded_features"] = results_df["discarded_features"].apply(lambda x: ", ".join(x))

    output_csv_path = os.path.join(results_folder, output_csv)
    results_df.to_csv(output_csv_path, index=False)

    logging.info(f"Results saved to {output_csv_path}. Best validation score: {high_score:.4f}")
    print(f"Logs salvos em: {log_file_path}")
    print(f"Resultados salvos em: {output_csv_path}")


def select_best_features_and_save(models_data, output_path="best_features_results.csv"):
    base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
    results_folder = os.path.join(base_dir, "results")
    results = []
    
    for model_name, df in models_data.items():
        df['score_diff'] = abs(df['train_score'] - df['val_score'])
        
        best_row = df.sort_values(by=['val_score', 'score_diff'], ascending=[False, True]).iloc[0]
        
        results.append({
            "Model": model_name,
            "Best_nof_features": best_row['nof_features'],
            "Best_train_score": best_row['train_score'],
            "Best_val_score": best_row['val_score'],
            "Score_Diff": best_row['score_diff'],
            "Selected_features": best_row['selected_features']
        })
    
    results_df = pd.DataFrame(results)
    
    output_csv_path = os.path.join(results_folder, output_path)

    results_df.to_csv(output_csv_path, index=False)
    print(f"Resultados salvos em: {output_csv_path}")
