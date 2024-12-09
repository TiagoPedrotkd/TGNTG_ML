import pandas as pd
import numpy as np

def save_predictions_to_csv(model, test_data, claim_ids, class_mapping, output_path):
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    predicted_labels_coded = [class_mapping[label] for label in predicted_labels]

    results = pd.DataFrame({
        "Claim Identifier": claim_ids,
        "Claim Injury Type": predicted_labels_coded
    })

    results.to_csv(output_path, index=False)
    print(f"Predições salvas em '{output_path}'.")

def save_predictions_to_csv_ar(model, test_data, claim_ids, output_path):
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    results = pd.DataFrame({
        "Claim Identifier": claim_ids,
        "Agreement Reached": predicted_labels
    })

    results.to_csv(output_path, index=False)
    print(f"Predições salvas em '{output_path}'.")