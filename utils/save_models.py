import os
import joblib

def save_model(model, model_name, directory="../models/"):
    
    os.makedirs(directory, exist_ok=True)
    
    model_path = os.path.join(directory, f"{model_name}.pkl")
    
    joblib.dump(model, model_path)
    
    print(f"Modelo '{model_name}' salvo com sucesso em: {model_path}")
    return model_path