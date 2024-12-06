import numpy as np

def load_predictions(file_path):
    """
    Carrega previsões de um arquivo .npy.
    
    Parâmetros:
    - file_path: Caminho do arquivo .npy a ser carregado.
    
    Retorno:
    - Array de previsões carregado.
    """
    predictions = np.load(file_path)
    print(f"Previsões carregadas de '{file_path}'.")
    return predictions