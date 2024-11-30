def decompress_zip(zip_path, extract_to):
    import zipfile
    import os
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)
    print(f"Ficheiros extraídos para a pasta: {extract_to}")
