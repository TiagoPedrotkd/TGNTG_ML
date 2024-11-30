def compress_csv(folder_path, output_zip):
    import zipfile
    import os
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                zipf.write(os.path.join(folder_path, file), arcname=file)
    print(f"Ficheiros CSV da pasta '{folder_path}' foram compactados em: {output_zip}")
