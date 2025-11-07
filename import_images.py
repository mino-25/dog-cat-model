import zipfile
from pathlib import Path
import random
import shutil

if __name__ == "__main__":
    # Chemins
    zip_path = Path('data/train.zip')  # ton fichier Kaggle
    extract_path = Path('animals')

    # Créer les sous-dossiers dog/ et cat/
    (extract_path/'dog').mkdir(parents=True, exist_ok=True)
    (extract_path/'cat').mkdir(parents=True, exist_ok=True)

    # Extraction du zip dans un dossier temporaire
    temp_extract_path = Path('data/temp')
    if temp_extract_path.exists():
        shutil.rmtree(temp_extract_path)
    temp_extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)

    # Prendre un sous-ensemble aléatoire (50 images par classe)
    train_files = list(temp_extract_path.glob('train/*'))
    dog_files = [f for f in train_files if f.name.startswith('dog')]
    cat_files = [f for f in train_files if f.name.startswith('cat')]

    for f in random.sample(dog_files, 50):
        f.rename(extract_path/'dog'/f.name)
    for f in random.sample(cat_files, 50):
        f.rename(extract_path/'cat'/f.name)

    # Supprimer le dossier temporaire
    shutil.rmtree(temp_extract_path)

    print("Images importées et sous-ensemble créé.")
