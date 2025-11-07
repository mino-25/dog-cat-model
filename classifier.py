from fastai.vision.all import *
from pathlib import Path
import zipfile
import random
import shutil

if __name__ == "__main__":
    # Chemins
    zip_path = Path('data/train.zip')
    extract_path = Path('animals')

    # 2️⃣ Créer les sous-dossiers dog/ et cat/
    (extract_path/'dog').mkdir(parents=True, exist_ok=True)
    (extract_path/'cat').mkdir(parents=True, exist_ok=True)

    # 3️⃣ Extraction du zip dans un dossier temporaire
    temp_extract_path = Path('data/temp')
    if temp_extract_path.exists():
        shutil.rmtree(temp_extract_path)
    temp_extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)

    # 4️⃣ Prendre un sous-ensemble aléatoire
    train_files = list(temp_extract_path.glob('train/*'))
    dog_files = [f for f in train_files if f.name.startswith('dog')]
    cat_files = [f for f in train_files if f.name.startswith('cat')]

    for f in random.sample(dog_files, 50):
        f.rename(extract_path/'dog'/f.name)
    for f in random.sample(cat_files, 50):
        f.rename(extract_path/'cat'/f.name)

    # 5️⃣ Supprimer le dossier temporaire
    shutil.rmtree(temp_extract_path)

    # 6️⃣ Nettoyage
    failed = verify_images(get_image_files(extract_path))
    failed.map(Path.unlink)
    print(f"Deleted {len(failed)} failed images")

    # 7️⃣ DataLoaders
    dls = ImageDataLoaders.from_folder(extract_path, valid_pct=0.2, seed=42,
                                       item_tfms=Resize(128))

    # 8️⃣ Modèle et entraînement
    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(3)
    learn.export('dog_cat_model.pkl')
    print("Modèle sauvegardé !")

    # 9️⃣ Test
    test_img_path = Path('cat.jpg')
    if test_img_path.exists():
        img = PILImage.create(test_img_path)
        pred, pred_idx, probs = learn.predict(img)
        print(f"Prediction: {pred}; Probability: {probs[pred_idx]:.4f}")
    else:
        print(f"Image {test_img_path} non trouvée. Place ton fichier cat.jpg au même niveau que classifier.py")
