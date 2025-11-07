from fastai.vision.all import *
from pathlib import Path

if __name__ == "__main__":
    path = Path('animals')

    # Vérifier les images corrompues
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(f"Deleted {len(failed)} failed images")

    # DataLoaders
    dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42,
                                       item_tfms=Resize(128))

    # Modèle et entraînement
    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(3)

    # Sauvegarde du modèle
    learn.export('dog_cat_model.pkl')
    print("Modèle entraîné et sauvegardé !")
