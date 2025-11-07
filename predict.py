from fastai.vision.all import *
from pathlib import Path

if __name__ == "__main__":
    # Charger le modèle exporté
    learn_inf = load_learner('dog_cat_model.pkl')

    # Image de test
    test_img_path = Path('chien.jpg')
    if test_img_path.exists():
        img = PILImage.create(test_img_path)
        pred, pred_idx, probs = learn_inf.predict(img)
        print(f"Prediction: {pred}; Probability: {probs[pred_idx]:.4f}")
    else:
        print(f"Image {test_img_path} non trouvée. Place ton fichier cat.jpg au même niveau que ce script")
