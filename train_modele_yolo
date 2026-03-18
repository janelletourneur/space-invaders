##code d'entrainement du modèle sur notre dataset (avec nos photos) ou le modèle passe 20x sur chacune des images + le code permet des modifications des images (ex: zoom) afin d'anticiper les cgt
## modèle YOLO (=You Only Look Once, ici YOLOv8)  

import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def get_val_name(dataset_path: Path) -> str:
    if (dataset_path / "val").exists():
        return "val"
    elif (dataset_path / "test").exists():
        return "test"
    else:
        raise FileNotFoundError(
            "Dossier de validation introuvable.\n"
            "Attendu : 'val/' ou 'test/' dans " + str(dataset_path)
        )


def check_dataset(dataset_path: str):
    dataset_path = Path(dataset_path)

    if not (dataset_path / "train").exists():
        raise FileNotFoundError(
            "Dossier 'train/' introuvable dans " + str(dataset_path) + "\n"
            "Verifie que le chemin est correct."
        )

    val_name = get_val_name(dataset_path)
    classes  = sorted([d.name for d in (dataset_path / "train").iterdir() if d.is_dir()])

    print("Dataset : " + str(dataset_path.resolve()))
    print("Dossier validation : " + val_name + "/\n")
    print("{:<12} {:>8} {:>8}".format("Classe", "Train", val_name.capitalize()))
    print("-" * 30)

    total_train, total_val = 0, 0
    for cls in classes:
        n_train = len(list((dataset_path / "train"   / cls).glob("*.*")))
        n_val   = len(list((dataset_path / val_name  / cls).glob("*.*")))
        total_train += n_train
        total_val   += n_val
        print("{:<12} {:>8} {:>8}".format(cls, n_train, n_val))

    print("-" * 30)
    print("{:<12} {:>8} {:>8}\n".format("TOTAL", total_train, total_val))

    return classes


def prepare_yolo_dataset(dataset_path: str, output_path: str) -> str:
    dataset_path = Path(dataset_path)
    output_path  = Path(output_path)

    if (output_path / "train").exists() and (output_path / "val").exists():
        print("Dataset YOLO deja prepare dans " + str(output_path) + "\n")
        return str(output_path)

    val_name = get_val_name(dataset_path)
    print("Copie du dataset vers " + str(output_path) + " ...")
    output_path.mkdir(parents=True, exist_ok=True)

    shutil.copytree(dataset_path / "train",    output_path / "train")
    print("  train/ copie")

    shutil.copytree(dataset_path / val_name,   output_path / "val")
    print("  " + val_name + "/ copie -> val/")

    print("Dataset pret !\n")
    return str(output_path)


def train(dataset_path: str, epochs: int, img_size: int, output_dir: str):

    classes = check_dataset(dataset_path)

    yolo_dataset = prepare_yolo_dataset(
        dataset_path=dataset_path,
        output_path=os.path.join(output_dir, "yolo_dataset"),
    )

    print("Chargement du modele YOLOv8n-cls (nano, ~6MB)...")
    model = YOLO("yolov8n-cls.pt")

    print("Entrainement (" + str(epochs) + " epochs max, early stopping patience=10)...\n")

    model.train(
        data=yolo_dataset,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        patience=10,
        save=True,
        project=output_dir,
        name="gesture_model",
        verbose=True,
        fliplr=0.0,      # PAS de flip gauche/droite (LEFT != RIGHT !)
        flipud=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.1,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
    )

    best = os.path.join(output_dir, "gesture_model", "weights", "best.pt")

    print("\nEntrainement termine !")
    print("Meilleur modele -> " + best)
    print("Courbes         -> " + output_dir + "/gesture_model/")
    print("\nPour jouer :")
    print("  python camera_control_yolo.py --model " + best)

    return best

def quick_test(model_path: str, dataset_path: str):
    print("\nTest rapide...\n")
    model = YOLO(model_path)

    dataset_path = Path(dataset_path)
    val_name = get_val_name(dataset_path)
    classes  = sorted([d.name for d in (dataset_path / val_name).iterdir() if d.is_dir()])

    correct, total = 0, 0
    for cls in classes:
        images = list((dataset_path / val_name / cls).glob("*.jpg")) + \
                 list((dataset_path / val_name / cls).glob("*.png"))
        for img_path in images[:5]:
            results  = model(str(img_path), verbose=False)
            pred_cls = results[0].names[results[0].probs.top1]
            ok = "OK" if pred_cls.upper() == cls.upper() else "XX"
            print("  [" + ok + "]  Vrai: " + cls.ljust(10) + "  Predit: " + pred_cls)
            correct += int(pred_cls.upper() == cls.upper())
            total   += 1

    if total > 0:
        print("\nPrecision : " + str(correct) + "/" + str(total) +
              " (" + str(round(correct / total * 100)) + "%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrainement YOLOv8 Classification - gestes Space Invaders"
    )
    parser.add_argument("--dataset", type=str, default="./dataset_fullframe",
                        help="Chemin vers le dossier contenant train/ et val/ (ou test/)")
    parser.add_argument("--epochs",  type=int, default=50,
                        help="Nombre d'epoques max (defaut: 50)")
    parser.add_argument("--imgsz",   type=int, default=224,
                        help="Taille des images (defaut: 224)")
    parser.add_argument("--output",  type=str, default="./model_output",
                        help="Dossier de sortie")
    parser.add_argument("--test",    action="store_true",
                        help="Test rapide apres entrainement")
    args = parser.parse_args()

    best = train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        img_size=args.imgsz,
        output_dir=args.output,
    )

    if args.test:
        quick_test(best, args.dataset)
