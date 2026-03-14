import os
import time
import cv2
from datetime import datetime

# =====================
# Config
# =====================
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_fullframe")

SPLITS = ["train", "val"]
CLASSES = ["NONE", "LEFT", "RIGHT", "FIRE", "ENTER", "QUIT"]

FPS_SAVE = 5.0
SAVE_INTERVAL = 1.0 / FPS_SAVE
JPEG_QUALITY = 95


def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_folders():
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)


def save_img(img_bgr, split, cls, counter):
    out_dir = os.path.join(DATASET_DIR, split, cls)
    filename = f"{cls}_{ts()}_{counter:06d}.jpg"
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return path


def overlay(frame, split, cls, capturing, counter):
    cv2.putText(frame, f"split={split} class={cls} capture={'ON' if capturing else 'OFF'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, "0..5 class | c auto | v train/val | q quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"saved total: {counter}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def main():
    ensure_folders()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    split = "train"
    cls = "NONE"
    capturing = False
    last_save = 0.0
    counter = 0

    print("Dataset:", DATASET_DIR)
    print("Touches: 0..5 classes | c capture auto ON/OFF | v train/val | q quit")
    print("0 NONE, 1 LEFT, 2 RIGHT, 3 FIRE, 4 ENTER, 5 QUIT")
    print("NOTE: NONE = à toi de ne pas faire de geste / idéalement main hors champ.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        overlay(frame, split, cls, capturing, counter)
        cv2.imshow("Collector", frame)

        # Capture auto (frame entière)
        t = time.time()
        if capturing and (t - last_save) >= SAVE_INTERVAL:
            counter += 1
            save_img(frame, split, cls, counter)
            last_save = t

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in [ord(str(i)) for i in range(6)]:
            cls = CLASSES[int(chr(key))]
            print("[CLASS]", cls)
        if key == ord("c"):
            capturing = not capturing
            print("[CAPTURE]", "ON" if capturing else "OFF")
        if key == ord("v"):
            split = "val" if split == "train" else "train"
            print("[SPLIT]", split)

    cap.release()
    cv2.destroyAllWindows()
    print("Fin.")


if __name__ == "__main__":
    main()