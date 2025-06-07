import os
import cv2
import numpy as np

from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
import yaml

# === CONFIG ===
input_root = "badania"
output_root = "aligned_data"
train_list_file = "train_list.txt"
model_path = "models"
device = "cuda:0"
scene = "non-mask"

# === LOAD MODELS ===
with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

# Detektor twarzy
det_loader = FaceDetModelLoader(model_path, 'face_detection', model_conf[scene]['face_detection'])
det_model, det_cfg = det_loader.load_model()
det_handler = FaceDetModelHandler(det_model, device, det_cfg)

# Wyrównywacz (landmarks)
align_loader = FaceAlignModelLoader(model_path, 'face_alignment', model_conf[scene]['face_alignment'])
align_model, align_cfg = align_loader.load_model()
align_handler = FaceAlignModelHandler(align_model, device, align_cfg)

# Cropper
cropper = FaceRecImageCropper()

# === PRZETWARZANIE ===
if not os.path.exists(output_root):
    os.makedirs(output_root)

id_map = {}  # ImieNazwisko -> ID
next_id = 0
lines = []

for person_name in os.listdir(input_root):
    person_path = os.path.join(input_root, person_name)
    if not os.path.isdir(person_path):
        continue

    person_output = os.path.join(output_root, person_name)
    os.makedirs(person_output, exist_ok=True)

    if person_name not in id_map:
        id_map[person_name] = next_id
        next_id += 1

    person_id = id_map[person_name]

    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Nie wczytano obrazu: {img_path}")
            continue

        dets = det_handler.inference_on_image(image)
        if len(dets) == 0:
            print(f"Brak twarzy w: {img_path}")
            continue

        box = dets[0][:4]
        landmarks = align_handler.inference_on_image(image, box)
        landmarks_list = []
        for (x, y) in landmarks.astype(np.int32):
            landmarks_list.extend((x, y))

        cropped = cropper.crop_image_by_mat(image, landmarks_list)
        cropped_resized = cv2.resize(cropped, (112, 112))  # Wymuszamy rozmiar

        save_path = os.path.join(person_output, img_name)
        cv2.imwrite(save_path, cropped_resized)

        rel_path = os.path.relpath(save_path, output_root)
        lines.append(f"{rel_path} {person_id}")

# === ZAPISZ LISTĘ TRENINGOWĄ ===
with open(train_list_file, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print("Gotowe! Dane zapisane w:", output_root)
print("Lista treningowa:", train_list_file)
