
import sys
import os
import hashlib

from PySide6.QtMultimedia import QSoundEffect

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import torch
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QFileDialog, QComboBox, QMessageBox, QDialog, QFormLayout, QLineEdit, QCheckBox, QDialogButtonBox, QSlider,
    QHBoxLayout, QSizePolicy
)
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
# from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
import yaml
#
from core.model_loader.face_recognition.MagFaceModelLoader import MagFaceModelLoader as FaceRecModelLoader
# from core.model_handler.face_recognition.MagFaceModelHandler import MagFaceModelHandler as FaceRecModelHandler


class ConfigurationDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Konfiguracja systemu")

        self.crop_mode = QComboBox()
        self.crop_mode.addItems(["Automatyczne", "Ręczne"])

        self.show_grid = QCheckBox("Pokaż linie pomocnicze")

        self.similarity_threshold = QSlider(Qt.Horizontal)
        self.similarity_threshold.setMinimum(30)
        self.similarity_threshold.setMaximum(100)
        self.similarity_threshold.setValue(40)
        self.similarity_threshold.setTickInterval(10)
        self.similarity_threshold.setTickPosition(QSlider.TicksBelow)

        self.threshold_value_label = QLabel("40%")
        self.similarity_threshold.valueChanged.connect(
            lambda val: self.threshold_value_label.setText(f"{val}%")
        )

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.similarity_threshold)
        threshold_layout.addWidget(self.threshold_value_label)

        self.notify_box = QCheckBox("Ramka")
        self.notify_screen = QCheckBox("Alert ekranowy")
        self.notify_sound = QCheckBox("Alert dźwiękowy")

        form = QFormLayout()
        form.addRow("Kadrowanie:", self.crop_mode)
        form.addRow("", self.show_grid)
        form.addRow("Próg podobieństwa:", threshold_layout)
        form.addRow("Powiadomienia:", self.notify_box)
        form.addRow("", self.notify_screen)
        form.addRow("", self.notify_sound)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def get_settings(self):
        return {
            "crop_mode": self.crop_mode.currentText(),
            "show_grid": self.show_grid.isChecked(),
            "similarity_threshold": self.similarity_threshold.value() / 100.0,
            "notify_box": self.notify_box.isChecked(),
            "notify_screen": self.notify_screen.isChecked(),
            "notify_sound": self.notify_sound.isChecked()
        }


class CropSelectionDialog(QDialog):
    def __init__(self, camera_index=0):
        super().__init__()
        self.setWindowTitle("Wybierz obszar kadrowania")
        self.setFixedSize(800, 600)
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.resize(800, 600)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.start_pos = None
        self.end_pos = None
        self.crop_rect = None

        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.video_label.mousePressEvent = self.mouse_press
        self.video_label.mouseMoveEvent = self.mouse_move
        self.video_label.mouseReleaseEvent = self.mouse_release

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return
        self.frame = frame.copy()

        if not hasattr(self, "camera_width"):
            self.camera_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        disp = frame.copy()

        if self.start_pos and self.end_pos:
            x1, y1 = map(int, self.start_pos)
            x2, y2 = map(int, self.end_pos)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rgb_image = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def get_crop_rect_scaled_to_camera(self):
        if not self.start_pos or not self.end_pos:
            return None

        scale_x = self.camera_width / self.frame.shape[1]
        scale_y = self.camera_height / self.frame.shape[0]

        x1, y1 = map(int, self.start_pos)
        x2, y2 = map(int, self.end_pos)

        x1 = int(min(x1, x2) * scale_x)
        y1 = int(min(y1, y2) * scale_y)
        w = int(abs(x2 - x1) * scale_x)
        h = int(abs(y2 - y1) * scale_y)

        return (x1, y1, w, h)
    def map_mouse_to_frame(self, pos):
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        frame_height, frame_width, _ = self.frame.shape

        # Oblicz skalę (z zachowaniem proporcji)
        scale = min(label_width / frame_width, label_height / frame_height)

        # Wymiary obrazu po skalowaniu
        scaled_w = int(frame_width * scale)
        scaled_h = int(frame_height * scale)

        # Offset (centrowanie)
        offset_x = (label_width - scaled_w) // 2
        offset_y = (label_height - scaled_h) // 2

        # Współrzędne kliknięcia
        x = int((pos.x() - offset_x) / scale)
        y = int((pos.y() - offset_y) / scale)

        # Zabezpieczenie przed kliknięciem poza obrazem
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))

        return (x, y)

    def mouse_press(self, event):
        self.start_pos = self.map_mouse_to_frame(event.position().toPoint())

    def mouse_move(self, event):
        if self.start_pos:
            self.end_pos = self.map_mouse_to_frame(event.position().toPoint())
            self.update()

    def mouse_release(self, event):
        self.end_pos = self.map_mouse_to_frame(event.position().toPoint())
        self.crop_rect = self.get_crop_rect()
        self.capture.release()
        self.timer.stop()
        self.accept()

    def get_crop_rect(self):
        if not self.start_pos or not self.end_pos:
            return None
        x1, y1 = map(int, self.start_pos)
        x2, y2 = map(int, self.end_pos)
        return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))  # x, y, w, h

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.settings = {}
        self.crop_rect = None

        self.sound_effect = QSoundEffect()
        self.sound_effect.setSource(QUrl.fromLocalFile("alert.wav"))
        self.sound_effect.setLoopCount(1)
        self.sound_effect.setVolume(0.9)

        self.setWindowTitle("Live Face Recognition")
        self.capture = None      # obiekt kamery / wideo
        self.timer = QTimer()    # zegar odświeżający podgląd kamery
        self.frame = None        # ostatnia klatka z kamery

        # Etykieta, w której wyświetlamy obraz z kamery
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Wybór źródła obrazu: kamera lub plik wideo
        self.source_combo = QComboBox()
        self.source_combo.addItems(self.detect_cameras() + ["Plik wideo"])

        # Przycisk uruchamiający kamerę lub wideo
        self.select_btn = QPushButton("Wybierz źródło")
        self.select_btn.clicked.connect(self.start_capture)

        # Przycisk zamykający aplikację
        self.quit_btn = QPushButton("Zakończ")
        self.quit_btn.clicked.connect(self.close_app)

        # Przycisk do analizy folderu badania
        self.analyze_btn = QPushButton("Analizuj folder 'badania'")
        self.analyze_btn.clicked.connect(self.analyze_study_images)

        # Inicjalizacja obiektu do przycinania twarzy
        self.face_cropper = FaceRecImageCropper()

        # Ustawienie layoutu aplikacji
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.source_combo)
        layout.addWidget(self.select_btn)
        layout.addWidget(self.quit_btn)
        layout.addWidget(self.analyze_btn)  # Dodaj przycisk do layoutu

        self.setLayout(layout)
        # Co 30 ms uruchamiana jest funkcja aktualizacji obrazu z kamery
        self.timer.timeout.connect(self.update_frame)

        # Wczytanie modeli i bazy twarzy
        self.load_models()
        self.load_face_database()

    def compute_db_hash(self, db_path):
        """Zwraca hash na podstawie ścieżek i dat modyfikacji zdjęć."""

        # Inicjalizacja obiektu do tworzenia sumy kontrolnej (hashu) w algorytmie MD5
        hash_md5 = hashlib.md5()

        # Przechodzimy przez wszystkie pliki w katalogu bazy danych (rekurencyjnie)
        for root, _, files in os.walk(db_path):
            # Iterujemy po posortowanej liście plików, aby zapewnić stabilność hasha
            for file in sorted(files):
                # Sprawdzamy, czy plik jest obrazem (obsługiwane formaty)
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                # Tworzymy pełną ścieżkę do pliku
                path = os.path.join(root, file)

                # Dodajemy ścieżkę pliku do danych wejściowych hasha
                hash_md5.update(path.encode())

                # Dodajemy czas ostatniej modyfikacji pliku do danych wejściowych hasha
                hash_md5.update(str(os.path.getmtime(path)).encode())

        # Zwracamy końcowy hash jako ciąg znaków (hex)
        return hash_md5.hexdigest()
    def load_models(self):
        # Odczytaj konfigurację modeli z pliku YAML
        with open('config/model_conf.yaml') as f:
            model_conf = yaml.load(f, Loader=yaml.FullLoader)

        # Katalog, w którym znajdują się wszystkie modele (np. .pth, .onnx itp.)
        model_path = 'models'
        scene = 'non-mask'

        # Detektor twarzy
        det_loader = FaceDetModelLoader(model_path, 'face_detection', model_conf[scene]['face_detection']) # Utwórz loader do modelu detekcji twarzy (np. RetinaFace)
        det_model, det_cfg = det_loader.load_model() # Załaduj model i jego konfigurację
        self.faceDetModelHandler = FaceDetModelHandler(det_model, 'cuda:0', det_cfg) # Utwórz handler do obsługi detekcji twarzy

        # Wyrównywacz twarzy (landmarki)
        align_loader = FaceAlignModelLoader(model_path, 'face_alignment', model_conf[scene]['face_alignment']) # Utwórz loader do modelu wykrywania punktów charakterystycznych (landmarks)
        align_model, align_cfg = align_loader.load_model() # Załaduj model i konfigurację
        self.faceAlignModelHandler = FaceAlignModelHandler(align_model, 'cuda:0', align_cfg) # Utwórz handler do obsługi wyrównywania twarzy

        # Rozpoznawanie (ekstrakcja cech)
        rec_loader = FaceRecModelLoader(model_path, 'face_recognition', model_conf[scene]['face_recognition']) # Loader do modelu ekstrakcji cech (np. ArcFace, MobileFaceNet)
        rec_model, rec_cfg = rec_loader.load_model() # Załaduj model i konfigurację
        self.faceRecModelHandler = FaceRecModelHandler(rec_model, 'cuda:0', rec_cfg) # Handler do rozpoznawania twarzy (porównywanie wektorów cech)

        self.cropper = FaceRecImageCropper() # Obiekt do przycinania twarzy według punktów charakterystycznych

    def load_face_database(self, db_path="faces_db", cache_file="face_cache.npz"):
        existing_files = []  # już istniejące zdjęcia
        updated_db = []  # cechy przetworzonych zdjęć
        updated_names = []  # odpowiadające im imiona

        # Sprawdź, czy istnieje cache
        if os.path.exists(cache_file):
            print("Ładowanie bazy z cache...")
            data = np.load(cache_file, allow_pickle=True)   # Wczytaj dane z pliku .npz (zapisana baza)
            cached_features = list(data["features"])        # Lista cech z cache
            cached_names = list(data["names"])              # Lista nazw przypisanych do cech
            cached_files = list(data["files"]) if "files" in data else [] # Lista ścieżek do plików odpowiadających zapisanym cechom

            # Odfiltruj usunięte zdjęcia
            for feat, name, file in zip(cached_features, cached_names, cached_files):
                if os.path.exists(file):        # Jeśli plik nadal istnieje, zachowaj go
                    updated_db.append(feat)
                    updated_names.append(name)
                    existing_files.append(file)
                else:
                    print(f"Plik usunięty z bazy: {file}")

        print("Skanowanie katalogu bazy twarzy...")
        # Skanuj bazę i przetwarzaj tylko nowe obrazy
        for person_name in os.listdir(db_path):                 # Iteruj po podfolderach (imionach osób)
            person_folder = os.path.join(db_path, person_name)
            if not os.path.isdir(person_folder):                # Pomijaj pliki – przetwarzaj tylko foldery
                continue

            for img_name in os.listdir(person_folder):          # Iteruj po zdjęciach w folderze osoby
                img_path = os.path.join(person_folder, img_name)

                if img_path in existing_files:                  # Jeśli to zdjęcie było już przetwarzane wcześniej – pomiń
                    continue

                image = cv2.imread(img_path)                    # Wczytaj zdjęcie
                if image is None:
                    continue

                dets = self.faceDetModelHandler.inference_on_image(image) # Wykryj twarz na obrazie
                if len(dets) == 0:
                    print(f"Nie wykryto twarzy w {img_path}")
                    continue

                box = dets[0][:4]               # Wydobądź współrzędne pierwszej wykrytej twarzy
                landmarks = self.faceAlignModelHandler.inference_on_image(image, box) # Oblicz punkty charakterystyczne (landmarki)
                landmarks_list = []             # Przekształć landmarki do listy [x1, y1, x2, y2, ..., x106, y106
                for (x, y) in landmarks.astype(np.int32):
                    landmarks_list.extend((x, y))

                cropped = self.face_cropper.crop_image_by_mat(image, landmarks_list) # Przytnij obraz twarzy według punktów
                feature = self.faceRecModelHandler.inference_on_image(cropped)      # Wyodrębnij cechy twarzy (wektor 512D, np. z ArcFace)

                # print("dbFEATURE:", feature)
                # print("dbSUMA:", np.sum(feature))
                feature = feature / np.linalg.norm(feature)


                # Dodaj nową twarz do bazy
                updated_db.append(feature)
                updated_names.append(person_name)
                existing_files.append(img_path)
                print(f"Dodano {person_name}: {img_path}")

        # Przypisz dane do obiektu
        self.db = updated_db
        self.names = updated_names
        self.db_files = existing_files

        # Zapisz cache zaktualizowany
        np.savez_compressed(cache_file, features=np.array(self.db), names=np.array(self.names),
                            files=np.array(self.db_files))
        print("Baza zapisana do cache!")

    def detect_cameras(self, max_devices=5):
        available = []                          # Tworzymy pustą listę na dostępne kamery
        for i in range(max_devices):            # Iterujemy po potencjalnych urządzeniach wideo (np. 0, 1, 2...)
            cap = cv2.VideoCapture(i)           # Próbujemy otworzyć kamerę o numerze i
            if cap.isOpened():                  # Jeśli kamera została poprawnie otwarta (czyli istnieje i działa)
                available.append(f"Kamera {i}") # Dodajemy ją do listy jako np. "Kamera 0", "Kamera 1", itd.
                cap.release()                   # Zwolnij kamerę po sprawdzeniu
        return available                        # Zwracamy listę dostępnych kamer

    def start_capture(self):
        self.crop_rect = self.settings.get("crop_rect")
        selected = self.source_combo.currentText()  # Pobierz aktualnie wybraną opcję z rozwijanego menu (kamera lub plik wideo)
        if selected == "Plik wideo":                # Jeśli wybrano "Plik wideo", otwórz okno dialogowe do wyboru pliku
            source = self.select_video_file()
        else:
            source = int(selected.split()[-1])  # np. "Kamera 1" → 1

        if source is None:
            return

        self.capture = cv2.VideoCapture(source)     # Utwórz obiekt wideo (kamera lub plik) dla OpenCV
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)     # Ustaw preferowaną rozdzielczość obrazu na Full HD
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not self.capture.isOpened():             # Sprawdź, czy źródło obrazu zostało poprawnie otwarte
            QMessageBox.critical(self, "Błąd", "Nie udało się otworzyć źródła obrazu.")
            return

        self.timer.start(30) # co 30ms odświeżenie

    def select_video_file(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Wybierz plik wideo")
        return file_path if file_path else None

    def l2_normalize(self, vec):
        return vec / np.linalg.norm(vec)
    def recognize(self, full_image, box):
        x1, y1, x2, y2 = [int(coord) for coord in box[:4]]  # Rozpakowanie współrzędnych ramki detekcji twarzy

        try:
            # Wykrywanie punktów charakterystycznych (landmarków) twarzy na podstawie współrzędnych detekcji
            landmarks = self.faceAlignModelHandler.inference_on_image(full_image, [x1, y1, x2, y2])
            landmarks_list = []     # Zamiana landmarków do listy współrzędnych
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))    # dodajemy x i y jako osobne wartości do jednej listy

            # Przycinanie obrazu wg punktów
            cropped = self.face_cropper.crop_image_by_mat(full_image, landmarks_list)

            # Ekstrakcja cech
            feature = self.faceRecModelHandler.inference_on_image(cropped)
            # print("recoFEATURE:", feature)
            # print("recoSUMA:", np.sum(feature))
            feature = self.l2_normalize(feature)

            # Porównanie z bazą
            best_score = -1
            best_name = "Nieznany"
            for i, vec in enumerate(self.db):
                vec = self.l2_normalize(vec)
                score = np.dot(feature, vec)    # oblicz iloczyn skalarny jako miarę podobieństwa
                if score > best_score:
                    best_score = score
                    best_name = self.names[i]
            threshold = self.settings.get("similarity_threshold", 0.4)
            return best_name if best_score > threshold else "Nieznany", best_score

        except Exception as e:
            print(f"Błąd przy rozpoznawaniu twarzy: {e}")
            return "Nieznany", 0.0

    def analyze_study_images(self):
        import re

        study_root = "badania"
        output_file = "badania.txt"

        filename_pattern = re.compile(r".*?_(?P<age>\d+)(?:_\d+)?\.jpg$", re.IGNORECASE)

        with open(output_file, "w", encoding="utf-8") as report:
            for person_name in os.listdir(study_root):
                person_folder = os.path.join(study_root, person_name)
                if not os.path.isdir(person_folder):
                    continue

                for file_name in os.listdir(person_folder):
                    if not file_name.lower().endswith(".jpg"):
                        continue

                    match = filename_pattern.match(file_name)
                    if not match:
                        print(f"[POMINIĘTO] Nieprawidłowa nazwa pliku: {file_name}")
                        continue

                    age = match.group("age")
                    img_path = os.path.join(person_folder, file_name)

                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"[BŁĄD] Nie udało się wczytać obrazu: {img_path}")
                        continue

                    dets = self.faceDetModelHandler.inference_on_image(image)
                    if len(dets) == 0:
                        print(f"[BRAK TWARZY] {img_path}")
                        report.write(f"{person_name};{age};NIE;0.00\n")
                        continue

                    name, score = self.recognize(image, dets[0])
                    matched = "TAK" if name == person_name else "NIE"
                    report.write(f"{person_name};{age};{matched};{score * 100:.2f}\n")
                    print(f"[ZAPISANO] {file_name} → {matched} ({score:.2f})")

    def update_frame(self):
        ret, self.frame = self.capture.read() # Próba odczytu jednej klatki z kamery / wideo
        if not ret:
            return

        frame = self.frame.copy()
        if self.settings.get("crop_mode") == "Ręczne" and self.crop_rect:
            x, y, w, h = self.crop_rect
            frame = frame[y:y + h, x:x + w]

        image = frame   # Utwórz kopię klatki do dalszego przetwarzania
        try:
            dets = self.faceDetModelHandler.inference_on_image(image)   # Wykryj twarze w obrazie
            print(f"Wykryto {len(dets)} twarze")
            print(f"dets typ: {type(dets)}")
            print(f"dets zawartość:\n{dets}")
            for i, box in enumerate(dets): # Przejdź po wszystkich wykrytych twarzach
                try:
                    x1, y1, x2, y2 = [int(coord) for coord in box[:4]]  # Rozpakuj współrzędne ramki detekcji twarzy
                    print(f"Rysuję ramkę: {x1}, {y1}, {x2}, {y2} dla ")

                    name, score = self.recognize(image, box)    # Rozpoznaj osobę na podstawie tej twarzy

                    if self.settings.get("notify_box", False):
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)    # Rysuj zieloną ramkę wokół twarzy
                        cv2.putText(image, f"{name} ({score:.2f})", (x1, y1 - 10),  # Wyświetl nazwę i podobieństwo nad ramką
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if self.settings.get("notify_screen", False):
                        QMessageBox.information(self, "Wykryto twarz", f"Wykryto: {name} ({score:.2f})")

                    if self.settings.get("notify_sound", False):
                        self.sound_effect.play()

                except Exception as e:
                    print(f"Błąd przy rysowaniu ramki {i}: {e}")
        except:
            pass

        if self.settings.get("show_grid", False):
            h, w, _ = image.shape
            for i in range(1, 3):
                cv2.line(image, (w * i // 3, 0), (w * i // 3, h), (200, 200, 200), 1)
                cv2.line(image, (0, h * i // 3), (w, h * i // 3), (200, 200, 200), 1)

        if self.settings.get("crop_mode") == "Ręczne":
            cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), (255, 0, 0), 2)

        # Konwersja obrazu z BGR (OpenCV) do RGB (do PyQt)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # Utworzenie obiektu QImage z numpy array (RGB)
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Pobierz maksymalne rozmiary dostępnego ekranu (pomniejszone o marginesy na UI)
        screen_rect = QApplication.primaryScreen().availableGeometry()
        max_w, max_h = screen_rect.width() - 100, screen_rect.height() - 200  # zapas na UI

        # Zamień QImage na QPixmap
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)             # Wyświetl wynik w `video_label`
        # self.video_label.setFixedSize(pixmap.size())    # Dopasuj rozmiar labela do obrazu
        # self.adjustSize()                               # Dostosuj rozmiar całego okna do rozmiaru zawartości

    def close_app(self):
        if self.capture:
            self.capture.release()
        self.timer.stop()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    config_dialog = ConfigurationDialog()
    if config_dialog.exec() == QDialog.Accepted:
        settings = config_dialog.get_settings()

        if settings["crop_mode"] == "Ręczne":
            crop_dialog = CropSelectionDialog(camera_index=0)
            crop_dialog.exec()
            crop_rect = crop_dialog.get_crop_rect_scaled_to_camera()
            settings["crop_rect"] = crop_rect

        print("Wybrane ustawienia:", settings)

        window = FaceRecognitionApp()
        window.settings = settings  # PRZEKAZANIE KONFIGURACJI DO GŁÓWNEGO OKNA
        window.show()
        sys.exit(app.exec())
    else:
        print("Konfiguracja anulowana.")
        sys.exit()
