# System Rozpoznawania Twarzy z Analizą Efektu Starzenia

Projekt badawczo-rozwojowy z zakresu przetwarzania obrazów i biometrii twarzy. System umożliwia rozpoznawanie twarzy w czasie rzeczywistym oraz analizę wpływu procesu starzenia na skuteczność identyfikacji.

## 🔍 Opis projektu

System wykorzystuje framework **FaceX-Zoo** do wykrywania i identyfikacji twarzy. Główną innowacją projektu jest analiza skuteczności rozpoznawania osób na podstawie ich zdjęć z różnych okresów życia – zarówno z zastosowaniem sztucznego starzenia (FaceApp), jak i rzeczywistych zdjęć z bazy AgeDB.

System wyposażono w interfejs graficzny (GUI) pozwalający na konfigurację parametrów, dodawanie nowych twarzy do bazy i wizualizację wyników w czasie rzeczywistym.

## ⚙️ Technologie

- **Python 3.10+**
- **FaceX-Zoo**
- **OpenCV**
- **PyTorch**
- **PySide6**
- **NumPy**, **scikit-learn**

## 🧠 Główne funkcje

- Wykrywanie i rozpoznawanie twarzy w czasie rzeczywistym
- Obsługa kamer USB i strumieni RTSP
- Porównywanie cech twarzy z uwzględnieniem zmian wiekowych
- Analiza skuteczności na bazie AgeDB i danych z FaceApp
- Eksport statystyk i raportów w formacie CSV

## 🧪 Wnioski z badań

- Skuteczność dla sztucznie postarzonych zdjęć: **100%**
- Skuteczność dla naturalnych zdjęć starszych osób: **44%** (poprawiona do **70%** po aktualizacji modelu)
- System jest w stanie rozpoznawać osoby z różnicą wieku do 40 lat względem wzorców

## 🚀 Instalacja i uruchomienie

### Wymagania systemowe

- Linux/Windows
- Karta graficzna NVIDIA z obsługą CUDA (zalecana)

### Instalacja

```bash
pip install PySide6 opencv-python numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Uruchomienie systemu

```bash
cd FaceX-Zoo-main/face_sdk
python live_usage/gui_face_recognition.py
```

### Dodawanie nowych twarzy

1. Utwórz nowy folder w `faces_db` o nazwie danej osoby (np. `Jan_Kowalski`)
2. Umieść w nim zdjęcia twarzy (JPG/PNG) – najlepiej frontalne, dobrze oświetlone

## 📊 Przykładowe wyniki

| Warunki              | Skuteczność | Średnie podobieństwo |
|----------------------|-------------|-----------------------|
| Warunki normalne     | 100%        | 0.82                  |
| FaceApp (sztuczne)   | 100%        | 0.75                  |
| AgeDB (naturalne)    | 70% (po ulepszeniu) | 0.41          |

## 📁 Dokumentacja

Pełna dokumentacja projektu znajduje się w pliku [`RiPO_Dokumentacja.pdf`](./RiPO_Dokumentacja.pdf) w repozytorium.

## 👥 Autorzy

- **Yustyna Sukhorab**  
- **Jakub Warczyk** 

---

> _Projekt zrealizowany na Politechnice Wrocławskiej w ramach kursu "Rozpoznawanie i Przetwarzanie Obrazów", 2025._
