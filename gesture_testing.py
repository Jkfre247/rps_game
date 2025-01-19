import pandas as pd
import tensorflow as tf
import mediapipe as mp
import cv2

# Wczytaj zapisane modele
model_raw = tf.keras.models.load_model("model_raw.h5")
model_normalized = tf.keras.models.load_model("model_normalized.h5")

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Funkcja do wyliczania odległości między dwoma punktami 3D
def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 +
            (point1.y - point2.y) ** 2 +
            (point1.z - point2.z) ** 2) ** 0.5

# Funkcja do zbierania odległości dla każdego palca
def extract_finger_distances(landmarks):
    """
    Zwraca słownik z odległościami TIP->MCP oraz TIP->Wrist
    dla każdego palca:
    - Kciuk (Thumb): 4-2, 4-0
    - Wskazujący (Index): 8-5, 8-0
    - Środkowy (Middle): 12-9, 12-0
    - Serdeczny (Ring): 16-13, 16-0
    - Mały (Pinky): 20-17, 20-0
    """
    fingers = {
        'Thumb': (4, 2),
        'Index': (8, 5),
        'Middle': (12, 9),
        'Ring': (16, 13),
        'Pinky': (20, 17)
    }

    distances = {}
    for finger, (tip, mcp) in fingers.items():
        distances[f'{finger}_TIP_MCP'] = calculate_distance(landmarks[tip], landmarks[mcp])
        distances[f'{finger}_TIP_Wrist'] = calculate_distance(landmarks[tip], landmarks[0])

    return distances

# Lista gestów w kolejności takiej samej, jak w trakcie treningu
gestures = ['fist', 'scissors', 'open']

# Otwórz kamerę
cap = cv2.VideoCapture(0)

print("Pokaż gest przed kamerą. Naciśnij 's', aby zrobić zdjęcie i wykonać predykcję. Naciśnij 'q', aby zakończyć.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się odczytać obrazu z kamery.")
        break

    # Konwersja BGR -> RGB dla MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Rysowanie punktów dłoni, jeśli wykryto
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS)

    # Wyświetl obraz z kamery
    cv2.imshow("Hand Gesture Recognition", frame)

    # Odczyt klawisza
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if result.multi_hand_landmarks:
            # Bierzemy pierwszą (jedyną) wykrytą dłoń
            hand_landmarks = result.multi_hand_landmarks[0]

            # Wyciągnij odległości dla każdego palca
            distances = extract_finger_distances(hand_landmarks.landmark)

            # Przygotuj dane do predykcji w formie DataFrame
            # -------------------------------------------------------------
            # Teraz mamy 10 kolumn: 5 palców x 2 odległości = 10 kluczy w 'distances'.
            # (np. 'Thumb_TIP_MCP', 'Thumb_TIP_Wrist', ..., 'Pinky_TIP_Wrist')
            # Zakładamy, że w CSV/treningu miałeś je w takiej kolejności/nazwach.
            # -------------------------------------------------------------
            data_raw = pd.DataFrame([distances])

            # Normalizacja (dzielenie przez Middle_TIP_Wrist)
            middle_tip_wrist = distances['Middle_TIP_Wrist']
            data_normalized = data_raw.div(middle_tip_wrist, axis=1)

            # UWAGA: Skoro teraz NIE usuwamy kolumny Middle_TIP_Wrist,
            # data_raw ma wciąż 10 kolumn, a data_normalized też 10,
            # tylko w normalizowanym Middle_TIP_Wrist = 1
            # i reszta kolumn też jest podzielona przez tę wartość.

            # Przewidywanie gestów
            pred_raw = model_raw.predict(data_raw, verbose=0)
            pred_normalized = model_normalized.predict(data_normalized, verbose=0)

            gesture_raw = gestures[pred_raw.argmax()]
            gesture_normalized = gestures[pred_normalized.argmax()]

            # Wyświetl wyniki
            print(f"Model bez normalizacji: {gesture_raw}")
            print(f"Model ze znormalizowanymi danymi: {gesture_normalized}")
        else:
            print("Nie wykryto dłoni. Spróbuj ponownie.")

    # Wyjście z pętli (np. klawisz 'q')
    if key == ord('q'):
        print("Zakończono działanie.")
        break

# Zwolnij zasoby
cap.release()
cv2.destroyAllWindows()
