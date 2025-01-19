import mediapipe as mp
import cv2
import pandas as pd

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


# Przygotowanie DataFrame do zapisu
# Kolumny to odległości dla każdego palca
columns = ['Gesture'] + [
    f'{finger}_{relation}'
    for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    for relation in ['TIP_MCP', 'TIP_Wrist']
]
data = pd.DataFrame(columns=columns)

# Kamera
cap = cv2.VideoCapture(0)

# Pobierz nazwę gestu
gesture_label = input("Podaj etykietę gestu (np. 'fist', 'open', 'scissors'): ")

# Maksymalna liczba próbek (zdjęć) do zebrania
num_samples = 40
samples_collected = 0

print(f"Ustaw gest '{gesture_label}' i naciskaj klawisz 's', aby zapisać kolejne próbki.")

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

    # Wyświetl licznik w lewym górnym rogu (np. 5/40)
    cv2.putText(frame,
                f"ZDJECIA: {samples_collected}/{num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2)

    # Pokaz okno
    cv2.imshow("Hand Tracking - Zbieranie danych", frame)

    # Odczyt klawisza
    key = cv2.waitKey(1) & 0xFF

    # Jeśli wciśnięto 's' - próbujemy zapisać próbkę
    if key == ord('s'):
        if result.multi_hand_landmarks:
            # Bierzemy pierwszą (jedyną) wykrytą dłoń
            hand_landmarks = result.multi_hand_landmarks[0]

            # Wyciągnij interesujące odległości dla każdego palca
            distances = extract_finger_distances(hand_landmarks.landmark)

            # Dodaj do DataFrame
            # UWAGA: używamy .loc[len(data)] zamiast .append() (które jest deprecated w nowszych Pandach)
            data.loc[len(data)] = {
                'Gesture': gesture_label,
                **distances
            }

            samples_collected += 1
            print(f"Zebrano próbkę: {samples_collected}/{num_samples}")

            # Sprawdź, czy mamy już 40 próbek
            if samples_collected >= num_samples:
                print(f"Zebrano wszystkie próbki dla gestu '{gesture_label}'.")
                break
        else:
            print("Nie wykryto dłoni. Spróbuj ponownie.")

    # Wyjście z pętli (np. klawisz 'q')
    if key == ord('q'):
        print("Przerwano zbieranie próbek.")
        break

# Zwolnij zasoby
cap.release()
cv2.destroyAllWindows()

# Eksport danych do CSV
output_file = f"{gesture_label}_data.csv"
data.to_csv(output_file, index=False)
print(f"Dane zapisane do pliku: {output_file}")

# Podgląd danych w konsoli
print(data)
