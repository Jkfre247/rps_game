import time
import random
import pandas as pd
import tensorflow as tf
import mediapipe as mp
import cv2

# -------------------------------
# 1. Wczytanie zapisanych modeli
# -------------------------------
model_raw = tf.keras.models.load_model("model_raw.h5")
model_normalized = tf.keras.models.load_model("model_normalized.h5")

# Gesty w takiej samej kolejności, jak były trenowane:
gestures = ['fist', 'scissors', 'open']

# -------------------------------
# 2. Konfiguracja MediaPipe
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# -------------------------------
# 3. Funkcje pomocnicze
# -------------------------------

def calculate_distance(point1, point2):
    """Oblicz odległość euklidesową między dwoma punktami 3D."""
    return ((point1.x - point2.x) ** 2 +
            (point1.y - point2.y) ** 2 +
            (point1.z - point2.z) ** 2) ** 0.5

def extract_finger_distances(landmarks):
    """
    Zwraca słownik z odległościami TIP->MCP oraz TIP->Wrist
    dla każdego palca:
    - Kciuk (Thumb):   4-2, 4-0
    - Wskazujący (Index):   8-5, 8-0
    - Środkowy (Middle):  12-9, 12-0
    - Serdeczny (Ring):   16-13, 16-0
    - Mały (Pinky):   20-17, 20-0
    """
    fingers = {
        'Thumb':  (4, 2),
        'Index':  (8, 5),
        'Middle': (12, 9),
        'Ring':   (16, 13),
        'Pinky':  (20, 17)
    }

    distances = {}
    for finger, (tip, mcp) in fingers.items():
        distances[f'{finger}_TIP_MCP']   = calculate_distance(landmarks[tip], landmarks[mcp])
        distances[f'{finger}_TIP_Wrist'] = calculate_distance(landmarks[tip], landmarks[0])
    return distances

def determine_winner(user_gesture, ai_gesture):
    """
    Określa zwycięzcę w papier-kamień-nożyce:
    - 'fist' = kamień
    - 'scissors' = nożyce
    - 'open' = papier
    Zwraca 'user', 'ai' lub 'tie'.
    """
    if user_gesture == ai_gesture:
        return "tie"

    if user_gesture == 'fist':
        # fist vs scissors -> user wygrywa
        # fist vs open -> AI wygrywa
        return "user" if ai_gesture == 'scissors' else "ai"

    if user_gesture == 'scissors':
        # scissors vs open -> user wygrywa
        # scissors vs fist -> AI wygrywa
        return "user" if ai_gesture == 'open' else "ai"

    if user_gesture == 'open':
        # open vs fist -> user wygrywa
        # open vs scissors -> AI wygrywa
        return "user" if ai_gesture == 'fist' else "ai"

    return "tie"  # w razie błędu

def put_centered_text(frame, text, color=(255, 255, 255), scale=2, thickness=2):
    """
    Wstawia tekst na środku klatki (obrazu).
    Domyślnie kolor biały, rozmiar 2, grubość 2.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    # oblicz środek ramki
    frame_h, frame_w, _ = frame.shape
    x = (frame_w - text_w) // 2
    y = (frame_h + text_h) // 2
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

# -------------------------------
# 4. Uruchomienie kamery
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Nie udało się otworzyć kamery.")
    exit()

print("Gra papier-kamień-nożyce startuje!")

# -------------------------------
# 5. Definicja "maszyny stanów"
# -------------------------------
game_state = 'countdown'  # 'countdown' lub 'show_result'

countdown_duration = 15  # sekundy
start_time = time.time()
user_gesture_raw = None
user_gesture_normalized = None
final_user_gesture = None
ai_gesture = None
winner = None

# -------------------------------
# 6. Pętla główna
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się odczytać obrazu z kamery.")
        break

    # Konwersja BGR -> RGB (dla MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # -----------------------------
    # STAN 1: COUNTDOWN
    # -----------------------------
    if game_state == 'countdown':
        elapsed_time = time.time() - start_time
        remaining_time = int(countdown_duration - elapsed_time)

        # Rozpoznawanie dłoni / gestu (tylko gdy mamy czas)
        if remaining_time >= 0:
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                # Bierzemy pierwszą wykrytą dłoń
                hand_landmarks = result.multi_hand_landmarks[0]
                distances = extract_finger_distances(hand_landmarks.landmark)

                # Dane w formie DataFrame (1 wiersz, 10 kolumn)
                data_raw = pd.DataFrame([distances])
                # Dane znormalizowane (dzielenie przez Middle_TIP_Wrist)
                middle_tip_wrist = distances['Middle_TIP_Wrist']
                data_normalized = data_raw.div(middle_tip_wrist, axis=1)

                # Przewidywanie
                pred_raw = model_raw.predict(data_raw, verbose=0)
                pred_normalized = model_normalized.predict(data_normalized, verbose=0)

                user_gesture_raw = gestures[pred_raw.argmax()]
                user_gesture_normalized = gestures[pred_normalized.argmax()]

            # Wyświetl odliczanie w **białym** kolorze
            countdown_text = f"Czas: {remaining_time}s"
            cv2.putText(frame, countdown_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Wyświetl aktualnie rozpoznany gest
            if user_gesture_raw:
                cv2.putText(
                    frame,
                    f"Gest: RAW={user_gesture_raw}, NORM={user_gesture_normalized}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
        else:
            # Czas się skończył -> ustal wynik
            if user_gesture_raw is not None:
                final_user_gesture = user_gesture_raw  # Możesz zmienić na user_gesture_normalized
                ai_gesture = random.choice(gestures)
                # Kto wygrał?
                outcome = determine_winner(final_user_gesture, ai_gesture)
                winner = outcome  # 'user', 'ai' lub 'tie'
            else:
                # Nie udało się rozpoznać gestu w czasie
                winner = None
            # Przejdź do stanu wyświetlania wyniku
            game_state = 'show_result'

    # -----------------------------
    # STAN 2: SHOW_RESULT
    # -----------------------------
    elif game_state == 'show_result':
        # Rysuj wynik na środku
        if winner is None:
            # Brak rozpoznanego gestu
            put_centered_text(frame, "Nie rozpoznano gestu!", color=(0, 0, 255), scale=1.5)
        else:
            # Tekst: "Wygrana", "Przegrana" albo "Remis" w określonym kolorze
            if winner == 'tie':
                text = "REMIS!"
                text_color = (0, 255, 255)  # żółty
            elif winner == 'user':
                text = "WYGRANA!"
                text_color = (0, 255, 0)  # zielony
            else:
                text = "PRZEGRANA!"
                text_color = (0, 0, 255)  # czerwony
            put_centered_text(frame, text, color=text_color, scale=2, thickness=3)

            # Wyświetl także obok, co pokazał użytkownik i co pokazał AI
            user_info = f"Ty: {final_user_gesture}"
            ai_info = f"AI: {ai_gesture}"

            cv2.putText(frame, user_info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, ai_info, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Instrukcja: "S - zagraj ponownie, Q - wyjdz"
        instruction_text = "Wcisnij [s] aby zagrac ponownie, [q] aby wyjsc"
        cv2.putText(frame, instruction_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # -----------------------------
    # 7. Wyświetl podgląd
    # -----------------------------
    cv2.imshow("Papier-Kamien-Nozyce", frame)

    # Odczyt klawisza
    key = cv2.waitKey(1) & 0xFF

    # Globalny 'q' - niezależnie od stanu, wychodzimy
    if key == ord('q'):
        print("Zamykanie gry...")
        break

    # Jeśli jesteśmy w stanie SHOW_RESULT, reaguj na 's' (ponowna gra)
    if game_state == 'show_result' and key == ord('s'):
        # Resetujemy zmienne, wracamy do countdown
        start_time = time.time()
        user_gesture_raw = None
        user_gesture_normalized = None
        final_user_gesture = None
        ai_gesture = None
        winner = None
        game_state = 'countdown'

# Po wyjsciu z pętli zwolnij zasoby
cap.release()
cv2.destroyAllWindows()
