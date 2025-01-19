import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Wczytaj dane
fist_data = pd.read_csv("fist_data.csv")
scissors_data = pd.read_csv("open_data.csv")
open_data = pd.read_csv("scissors_data.csv")

# Połącz wszystkie zestawy w jeden DataFrame
data = pd.concat([fist_data, scissors_data, open_data], ignore_index=True)

# 2. Przygotowanie danych
# Oddziel cechy (X) od etykiet (y)
# -----------------------------------------------------
# W tym momencie X zawiera 10 kolumn (bo w CSV mamy 11: Gesture + 10 odległości).
# -----------------------------------------------------
X = data.drop(columns=["Gesture"])
y = data["Gesture"]

# Zakoduj etykiety do postaci numerycznej
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. Normalizacja danych
# -----------------------------------------------------
# Tutaj NIE usuwamy Middle_TIP_Wrist. Dalej go zostawiamy, ale
# cały DataFrame (X_train_normalized) dzielimy przez tę kolumnę.
# Tym samym Middle_TIP_Wrist w znormalizowanym DF będzie równe 1.
# -----------------------------------------------------
X_train_normalized = X_train.div(X_train["Middle_TIP_Wrist"], axis=0)
X_test_normalized = X_test.div(X_test["Middle_TIP_Wrist"], axis=0)

# 4. Budowa modeli
# Liczba wejść (cech) = 10
input_shape = X_train.shape[1]  # powinno być 10

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Model bez normalizacji
model_raw = build_model()
history_raw = model_raw.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30, batch_size=16, verbose=1
)

# Model ze znormalizowanymi danymi
model_normalized = build_model()
history_normalized = model_normalized.fit(
    X_train_normalized, y_train,
    validation_data=(X_test_normalized, y_test),
    epochs=30, batch_size=16, verbose=1
)

# 5. Wyniki
print("Model bez normalizacji")
loss_raw, accuracy_raw = model_raw.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {loss_raw}, Accuracy: {accuracy_raw}")

print("\nModel ze znormalizowanymi danymi")
loss_normalized, accuracy_normalized = model_normalized.evaluate(X_test_normalized, y_test, verbose=0)
print(f"Loss: {loss_normalized}, Accuracy: {accuracy_normalized}")

# 6. Zapis modeli
model_raw.save("model_raw.h5")
model_normalized.save("model_normalized.h5")
