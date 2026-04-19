import tkinter as tk
from tkinter import messagebox
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# =========================
# LOAD + TRAIN MODEL
# =========================
data = fetch_california_housing(as_frame=True)
df = data.frame

selected_features = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup"
]

X = df[selected_features]
y = df["MedHouseVal"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# GUI SETUP
# =========================
root = tk.Tk()
root.title("California Housing Price Predictor")
root.geometry("520x450")
root.configure(bg="#f2f6ff")  # light background

FONT_TITLE = ("Helvetica", 18, "bold")
FONT_LABEL = ("Helvetica", 11)
FONT_ENTRY = ("Helvetica", 11)
FONT_BUTTON = ("Helvetica", 11, "bold")

entries = {}

# =========================
# TITLE
# =========================
title = tk.Label(
    root,
    text="Housing Price Predictor",
    font=FONT_TITLE,
    bg="#f2f6ff",
    fg="#1f3b73"
)
title.grid(row=0, column=0, columnspan=2, pady=15)

# =========================
# INPUT FIELDS
# =========================
for i, feature in enumerate(selected_features):
    tk.Label(
        root,
        text=feature,
        font=FONT_LABEL,
        bg="#f2f6ff",
        fg="#333"
    ).grid(row=i + 1, column=0, padx=15, pady=6, sticky="w")

    entry = tk.Entry(root, font=FONT_ENTRY, width=20, relief="solid", bd=1)
    entry.grid(row=i + 1, column=1, padx=15, pady=6)

    entries[feature] = entry

# =========================
# RESULT LABEL
# =========================
result_label = tk.Label(
    root,
    text="Predicted Price: ---",
    font=("Helvetica", 14, "bold"),
    bg="#f2f6ff",
    fg="#2e7d32"
)
result_label.grid(row=8, column=0, columnspan=2, pady=20)

# =========================
# FUNCTIONS
# =========================
def predict_price():
    try:
        values = []

        for feature in selected_features:
            value = entries[feature].get().strip()

            if value == "":
                raise ValueError(f"{feature} is empty")

            value = value.replace(",", "")
            values.append(float(value))

        input_array = np.array(values).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        price = prediction * 100000

        result_label.config(
            text=f"Predicted Price: ${price:,.2f}"
        )

    except Exception as e:
        messagebox.showerror("Input Error", str(e))


def reset_fields():
    for entry in entries.values():
        entry.delete(0, tk.END)

    result_label.config(text="Predicted Price: ---")

# =========================
# BUTTONS
# =========================
button_frame = tk.Frame(root, bg="#f2f6ff")
button_frame.grid(row=9, column=0, columnspan=2, pady=10)

predict_btn = tk.Button(
    button_frame,
    text="Predict Price",
    command=predict_price,
    bg="#1976d2",
    fg="white",
    font=FONT_BUTTON,
    width=15
)
predict_btn.grid(row=0, column=0, padx=10)

reset_btn = tk.Button(
    button_frame,
    text="Reset",
    command=reset_fields,
    bg="#d32f2f",
    fg="white",
    font=FONT_BUTTON,
    width=10
)
reset_btn.grid(row=0, column=1, padx=10)

# =========================
# RUN APP
# =========================
root.mainloop()