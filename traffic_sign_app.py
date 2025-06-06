import tkinter as tk
from tkinter import Label, Button, PhotoImage, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import time
import os
import csv
import winsound

# Load model
model = load_model('traffic_sign_model_cv2.h5')
IMG_SIZE = 32

# Class names
class_names = {

    0: 'speed limit 20',
    1: 'speed limit 30',
    2: 'speed limit 50',
    3: 'speed limit 60',
    4: 'speed limit 70',
    5: 'speed limit 80',
    6: 'end of speed limit 80',
    7: 'speed limit 100',
    8: 'speed limit 120',
    9: 'no overtaking allowed',
    10: 'no overtaking for trucks',
    11: 'T-intersection ahead',
    12: 'right of way',
    13: 'give way',
    14: 'stop',
    15: 'no entry',
    16: 'no entry for trucks',
    17: 'no entry alternate',
    18: 'hazard ahead',
    19: 'left turn ahead',
    20: 'right turn ahead',
    21: 'double curve ahead',
    22: 'speed breaker ahead',
    23: 'slippery road',
    24: 'narrowing road ahead',
    25: 'work ahead',
    26: 'traffic lights ahead',
    27: 'pedestrian crossing',
    28: 'watch out for children',
    29: 'caution cyclist',
    30: 'ice on road',
    31: 'wild animal crossing',
    32: 'no entry duplicate',
    33: 'turn right',
    34: 'turn left',
    35: 'straight ahead',
    36: 'right intersection',
    37: 'Go Straight or Left',
    38: 'right directional indicator',
    39: 'left directional indicator',
    40: 'ring road',
    41: 'no overtaking',
    42: 'no truck overtaking'


}

# Init CSV log file
log_file = 'prediction_log.csv'
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'Prediction', 'Confidence'])

# Splash screen
# Splash screen with fade-in effect
splash = tk.Tk()
splash.overrideredirect(True)
splash.geometry("500x300+500+250")
splash.configure(bg='black')

# Splash content
label = Label(splash, text="Traffic Sign Recognition", font=("Helvetica", 24, "bold"), fg="red", bg="black")
label.pack(expand=True)

# Set initial opacity to 0 (fully transparent)
splash.attributes("-alpha", 0.0)

# Fade-in loop
for i in range(0, 11):
    splash.attributes("-alpha", i / 10)
    splash.update()
    time.sleep(0.1)

time.sleep(2)  # Keep visible after fade-in
splash.destroy()


# Main window
root = tk.Tk()
root.title("Traffic Sign Recognition")
root.geometry("900x700")
root.configure(bg='black')
root.resizable(False, False)

# Layout
frame = tk.Frame(root, bg='black')
frame.pack(pady=10)

# Logo
try:
    logo_img = Image.open("logo.jpg")
    logo_img = logo_img.resize((80, 80))
    logo_imgtk = ImageTk.PhotoImage(logo_img)
    logo_label = Label(root, image=logo_imgtk, bg="black")
    logo_label.place(x=20, y=10)
except:
    pass

label = Label(frame, bg='black')
label.pack()

result_label = Label(root, text="", font=("Helvetica", 20), bg="black", fg="red")
result_label.pack(pady=20)

cap = cv2.VideoCapture(0)
running = False
frame_data = None
screenshot_count = 0

# Detection function
def detect_frame():
    global frame_data
    if not running:
        return
    ret, frame = cap.read()
    if ret:
        frame_data = frame.copy()
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed = rgb.astype('float32') / 255.0
        processed = np.expand_dims(processed, axis=0)

        prediction = model.predict(processed)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)
        label_text = f"{class_names[class_id]} ({confidence*100:.1f}%)"

        # Beep and log if label changes
        if result_label.cget("text") != label_text:
            winsound.Beep(1000, 150)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([timestamp, class_names[class_id], f"{confidence*100:.1f}%"])

        result_label.config(text=label_text)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    root.after(10, detect_frame)

# Start button function
def start_detection():
    global running
    if not running:
        result_label.config(text="Starting...", fg="lightgreen")
        root.after(500, lambda: result_label.config(text=""))
        running = True
        detect_frame()

# Stop button function
def stop_detection():
    global running
    if running:
        running = False
        result_label.config(text="Stopped.", fg="orange")
        label.configure(image='')

# Screenshot capture
def capture_screenshot():
    global screenshot_count
    if frame_data is not None:
        screenshot_count += 1
        filename = f"screenshot_{screenshot_count}.png"
        cv2.imwrite(filename, frame_data)
        os.startfile(os.getcwd())
        result_label.config(text=f"Saved: {filename}", fg="lightblue")

# Exit confirm
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to exit?"):
        root.destroy()
        cap.release()
        cv2.destroyAllWindows()

# Buttons frame
btn_frame = tk.Frame(root, bg='black')
btn_frame.pack(pady=10)

Button(btn_frame, text="Start Recognition", font=("Helvetica", 14), bg="red", fg="white",
       activebackground="#cc0000", padx=20, pady=10, command=start_detection).pack(pady=5)

Button(btn_frame, text="Stop Recognition", font=("Helvetica", 14), bg="red", fg="white",
       activebackground="#cc0000", padx=20, pady=10, command=stop_detection).pack(pady=5)

Button(btn_frame, text="Capture Screenshot", font=("Helvetica", 14), bg="red", fg="white",
       activebackground="#cc0000", padx=20, pady=10, command=capture_screenshot).pack(pady=5)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()


