import tkinter as tk
from tkinter import filedialog, messagebox, Radiobutton, Label, font as tkFont
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import os

class SignLanguageApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        # Dynamically construct the model path relative to the script's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, 'yolov5', 'runs', 'train', 'letter_hand_gestures_fourth7', 'weights', 'best.pt')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        
        self.input_dialect = tk.StringVar(value='ASL')
        self.output_language = tk.StringVar(value='ISL')
        
        self.create_dialect_radio_buttons()
        self.create_language_radio_buttons()
        
        self.upload_button = tk.Button(window, text="Upload Image", command=self.upload_picture)
        self.upload_button.pack()
        
        self.result_label = tk.Label(window, text="Upload an image for detection.")
        self.result_label.pack()

        self.letter_font = tkFont.Font(size=24, weight="bold")
        self.detected_letter_label = tk.Label(window, text="", font=self.letter_font, fg="red")
        self.detected_letter_label.pack()

        self.translate_button = tk.Button(window, text="Translate Letter", command=self.translate_letter)
        self.translate_button.pack()
        self.translate_button.pack_forget()

        self.restart_button = tk.Button(window, text="Restart", command=self.restart_app)
        self.restart_button.pack()
        
        self.original_label = None
        self.translated_label = None
        
        self.original_image_path = None
        self.detected_letter = None

    def create_dialect_radio_buttons(self):
        Label(self.window, text="Select Input Dialect:").pack()
        for dialect in ['ASL', 'BSL', 'ISL']:
            Radiobutton(self.window, text=dialect, variable=self.input_dialect, value=dialect).pack()

    def create_language_radio_buttons(self):
        Label(self.window, text="Select Output Language:").pack()
        for language in ['BSL', 'ISL']:
            Radiobutton(self.window, text=language, variable=self.output_language, value=language).pack()

    def upload_picture(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.original_image_path = file_path
            self.detect_sign_language(file_path)

    def detect_sign_language(self, file_path):
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb)
        if len(results.xyxy[0]) > 0:
            detected_class_id = int(results.xyxy[0][0, 5])
            self.detected_letter = self.model.names[detected_class_id]
            self.display_original_image(file_path)
            self.detected_letter_label.config(text=f"Detected Letter: {self.detected_letter.upper()}")
            self.translate_button.pack()  # Show the translate button
        else:
            messagebox.showerror("Error", "No sign language gesture detected in the image.")

    def display_original_image(self, file_path):
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize the image to fit a predefined space
        photo = ImageTk.PhotoImage(img)
        
        if self.original_label is None:
            self.original_label = Label(self.window, image=photo)
            self.original_label.image = photo
            self.original_label.pack(side="left")
        else:
            self.original_label.configure(image=photo)
            self.original_label.image = photo

    def translate_letter(self):
        if self.detected_letter:
            # Dynamically construct the translated image path
            base_dir = os.path.dirname(os.path.abspath(__file__))
            translated_image_path = os.path.join(base_dir, 'yolov5', 'letter images', self.output_language.get(), f'{self.detected_letter.upper()}.jpg')
            
            self.display_translated_image(translated_image_path)

    def display_translated_image(self, translated_image_path):
        img = Image.open(translated_image_path)
        img.thumbnail((300, 300))  # Resize the image to fit a predefined space
        photo = ImageTk.PhotoImage(img)
        
        if self.translated_label is None:
            self.translated_label = Label(self.window, image=photo)
            self.translated_label.image = photo
            self.translated_label.pack(side="right")
        else:
            self.translated_label.configure(image=photo)
            self.translated_label.image = photo

        self.result_label.config(text="Translation completed.")

    def restart_app(self):
        if self.original_label is not None:
            self.original_label.pack_forget()
            self.original_label = None
        if self.translated_label is not None:
            self.translated_label.pack_forget()
            self.translated_label = None
        
        self.detected_letter_label.config(text="")
        self.translate_button.pack_forget()
        self.result_label.config(text="Upload an image for detection.")

    def run(self):
        self.window.mainloop()

root = tk.Tk()
app = SignLanguageApp(root, "Sign Language Translation App")
app.run()
