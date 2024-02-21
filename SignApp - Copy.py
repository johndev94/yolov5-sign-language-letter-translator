import tkinter as tk
from tkinter import filedialog, messagebox, Radiobutton
from PIL import Image, ImageTk
import cv2
import torch

class SignLanguageApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Dictionary to hold model paths for each sign language
        self.model_paths = {
            'ISL': 'c:/Users/John/Desktop/Coding Projects/Python/New Yolov5 test/yolov5/runs/train/letter_hand_gestures_fourth7/weights/best.pt',
            'ASL': 'c:/Users/John/Desktop/Coding Projects/Python/New Yolov5 test/yolov5/runs/train/letter_hand_gestures_sixth3/weights/best.pt',
            'BSL': 'c:/Users/John/Desktop/Coding Projects/Python/New Yolov5 test/yolov5/runs/train/letter_hand_gestures_fourth7/weights/best.pt'
        }
        
        # Variable to store the selected sign language
        self.selected_language = tk.StringVar(value='ISL')  # Default selection
        
        # Create radio buttons for ISL, ASL, BSL
        self.create_radio_buttons()
        
        # Create a button for uploading and processing videos
        self.upload_button = tk.Button(window, text="Upload Video", command=self.upload_video)
        self.upload_button.pack()

        # Label for displaying the results
        self.result_label = tk.Label(window, text="Select a sign language and upload a video for detection.")
        self.result_label.pack()

    def create_radio_buttons(self):
        # Create and pack the radio buttons
        for language in ['ISL', 'ASL', 'BSL']:
            Radiobutton(self.window, text=language, variable=self.selected_language, value=language, command=self.load_model).pack()
        
        # Initially load the default model
        self.load_model()

    def load_model(self):
        # Load the model based on the selected sign language
        selected_model_path = self.model_paths[self.selected_language.get()]
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=selected_model_path)  # Load model
        print(f"Loaded model for {self.selected_language.get()}")

    def upload_video(self):
        # Open a dialog to select a video file
        file_path = filedialog.askopenfilename()
        if file_path:  # If a file is selected
            self.detect_gestures_in_video(file_path)

    def detect_gestures_in_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return

        # Create a window for displaying the video frames
        video_window = tk.Toplevel(self.window)
        video_window.title("Video Playback")
        video_label = tk.Label(video_window)
        video_label.pack()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for YOLOv5 and back to BGR for OpenCV display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame_rgb)

            # Convert back to BGR for OpenCV operations
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw bounding boxes and labels on the frame
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > 0.5:  # Filter detections with confidence > 0.5
                    label = f'{self.model.names[int(cls)]} {conf:.2f}'
                    start_point = (int(xyxy[0]), int(xyxy[1]))
                    end_point = (int(xyxy[2]), int(xyxy[3]))
                    color = (0, 255, 0)  # Green for detected gestures
                    frame_bgr = cv2.rectangle(frame_bgr, start_point, end_point, color, 2)
                    frame_bgr = cv2.putText(frame_bgr, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert the frame to RGB before displaying in Tkinter
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            video_label.update()

        cap.release()  # Release the video capture object

        # Close the video window once the video is done playing
        video_window.destroy()

        # Update the GUI to indicate the video processing is complete
        self.result_label.config(text="Detection completed.")


    def run(self):
        self.window.mainloop()

# Create a window and pass it to the SignLanguageApp class
root = tk.Tk()
app = SignLanguageApp(root, "Sign Language Detection App")
app.run()
