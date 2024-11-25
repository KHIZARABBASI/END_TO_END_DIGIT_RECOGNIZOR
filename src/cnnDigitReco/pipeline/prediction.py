from tensorflow.keras.models import load_model
import cv2
import numpy as np

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename



    def predict(self):
        # Load the trained model
        model = load_model(r'artifacts\training\model.h5')

        # Load the image
        img = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)

        # Resize the image
        img_resized = cv2.resize(img, (28, 28))

        # Normalize the image
        img_normalized = img_resized.astype('float32') / 255.0

        # Reshape the image for the model
        img_reshaped = img_normalized.reshape((1, 28, 28, 1))

        # Make a prediction
        prediction = model.predict(img_reshaped)

        # Return the predicted digit
        return np.argmax(prediction)
    
    # Example usage
filename = r'research\notebook\num.jpg'
pipeline = PredictionPipeline(filename)
predicted_digit = pipeline.predict()
print(f"Predicted Digit: {predicted_digit}")

# import tkinter as tk
# from tkinter import messagebox
# import numpy as np
# import cv2
# from PIL import Image, ImageGrab
# from tensorflow.keras.models import load_model

# class DigitRecognizerApp:
#     def __init__(self, root, model_path):
#         self.root = root
#         self.root.title("Digit Recognizer")
#         self.root.geometry("400x500")

#         # Load the trained model
#         self.model = load_model(model_path)

#         # Create canvas for drawing
#         self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white", cursor="cross")
#         self.canvas.pack(pady=20)

#         # Buttons for actions
#         self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas, bg="red", fg="white")
#         self.clear_button.pack(side=tk.LEFT, padx=10)

#         self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_digit, bg="green", fg="white")
#         self.predict_button.pack(side=tk.RIGHT, padx=10)

#         # Bind mouse events for drawing
#         self.canvas.bind("<B1-Motion>", self.paint)

#     def paint(self, event):
#         x, y = event.x, event.y
#         self.canvas.create_oval((x - 8, y - 8, x + 8, y + 8), fill="black", outline="black")

#     def clear_canvas(self):
#         self.canvas.delete("all")

#     def predict_digit(self):
#         # Save the canvas content as an image
#         x = self.root.winfo_rootx() + self.canvas.winfo_x()
#         y = self.root.winfo_rooty() + self.canvas.winfo_y()
#         x1 = x + self.canvas.winfo_width()
#         y1 = y + self.canvas.winfo_height()

#         # Capture canvas and preprocess
#         img = ImageGrab.grab(bbox=(x, y, x1, y1)).convert("L")  # Convert to grayscale
#         img = img.resize((28, 28))  # Resize to 28x28
#         img = np.array(img)
#         img = img.astype('float32') / 255.0  # Normalize
#         img = img.reshape(1, 28, 28, 1)  # Reshape for model

#         # Predict
#         prediction = self.model.predict(img)
#         digit = np.argmax(prediction)

#         # Display the prediction
#         messagebox.showinfo("Prediction", f"Predicted Digit: {digit}")

# # Run the app
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = DigitRecognizerApp(root, model_path="artifacts/training/model.h5")
#     root.mainloop()
