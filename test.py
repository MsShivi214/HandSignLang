import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import tensorflow as tf
import os

# Custom DepthwiseConv2D layer to handle compatibility
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Custom TFLite model loader class
class CustomClassifier:
    def __init__(self, model_path, labels_path):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        self.load_model(model_path)
        self.load_labels(labels_path)

    def load_model(self, model_path):
        # Convert h5 to TFLite format
        try:
            # Load model with custom objects
            custom_objects = {
                'DepthwiseConv2D': CustomDepthwiseConv2D
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            # Save the TFLite model
            tflite_path = model_path.replace('.h5', '.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def load_labels(self, labels_path):
        try:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading labels: {str(e)}")
            raise

    def getPrediction(self, img):
        try:
            # Preprocess the image
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get the output tensor
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process the results
            prediction = output_data[0]
            index = np.argmax(prediction)
            
            return prediction, index
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None

# Check if model files exist
model_path = "Model/keras_model.h5"
labels_path = "Model/labels.txt"

if not os.path.exists(model_path) or not os.path.exists(labels_path):
    print(f"Error: Model files not found. Please ensure {model_path} and {labels_path} exist.")
    exit()

# Initialize camera and detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

try:
    print("Loading model...")
    classifier = CustomClassifier(model_path, labels_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error initializing classifier: {str(e)}")
    exit()

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

labels = ["A", "B", "C"]

print("Starting hand gesture recognition...")
print("Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        continue
        
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        try:
            imgCropShape = imgCrop.shape
            aspectRation = h / w

            if aspectRation > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop,(wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Make prediction using the processed white image
            prediction, index = classifier.getPrediction(imgWhite)
            if prediction is not None and index is not None:
                confidence = prediction[index]
                if confidence > 0.5:  # Only show predictions with confidence > 50%
                    print(f"Predicted: {labels[index]} with confidence: {confidence:.2f}")
                    # Display the prediction on the image
                    cv2.putText(img, f"{labels[index]} {confidence:.2f}", (x, y-20), 
                              cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print(f"Error processing hand: {str(e)}")
            continue

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()


