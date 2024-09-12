import os
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class CarbonDetectorApp(App):
    def build(self):
        try:
            self.load_model()
            self.layout = BoxLayout(orientation='vertical')
            self.img = Image()
            self.label = Label(text="Carbon Emission: Scanning...", size_hint=(1, 0.1))
            self.layout.add_widget(self.img)
            self.layout.add_widget(self.label)
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise IOError("Cannot open webcam")
            Clock.schedule_interval(self.update, 1.0 / 30.0)
            return self.layout
        except Exception as e:
            print(f"Error in build method: {str(e)}")
            return Label(text=f"Error: {str(e)}")

    def load_model(self):
        try:
            model_path = os.path.dirname(os.path.abspath(__file__))
            self.net = cv2.dnn.readNet(
                os.path.join(model_path, "yolov3.weights"),
                os.path.join(model_path, "yolov3.cfg")
            )
            with open(os.path.join(model_path, "coco.names"), "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def detect_objects(self, frame):
        try:
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids, confidences, boxes = [], [], []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and self.classes[class_id] in self.vehicle_classes:
                        center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                        w, h = int(detection[2] * width), int(detection[3] * height)
                        x, y = int(center_x - w / 2), int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            return [boxes[i] for i in indexes], [confidences[i] for i in indexes], [class_ids[i] for i in indexes]
        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            return [], [], []

    def estimate_carbon_emission(self, vehicle_type):
        vehicle_emissions = {
            "car": 180,
            "truck": 250,
            "bus": 300,
            "motorcycle": 100,
            "bicycle": 0
        }
        return vehicle_emissions.get(vehicle_type.lower(), "Unknown")

    def update(self, dt):
        try:
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to grab frame")
                return

            boxes, confidences, class_ids = self.detect_objects(frame)
            
            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detected_objects = [self.classes[class_id] for class_id in class_ids]
            emissions = [self.estimate_carbon_emission(obj) for obj in detected_objects]
            emission_text = ", ".join([f"{obj}: {emission} gCO2/km" for obj, emission in zip(detected_objects, emissions)])
            self.label.text = f"Vehicle Type & Emissions: {emission_text}" if emission_text else "No vehicles detected"

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture
        except Exception as e:
            print(f"Error in update method: {str(e)}")

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    CarbonDetectorApp().run()
