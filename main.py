from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.clock import Clock
import cv2
import numpy as np


class CarbonDetectorApp(App):
    def build(self):
        self.load_model()
        self.layout = BoxLayout(orientation='vertical')
        self.camera = Camera(play=True)
        self.label = Label(text="Carbon Emission: Scanning...")
        
        self.layout.add_widget(self.camera)
        self.layout.add_widget(self.label)
        
        Clock.schedule_interval(self.update, 1.0/30.0)  # 30 FPS
        
        return self.layout
    
    def update(self, dt):
        # This is where we'll process frames and update the UI
        pass
    def load_model(self):
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = net

    def detect_objects(self, frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids

    def draw_bounding_boxes(self, frame, boxes, confidences, class_ids):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    def update(self, dt):
        ret, frame = self.camera.texture.pixels
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            boxes, confidences, class_ids = self.detect_objects(frame)
            frame = self.draw_bounding_boxes(frame, boxes, confidences, class_ids)
            # Convert back to texture
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera.texture = texture
            detected_objects = [self.classes[class_id] for class_id in class_ids]
            emissions = [self.estimate_carbon_emission(obj) for obj in detected_objects]
            emission_text = ", ".join([f"{obj}: {emission}" for obj, emission in zip(detected_objects, emissions)])
            self.label.text = f"Carbon Emissions: {emission_text}"

    def estimate_carbon_emission(self, object_name):
        # This is a simplified estimation. In a real app, you'd use a more comprehensive database or API
        emissions = {
            "car": 120,  # gCO2e/km
            "bus": 80,   # gCO2e/km
            "bicycle": 0,
            "person": 0,
            "laptop": 52, # kgCO2e/year
            # Add more objects and their emission data
        }
        return emissions.get(object_name, "Unknown")

if __name__ == '__main__':
    CarbonDetectorApp().run()