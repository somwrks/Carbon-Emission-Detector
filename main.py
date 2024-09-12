import torch
import torchvision
from torchvision import transforms
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Load the emissions data
emissions_data = pd.read_csv('data/CO2_Emissions_Canada.csv')

def get_emissions(vehicle_class):
    return emissions_data[emissions_data['Vehicle Class'] == vehicle_class]['CO2 Emissions(g/km)'].mean()

def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    model.eval()
    return model

class CarDetectorApp(App):
    def build(self):
        self.model = load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.layout = BoxLayout(orientation='vertical')
        self.camera = Camera(play=True, resolution=(640, 480))
        self.label = Label(text="Car Detection: Scanning...")
        self.layout.add_widget(self.camera)
        self.layout.add_widget(self.label)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.layout

    def get_frame(self):
        pixels = self.camera.texture.pixels
        frame = np.frombuffer(pixels, dtype=np.uint8)
        frame = frame.reshape(self.camera.texture.height, self.camera.texture.width, 4)
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    def detect_cars(self, frame):
        image = Image.fromarray(frame)
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        car_detections = predictions[0]['boxes'][predictions[0]['labels'] == 3]
        car_scores = predictions[0]['scores'][predictions[0]['labels'] == 3]
        
        threshold = 0.5
        car_detections = car_detections[car_scores > threshold]
        
        return car_detections.cpu().numpy()

    def estimate_car_size(self, box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        if area < 20000:
            return "COMPACT"
        elif area < 40000:
            return "MID-SIZE"
        else:
            return "SUV - SMALL"

    def update(self, dt):
        if self.camera.texture:
            frame = self.get_frame()
            detections = self.detect_cars(frame)
            if len(detections) > 0:
                car = detections[0]
                vehicle_class = self.estimate_car_size(car)
                emissions = get_emissions(vehicle_class)
                self.label.text = f"Detected: {vehicle_class}\nEstimated CO2 Emissions: {emissions:.2f} g/km"
            else:
                self.label.text = "No cars detected"
            self.display_frame(frame, detections)

    def display_frame(self, frame, detections):
        for box in detections:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.camera.texture = texture

if __name__ == "__main__":
    CarDetectorApp().run()