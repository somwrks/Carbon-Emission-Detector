import json
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

# Load the CSV data
emissions_data = pd.read_csv('data/CO2_Emissions_Canada.csv')

def get_emissions(vehicle_class):
    matching_rows = emissions_data[emissions_data['Vehicle Class'] == vehicle_class]
    if not matching_rows.empty:
        return matching_rows['CO2 Emissions(g/km)'].mean()
    else:
        return emissions_data['CO2 Emissions(g/km)'].mean()

# Function to save the models after they are loaded/trained
def save_models(object_detection_model, car_classification_model):
    torch.save(object_detection_model.state_dict(), 'object_detection_model.pth')
    torch.save(car_classification_model.state_dict(), 'car_classification_model.pth')

# Function to load the object detection model from saved file
def load_object_detection_model():
    model = torchvision.models.detection.ssd300_vgg16(pretrained=False)  # Set pretrained=False
    model.load_state_dict(torch.load('object_detection_model.pth'))
    model.eval()
    return model

# Function to load the car classification model from saved file
def load_car_classification_model():
    model = torchvision.models.resnet50(pretrained=False)  # Set pretrained=False
    model.load_state_dict(torch.load('car_classification_model.pth'))
    model.eval()
    return model

# Load ImageNet classes
with open('imagenet_classes.json', 'r') as f:
    imagenet_classes = json.load(f)

class CarDetectorApp(App):
    def build(self):
        # Load the pre-saved models
        self.object_detection_model = load_object_detection_model()
        self.car_classification_model = load_car_classification_model()
        
        # Set the device to CUDA if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.object_detection_model.to(self.device)
        self.car_classification_model.to(self.device)
        
        # Set up the camera and display
        self.layout = BoxLayout(orientation='vertical')
        self.camera = Camera(play=True, resolution=(640, 480))
        self.label = Label(text="Car Detection: Scanning...")
        self.layout.add_widget(self.camera)
        self.layout.add_widget(self.label)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.layout

    # Get the current frame from the camera
    def get_frame(self):
        pixels = self.camera.texture.pixels
        frame = np.frombuffer(pixels, dtype=np.uint8)
        frame = frame.reshape(self.camera.texture.height, self.camera.texture.width, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        frame = cv2.resize(frame, (300, 300))
        return frame

    # Detect and classify cars in the frame
    def detect_and_classify_cars(self, frame):
        image = Image.fromarray(frame)
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Detect objects
        with torch.no_grad():
            detections = self.object_detection_model(image_tensor)[0]
        
        car_detections = detections['boxes'][detections['labels'] == 3]  # Label 3 is 'car'
        car_scores = detections['scores'][detections['labels'] == 3]
        
        # Thresholding
        threshold = 0.6
        car_detections = car_detections[car_scores > threshold]
        
        classified_cars = []
        for box in car_detections:
            x1, y1, x2, y2 = box.int()
            car_image = frame[y1:y2, x1:x2]
            car_image = Image.fromarray(car_image)
            car_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(car_image).unsqueeze(0).to(self.device)
            
            # Classify the car
            with torch.no_grad():
                car_class = self.car_classification_model(car_tensor).argmax().item()
            
            classified_cars.append((box.cpu().numpy(), car_class))
        
        return classified_cars

    # Retrieve the make and model of the car from ImageNet classes
    def get_car_make_model(self, class_id):
        return imagenet_classes[class_id]

    # Estimate the car size based on the bounding box
    def estimate_car_size(self, box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        if area < 2000:
            return "COMPACT"
        elif area < 4000:
            return "MID-SIZE"
        else:
            return "SUV - SMALL"

    # Update the camera feed
    def update(self, dt):
        if self.camera.texture:
            frame = self.get_frame()
            detections = self.detect_and_classify_cars(frame)
            if len(detections) > 0:
                car, car_class = detections[0]
                make_model = self.get_car_make_model(car_class)
                vehicle_class = self.estimate_car_size(car)
                emissions = get_emissions(vehicle_class)
                self.label.text = f"Detected: {make_model}\nEstimated class: {vehicle_class}\nEstimated CO2 Emissions: {emissions:.2f} g/km"
            else:
                self.label.text = "No cars detected"
            self.display_frame(frame, [d[0] for d in detections])

    # Display the bounding boxes around the detected cars
    def display_frame(self, frame, detections):
        for box in detections:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.camera.texture = texture

if __name__ == "__main__":
    # Uncomment these lines to save the models after loading them the first time
    object_detection_model = load_object_detection_model()
    car_classification_model = load_car_classification_model()
    save_models(object_detection_model, car_classification_model)
    
    CarDetectorApp().run()
