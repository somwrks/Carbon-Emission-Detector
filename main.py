import torch
import torchvision
from torchvision import transforms
from PIL import Image
import json

class CarDetectorApp(App):
    def build(self):
        self.load_models()
        self.layout = BoxLayout(orientation='vertical')
        self.camera = Camera(play=True, resolution=(640, 480))
        self.label = Label(text="Car Detection: Scanning...")
        self.layout.add_widget(self.camera)
        self.layout.add_widget(self.label)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.layout

    def load_models(self):
        # Load car detection model (YOLO)
        self.detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Load car recognition model (ResNet50)
        self.recognition_model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.recognition_model.fc.in_features
        self.recognition_model.fc = torch.nn.Linear(num_ftrs, 196)  # 196 classes in Stanford Cars Dataset
        self.recognition_model.load_state_dict(torch.load('path_to_stanford_cars_model.pth'))
        self.recognition_model.eval()

        # Load class names
        with open('stanford_cars_classes.json', 'r') as f:
            self.class_names = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def update(self, dt):
        if self.camera.texture:
            frame = self.get_frame()
            detections = self.detect_cars(frame)
            if len(detections):
                car = detections[0]  # Process only the first detected car
                car_image = frame[int(car[1]):int(car[3]), int(car[0]):int(car[2])]
                make_model = self.recognize_car(car_image)
                emissions = self.predict_emissions(make_model)
                self.label.text = f"Detected: {make_model}\nEmissions: {emissions:.2f} g/km"
            else:
                self.label.text = "No cars detected"
            self.display_frame(frame)

    def recognize_car(self, car_image):
        input_tensor = self.transform(Image.fromarray(car_image))
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.recognition_model(input_batch)
        _, predicted = torch.max(output, 1)
        return self.class_names[predicted.item()]

    def predict_emissions(self, make_model):
        # Base emissions for an average passenger vehicle (400 g/mi = 248.55 g/km)
        base_emissions = 248.55
        
        # Adjust emissions based on vehicle type (example adjustments)
        if "SUV" in make_model or "Truck" in make_model:
            return base_emissions * 1.5  # 50% higher emissions for larger vehicles
        elif "Hybrid" in make_model or "Electric" in make_model:
            return base_emissions * 0.5  # 50% lower emissions for hybrid/electric vehicles
        else:
            return base_emissions

    def display_frame(self, frame):
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.camera.texture = texture

if __name__ == "__main__":
    CarDetectorApp().run()