# Importing necessary libraries
from kivy.app import App  # Main Kivy app class
from kivy.uix.boxlayout import BoxLayout  # Layout to arrange widgets
from kivy.uix.camera import Camera  # Camera widget to capture video feed
from kivy.uix.label import Label  # Label to display text (e.g., carbon emission)
from kivy.clock import Clock  # For scheduling interval updates
import cv2  # OpenCV for handling image processing and object detection
import numpy as np  # Numpy for array manipulation
from kivy.graphics.texture import Texture  # Convert image frames to textures for display

class CarbonDetectorApp(App):
    # This function initializes the app, loads the model, sets up the layout, and starts the camera.
    def build(self):
        self.load_model()  # Load YOLO model for object detection
        self.layout = BoxLayout(orientation='vertical')  # Create a vertical layout
        self.camera = Camera(play=True, resolution=(640, 480))  # Initialize camera feed with a 640x480 resolution
        self.label = Label(text="Carbon Emission: Scanning...")  # Create a label to display emission info
        
        self.layout.add_widget(self.camera)  # Add camera feed to the layout
        self.layout.add_widget(self.label)  # Add label to the layout
        
        # Schedule the 'update' method to run at 30 frames per second (FPS)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        return self.layout  # Return the layout as the main UI element
    
    # This function updates the camera feed, performs object detection, and estimates carbon emissions.
    def update(self, dt):
        if self.camera.texture:
            # Extract pixel data from the camera texture
            pixels = self.camera.texture.pixels
            
            # Convert the pixel buffer to a numpy array
            frame = np.frombuffer(pixels, dtype=np.uint8)
            # Reshape the array to match the camera resolution with RGBA channels
            frame = frame.reshape(self.camera.texture.height, self.camera.texture.width, 4)
            
            # Convert RGBA format to BGR (used by OpenCV)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Perform object detection using YOLO model
            boxes, confidences, class_ids = self.detect_objects(frame)
            
            # Draw bounding boxes around detected objects
            frame = self.draw_bounding_boxes(frame, boxes, confidences, class_ids)
            
            # Update carbon emission estimation for detected objects
            detected_objects = [self.classes[class_id] for class_id in class_ids]
            # Estimate emissions for each detected object
            emissions = [self.estimate_carbon_emission(obj) for obj in detected_objects]
            # Format the emission data into a readable string
            emission_text = ", ".join([f"{obj}: {emission}" for obj, emission in zip(detected_objects, emissions)])
            self.label.text = f"Carbon Emissions: {emission_text}"  # Update label with emission information
            
            # Convert the processed frame back to a texture and display it
            buf = cv2.flip(frame, 0).tostring()  # Flip the frame vertically (for correct orientation)
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')  # Create a Kivy texture
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')  # Copy buffer to texture
            self.camera.texture = texture  # Set the updated texture as the camera feed
    
    # Load the YOLO model and classes for object detection
    def load_model(self):
        # Load pre-trained YOLO weights and configuration file
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        # Load object class names from the coco.names file (e.g., car, bus, person, etc.)
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        # Generate random colors for each class
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = net  # Store the YOLO model for later use

    # Function to detect objects in a given frame
    def detect_objects(self, frame):
        height, width, _ = frame.shape  # Get the frame's dimensions
        # Create a blob from the image, normalize and resize it for YOLO model
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)  # Set the blob as input to the YOLO network
        
        # Get the YOLO model's output layers
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # Perform forward pass to get output from YOLO
        outputs = self.net.forward(output_layers)
        
        boxes = []  # List to store bounding box coordinates
        confidences = []  # List to store detection confidence levels
        class_ids = []  # List to store detected object class IDs
        
        # Iterate through the outputs and extract detected objects
        for output in outputs:
            for detection in output:
                scores = detection[5:]  # Extract the scores for all object classes
                class_id = np.argmax(scores)  # Get the class with the highest score
                confidence = scores[class_id]  # Confidence level of the detected object
                if confidence > 0.5:  # Only consider detections with confidence > 50%
                    # Get the coordinates for the center of the object
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    # Get the width and height of the detected object
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Calculate the top-left corner of the bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # Append the bounding box, confidence, and class ID to their respective lists
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids  # Return the detected boxes, confidences, and class IDs

    # Draw bounding boxes around detected objects
    def draw_bounding_boxes(self, frame, boxes, confidences, class_ids):
        # Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:  # Only consider boxes that passed NMS
                x, y, w, h = boxes[i]  # Extract the box coordinates
                label = str(self.classes[class_ids[i]])  # Get the object label (class name)
                color = self.colors[class_ids[i]]  # Get the color for the class
                # Draw the rectangle around the object
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # Add the label (class name) above the rectangle
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame  # Return the frame with bounding boxes drawn

    # Estimate carbon emissions based on the detected object type
    def estimate_carbon_emission(self, object_name):
        # Simplified carbon emission estimates for different objects (in gCO2e)
        emissions = {
            "car": 120,  # Car emits 120 gCO2e per km
            "bus": 80,   # Bus emits 80 gCO2e per km
            "bicycle": 0,  # Bicycle emits 0 carbon
            "person": 0,  # Human walking has negligible carbon emission
            "laptop": 52, # Laptop emits 52 kgCO2e/year
            # Add more objects and their emission data as needed
        }
        # Return the emission value for the object, or "Unknown" if not found
        return emissions.get(object_name, "Unknown")

# Run the app
if __name__ == '__main__':
    CarbonDetectorApp().run()
