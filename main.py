# Import necessary libraries for the Carbon Emission Detector App
# The app uses Kivy for the UI, OpenCV for image processing, and YOLO for object detection

from kivy.app import App  # Main Kivy app class
from kivy.uix.boxlayout import BoxLayout  # Layout for arranging widgets vertically
from kivy.uix.camera import Camera  # Camera widget to capture and display the video feed
from kivy.uix.label import Label  # Label to display carbon emission data
from kivy.clock import Clock  # Clock to schedule updates at a regular interval (frame update)
import cv2  # OpenCV for handling image processing and object detection
import numpy as np  # Numpy for efficient array manipulation
from kivy.graphics.texture import Texture  # Converts image frames into textures to display in the Kivy app

# Define the CarbonDetectorApp class, which inherits from the Kivy App class.
# This app captures live camera feed, detects objects using YOLO, and estimates their carbon emissions.
class CarbonDetectorApp(App):
    
    # Initialize the app and build the user interface (UI)
    def build(self):
        # Load the YOLO model for vehicle detection
        self.load_model()  # Custom function to load the pre-trained YOLO model and its configuration
        
        # Create the main layout (vertical box) for displaying the camera feed and carbon emission label
        self.layout = BoxLayout(orientation='vertical')  
        # Initialize the camera with a resolution of 640x480 and start the live feed
        self.camera = Camera(play=True, resolution=(640, 480))  
        # Create a label to display the carbon emission estimates
        self.label = Label(text="Carbon Emission: Scanning...")  
        
        # Add the camera and label widgets to the layout
        self.layout.add_widget(self.camera)
        self.layout.add_widget(self.label)
        
        # Schedule the update function to run 30 times per second (30 FPS) to process camera frames
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        return self.layout  # Return the completed layout as the app's root widget
    
    # This function is called at every scheduled interval to update the camera feed and process object detection
    def update(self, dt):
        if self.camera.texture:  # Ensure the camera is capturing data
            # Extract the pixel data from the camera texture
            pixels = self.camera.texture.pixels
            # Convert the pixel data into a numpy array (for OpenCV processing)
            frame = np.frombuffer(pixels, dtype=np.uint8)
            # Reshape the array into the proper frame dimensions (height, width, 4 channels - RGBA)
            frame = frame.reshape(self.camera.texture.height, self.camera.texture.width, 4)
            
            # Convert the frame from RGBA (Kivy format) to BGR (OpenCV format) for processing
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Perform object detection using YOLO on the captured frame
            boxes, confidences, class_ids = self.detect_objects(frame)
            
            # Draw bounding boxes around detected objects and label them
            frame = self.draw_bounding_boxes(frame, boxes, confidences, class_ids)
            
            # Retrieve the detected vehicle types and estimate their carbon emissions
            detected_objects = [self.classes[class_id] for class_id in class_ids]
            emissions = [self.estimate_carbon_emission(obj) for obj in detected_objects]
            # Display the vehicle type and emission estimates on the label
            emission_text = ", ".join([f"{obj}: {emission} gCO2/km" for obj, emission in zip(detected_objects, emissions)])
            self.label.text = f"Vehicle Type & Emissions: {emission_text}"  # Update the label text
            
            # Convert the processed frame back into a Kivy texture for display
            buf = cv2.flip(frame, 0).tostring()  # Flip the frame vertically (needed for correct display)
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')  # Create a new texture
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')  # Update texture with the new frame data
            self.camera.texture = texture  # Set the updated texture as the current camera feed
    
    # Load the YOLO object detection model and corresponding class names for vehicle detection
    def load_model(self):
        # Load pre-trained YOLO weights and configuration from disk
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        # Read the vehicle class names from a text file (e.g., vehicle_classes.names)
        with open("vehicle_classes.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]  # List of detectable vehicle classes (e.g., sedan, truck)
        # Generate random colors to visually distinguish between different object classes
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = net  # Store the loaded YOLO model for later use in detection

    # Perform object detection on the given frame using the YOLO model
    def detect_objects(self, frame):
        height, width, _ = frame.shape  # Get the dimensions of the frame
        # Create a blob (image preprocessing) for YOLO: normalize, resize, and swap RGB channels
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)  # Set the blob as the input to the YOLO network
        
        # Get the names of the output layers from the YOLO model
        layer_names = self.net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Perform a forward pass through the YOLO model to get detection results
        outputs = self.net.forward(output_layers)
        
        boxes, confidences, class_ids = [], [], []  # Lists to store detected boxes, confidences, and class IDs
        
        # Loop through each output layer to extract detection data
        for output in outputs:
            for detection in output:
                scores = detection[5:]  # Detection confidence scores for each object class
                class_id = np.argmax(scores)  # Get the class with the highest score
                confidence = scores[class_id]  # Get the confidence level of the detection
                if confidence > 0.5:  # Only consider detections with confidence > 50%
                    # Calculate the center and dimensions of the detected object
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Calculate the top-left corner of the bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # Append the detection data to the respective lists
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids  # Return the detected boxes, confidences, and class IDs

    # Draw bounding boxes around detected objects in the frame
    def draw_bounding_boxes(self, frame, boxes, confidences, class_ids):
        # Use Non-Maximum Suppression (NMS) to reduce redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:  # Only draw boxes that passed NMS filtering
                x, y, w, h = boxes[i]  # Get the box coordinates
                label = str(self.classes[class_ids[i]])  # Get the object label (e.g., sedan, truck)
                color = self.colors[class_ids[i]]  # Get the color associated with the object class
                # Draw a rectangle around the detected object
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # Add the object label and confidence score above the box
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame  # Return the processed frame with bounding boxes

    # Estimate the carbon emission based on the detected vehicle type
    def estimate_carbon_emission(self, vehicle_type):
        # Carbon emission data (in gCO2/km) for different vehicle types
        vehicle_emissions = {
            "sedan": 180,
            "suv": 220,
            "truck": 250,
            "van": 200,
            "motorcycle": 100
        }
        # Return the estimated emissions for the detected vehicle type
        return vehicle_emissions.get(vehicle_type.lower(), "Unknown")  # Return "Unknown" if vehicle type is not found

# Run the CarbonDetectorApp
if __name__ == '__main__':
    CarbonDetectorApp().run()
