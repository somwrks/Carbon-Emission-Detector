# Carbon Emission Detector App

This project is a Kivy-based mobile app that uses a pre-trained Faster R-CNN model for car detection and estimates carbon emissions based on the detected vehicles. It leverages OpenCV for image processing and provides live updates of the detected cars and their corresponding estimated carbon footprint.

### Here's what the code is trying to do-

1. Use a pre-trained Faster R-CNN model from torchvision for car detection.
2. Process camera input in real-time using a Kivy app.
3. Detect cars in the camera feed.
4. Estimate the size/class of the detected car.
5. Predict carbon emissions based on the estimated car size using a Canadian CO2 emissions dataset.
6. Display the results in real-time, including the detected car's class and estimated CO2 emissions.

![image](https://github.com/user-attachments/assets/19bdaebc-4840-499f-9551-9045b36d0ee2)


## Run the Project

Follow these steps to set up the development environment and run the project:

### Prerequisites
Ensure you have Python installed on your machine. Then, follow these steps:

1. Install required dependencies:
   ```
   pip install kivy opencv-python numpy torch torchvision pillow pandas

   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Linux/MacOS:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```

4. Download the YOLO weights, configuration file, and class names:
   ```
   wget https://pjreddie.com/media/files/yolov3.weights
   wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
   ```


5. Run the app:
   ```
   python main.py
   ```

## Project Structure
`main.py`: The main application file containing the Kivy app and car detection logic.
`data/CO2_Emissions_Canada.csv`: Dataset containing CO2 emissions data for various vehicle classes.

## How It Works

- The app uses a pre-trained Faster R-CNN model to detect cars in the camera feed.
- For each detected car, it estimates the vehicle class (COMPACT, MID-SIZE, or SUV - SMALL) based on the size of the detected bounding box.
- Using the estimated vehicle class, it looks up the average CO2 emissions from the Canadian emissions dataset.
- The app displays the detected vehicle class and its estimated CO2 emissions in real-time.

## Future Improvements

1. Implement a more sophisticated car make and model recognition system.
2. Improve the accuracy of CO2 emissions estimation by considering more factors.
3. Optimize the app for better performance on mobile devices.
4. Expand the dataset to include a wider range of vehicles and more accurate emissions data.


## Contributing

We welcome contributions to improve the app! Whether it's fixing a bug, adding a feature, or improving documentation, your help is appreciated.

### How to Contribute

1. **Fork the repository**:  
   Start by forking this repository to your own GitHub account.

2. **Clone the forked repository**:  
   Clone your forked repository to your local machine:
   ```
   git clone https://github.com/somwrks/Carbon-Emission-Detector.git
   ```

3. **Create a new branch**:  
   Make sure to create a new branch for your changes:
   ```
   git checkout -b main
   ```

4. **Make your changes**:  
   Implement your changes in this branch. Ensure your code follows best practices and is well-documented.

5. **Run tests**:  
   Ensure that your changes don't break any existing functionality. Test the app locally before submitting a PR.

6. **Commit your changes**:  
   Use clear and concise commit messages:
   ```
   git commit -m "Added feature X" 
   ```

7. **Push your changes**:  
   Push your changes to your fork:
   ```
   git push origin main
   ```

8. **Submit a Pull Request (PR)**:  
   Open a pull request to the main repository:
   - Make sure to describe your changes clearly in the PR description.
   - Reference any issues that are being addressed by the PR (if applicable).

### Pull Request Guidelines

- Ensure your PR is up-to-date with the `main` branch before submitting.
- Add appropriate comments and documentation to your code.
- Write meaningful commit messages.
- Be respectful of code reviews and make requested changes promptly.
- Please do not make any direct changes to the `main` branch.
- Small, focused PRs are preferred over large, all-encompassing ones.

### Reporting Issues

If you encounter any bugs, problems, or have feature requests, feel free to open an issue on GitHub. Please provide as much context as possible, including steps to reproduce the problem.

---

Thank you for contributing to the Carbon Emission Detector App! Your help is vital in making this project better for everyone.

