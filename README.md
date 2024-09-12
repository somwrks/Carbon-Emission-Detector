# Carbon Emission Detector App

This project is a Kivy-based mobile app that uses YOLO for object detection and estimates carbon emissions based on the detected objects. It leverages OpenCV for image processing and provides live updates of the detected objects and their corresponding carbon footprint.

### Exact Workflow/Structure of Application and Dataset is undecided, feel free to @ me to discuss or join discord 


## Run the Project

Follow these steps to set up the development environment and run the project:

### Prerequisites
Ensure you have Python installed on your machine. Then, follow these steps:

1. Install required dependencies:
   ```
   pip install kivy tensorflow opencv-python kaggle numpy pandas torch torchvision
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
   wget https://github.com/pjreddie/darknet/blob/master/data/coco.names
   ```


5. Run the app:
   ```
   python main.py
   ```

## Contributing

We welcome contributions to improve the app! Whether it's fixing a bug, adding a feature, or improving documentation, your help is appreciated.

### How to Contribute

1. **Fork the repository**:  
   Start by forking this repository to your own GitHub account.

2. **Clone the forked repository**:  
   Clone your forked repository to your local machine:
   ```
   git clone https://github.com/<your-username>/carbon-emission-detector.git
   ```

3. **Create a new branch**:  
   Make sure to create a new branch for your changes:
   ```
   git checkout -b your-branch-name
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
   git push origin your-branch-name
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

