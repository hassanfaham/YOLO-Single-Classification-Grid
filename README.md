# YOLO Single Classification Grid

## Description
This project is an image classification application that uses the YOLO model to classify images into predefined categories. It processes images in real-time from a monitored folder and visualizes results in a grid format. The application provides an easy-to-use interface with the ability to upload models, start/stop monitoring, and display image processing results.

The project includes scripts for:
- **Inference**: Run inference on a folder of images using a pre-trained model.
- **Training and Evaluation**: Train and evaluate a YOLOv8 or YOLOv11 model with custom hyperparameters, augmentations, and optional layer freezing.
  
## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Demo](#demo)
4. [File Structure](#file-structure)
5. [Requirements](#requirements)
6. [Contributing](#contributing)
7. [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Install dependencies:
    ```bash
    pip install -r reqs
    ```

3. Make sure you have Python 3.x installed.

4. Download the YOLO model files and place them in the appropriate folder (e.g., `./models/single_perspective_cls.pt`).

## Usage

1. **Running the App**:
    - Start the application UI:
        ```bash
        python app_ui.py
        ```
    - Upload a model through the GUI and start monitoring a folder for new images.
    - The app will automatically process images and display them in the grid format, showing the classification status as "OK" or "NOK".

2. **Inference**:
    - Run the `inference.py` script to perform inference using a pre-trained model on a folder of images:
      ```bash
      python scripts/inference.py --model-path ./models/single_perspective_cls.pt --images-path ./images
      ```

    - The script supports running inference on a single model or a folder of models and images. It will save annotated results for each processed image.

3. **Training and Evaluation**:
    - To train a model, use the `train_and_eval.py` script:
      ```bash
      python scripts/train_and_eval.py --data-path ./data --model-path ./models/yolo11s-cls.pt
      ```

    - The script trains a YOLOv8 or YOLOv11 model, evaluates it, and computes precision, recall, F-scores, and inference speed.

## Demo
[![Demo Video](link-to-thumbnail-image)](link-to-your-demo-video)

## File Structure

- `app_ui.py`: The main user interface for the application.
- `model_manager.py`: Handles loading and managing machine learning models.
- `processing_manager.py`: Monitors the folder and processes images.
- `thread_manager.py`: Manages threading for background tasks.
- `fatal_error_handler.py`: Handles and logs fatal errors.
- `logger_config.py`: Configures the logging system for the app.
- `config.json`: Configuration file for the app.
- `scripts/inference.py`: Script for performing inference on a folder of images using a pre-trained model.
- `scripts/train_and_eval.py`: Script for training and evaluating a YOLOv8/YOLOv11 model.
- `reqs`: A file listing the required dependencies for the project.

## Requirements

- Python 3.x
- Required libraries:
    - `customtkinter`
    - `pillow`
    - `opencv-python`
    - `numpy`
    - `torch`
    - `watchdog`
    - `loguru`
    - `ultralytics` (for YOLO)

You can install the dependencies by running:
```bash
pip install -r reqs
