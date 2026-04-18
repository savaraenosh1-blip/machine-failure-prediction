# Machine Failure Prediction Project

This project provides an industrial dashboard to predict machine failures based on sensor data.

## Features
- Real-time sensor input tracking (Temperature, RPM, Torque, Tool Wear).
- Explainable AI (SHAP) integration.
- PDF diagnostic report generation.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python app.py
   ```
3. Open `http://127.0.0.1:5000` in your browser.

## Project Structure
- `app.py`: Main Flask application.
- `model.py`: Script to train the Random Forest model.
- `machine_failure_dataset.csv`: Dataset used for training.
- `model.pkl`: Trained model file.
- `columns.json`: Feature names for the model.
- `templates/`: HTML templates for the dashboard.
- `static/`: CSS styling for the dashboard.
