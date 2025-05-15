# Predictive Maintenance System

Predicts machine failure using sensor data like temperature, torque, RPM, and tool wear. The model uses rolling features and an XGBoost classifier.

## Features:
- Sensor feature extraction with rolling mean/std
- Failure prediction (0 = Healthy, 1 = Failure)
- Model: XGBoost Classifier

## Technologies:
- pandas, matplotlib, scikit-learn, XGBoost

## Usage:
1. Place your CSV data file in the `/data` folder.
2. Run the preprocessing and training notebook from `/notebooks`.
3. The trained model will be saved in `/models/saved_model.pkl`.

## Project Structure:
Predictive_Maintenance_for_Industrial_Equipment/
├── Dataset/
│ └── sensor_data.csv
├── Jupeter_notebook/
│ └── predictive_maintenance.ipynb
└── streamlit_app/
└──   App.py
└──   saved_model.pkl
|__   README.md
|__   requirements.txt