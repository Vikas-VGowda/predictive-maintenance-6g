# Predictive Maintenance and Anomaly Detection

A data science project for 6G-integrated smart manufacturing.

## Features
- Manufacturing sensor data analysis
- Isolation Forest anomaly detection
- Relative maintenance risk categories
- Machine-level sensor visualizations
- Maintenance alert table
- Historical trend analysis
- CSV downloads

## Requirements
Python 3.10 or newer

## Run locally
1. Install dependencies:
   pip install -r requirements.txt

2. Place processed_manufacturing_data.csv next to app.py.

3. Launch the dashboard:
   streamlit run app.py

## Important
Risk levels represent relative anomaly scores, not validated
probabilities of equipment failure.
