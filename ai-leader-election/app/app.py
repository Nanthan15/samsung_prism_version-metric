#main file to load model , collect metrix , get results

import pandas as pd
from model import LeaderElectionModel  # Assuming model.py is in the same directory

# Initialize the model
model = LeaderElectionModel()

# Read data from CSV file for prediction
try:
    input_data = pd.read_csv('pod_metrics.csv')
    print("Input data loaded successfully from pod_metrics.csv")
except FileNotFoundError:
    print("Error: pod_metrics.csv file not found. Ensure it exists in the directory.")
    exit()

# Verify the CSV contains the required columns
required_columns = ['cpu_usage', 'memory_usage', 'network_latency', 'uptime']
if not all(col in input_data.columns for col in required_columns):
    print("Error: CSV file does not contain all required columns:", required_columns)
    exit()

# Predict using the CSV data
predictions = model.update_leader(input_data[required_columns])

# Display the predictions for each node
for result in predictions:
    print(result)
