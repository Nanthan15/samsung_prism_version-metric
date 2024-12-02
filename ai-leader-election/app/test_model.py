#used to test the model based on the train model 
# test_model.py
import pandas as pd
from model import LeaderElectionModel  # Assuming model.py is in the same directory

# Additional test data for verification
test_data = pd.DataFrame({
    'cpu_usage': [10, 30, 50, 70, 90, 20, 40, 60, 80, 95, 5, 37],
    'memory_usage': [55, 25, 45, 85, 95, 50, 30, 70, 5, 90, 15, 62],
    'network_latency': [2, 3, 7, 6, 9, 4, 1, 3, 8, 2, 6, 5],
    'uptime': [95, 96, 75, 60, 65, 85, 90, 89, 99, 55, 78, 68]
})

# Initialize the model
model = LeaderElectionModel()

# Predict using the test data
predictions = model.update_leader(test_data)

# Display the predictions for each node
for result in predictions:
    print(result)
