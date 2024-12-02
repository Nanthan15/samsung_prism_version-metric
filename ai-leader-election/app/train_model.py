## this is used to train the model and store the trained model data in PKL file to use it test on various diffrent scenario


# train.py

import pandas as pd
from model import LeaderElectionModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Import tqdm for progress bar

# Enhanced data (consider expanding this dataset)
data = pd.DataFrame({
    'cpu_usage': [20, 30, 50, 70, 90, 60, 40, 80, 85, 15, 45, 55],
    'memory_usage': [30, 20, 10, 50, 60, 55, 25, 45, 70, 15, 40, 30],
    'network_latency': [5, 10, 3, 7, 2, 6, 4, 9, 8, 3, 1, 5],
    'uptime': [99, 95, 90, 98, 85, 80, 88, 93, 60, 99, 70, 75]
})
labels = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # Example leader (1) or not (0)

# Split the data for validation
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Using Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

# Initialize variables to track the best model
best_model = None
best_score = 0

# Manually iterate over parameter combinations for progress bar
total_combinations = (
    len(param_grid['n_estimators']) *
    len(param_grid['max_depth']) *
    len(param_grid['min_samples_split']) *
    len(param_grid['min_samples_leaf']) *
    len(param_grid['class_weight'])
)

with tqdm(total=total_combinations, desc='Training', unit='combination') as pbar:
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for min_samples_leaf in param_grid['min_samples_leaf']:
                    for class_weight in param_grid['class_weight']:
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            class_weight=class_weight,
                            random_state=42
                        )
                        model.fit(X_train, y_train)  # Train the model
                        
                        # Evaluate the model
                        score = model.score(X_test, y_test)  # Get accuracy on the test set
                        if score > best_score:
                            best_score = score
                            best_model = model  # Update best model

                        pbar.update(1)  # Update progress bar

# Train with the best estimator found
if best_model is not None:
    model = LeaderElectionModel()
    model.model = best_model
    model.train(X_train, y_train)

    # Predict and evaluate
    y_pred = model.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
else:
    print("No model was trained.")
