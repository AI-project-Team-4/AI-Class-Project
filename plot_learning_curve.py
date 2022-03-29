from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Function is from https://github.com/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb
def plot_learning_curve(model, X, y, step):
    # Train and test model
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    train_errors, val_errors = [], []

    # For each of the rows in the training data
    for m in tqdm(range(model.n_neighbors, len(X_train) + 1, step)):
        model.fit(X_train[:m], y_train[:m]) # Train the provided model

        y_train_predict = model.predict(X_train[:m]) # Get predictions for training
        y_val_predict = model.predict(X_val) # Get predictions for values

        # Add results to lists
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    # Create and show the learning curve plot 
    plt.plot(np.sqrt(train_errors), "r-", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=2, label="val")
    plt.legend(loc="upper right")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")