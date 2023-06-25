from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import time

# Load the digits dataset
digits = datasets.load_digits()

# Prepare the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Specify the hyperparameter space
parameters = {
    "hidden_layer_sizes": [(50,), (50, 50), (100,)],
    "activation": ["relu", "tanh", "logistic"],
    "alpha": [0.0001, 0.001],
}

# Initialize a Multilayer Perceptron Classifier
mlp = MLPClassifier(max_iter=2000, verbose=True)

# Starting time
starting_time = time.time()
print(f"Starting time: {starting_time}")

# Conduct Grid Search
clf = GridSearchCV(mlp, parameters, cv=5, verbose=2)

# Fit the model
clf.fit(X_train, y_train)

# Print the best parameters
print("Best parameters set found on development set:")
print(clf.best_params_)
end_time = time.time()
print(f"Ending time: {end_time}")
print(f"Elapsed time: {end_time - starting_time}")
