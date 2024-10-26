# This is a test file for sci kit learn

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# load the iris data set
iris = load_iris()

# Convert to pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Explore the data
print(iris_df.head())
print(iris_df.describe())

# Separate features (X) and target variable (y)
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but often beneficial)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #this fits and transforms the data
X_test = scaler.transform(X_test) # this uses the same fit from the training set

# K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Plot y_pred vs y_test, and save plot as png
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.savefig('knn_plot.png')
