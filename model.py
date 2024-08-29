from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Add the target (species) to the DataFrame
# Target variable (species: 0 for setosa, 1 for versicolor, 2 for virginica)
df['species'] = iris.target

# Print the feature names and target names
print("Feature names:", iris.feature_names)

print("Target names:", iris.target_names)

# Save the DataFrame to a CSV file
df.to_csv('iris_dataset.csv', index=False)

#shaffle the rows of the dataframe
df_shaffled = df.sample(frac=1, random_state=42)


# selecting features and target data
x = df_shaffled.drop("species", axis=1)
y = df_shaffled["species"]


## split data into train and test sets
from sklearn.model_selection import train_test_split

# 70% training and 30% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,  random_state=42, stratify=y)


from sklearn.ensemble import RandomForestClassifier
# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
# train the classifier on the training data
clf.fit(x_train, y_train)


# predict on the test set
y_pred = clf.predict(x_test)

# calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%") ## Accuracy: 0.93

#pip install joblib
#from joblib import Memory
import joblib

#save the model to disk
joblib.dump(clf, "rf_model.sav")


##install streamlit with conda within jupyter notebook
#import sys
#!conda install --yes --prefix {sys.prefix} streamlit
