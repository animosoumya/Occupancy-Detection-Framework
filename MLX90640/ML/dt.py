import pandas as pd
from sklearn import metrics
from numpy import std, mean
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\animo\Downloads\MS\Pi_4B\MLX\data\dc_2\frame_features.csv")
X = dataset.iloc[:, [1, 2, 3, 4]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

n_scores = cross_val_score(dtc, X_train, y_train, cv=10)

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(f1_score(y_test, y_pred, average='macro'))