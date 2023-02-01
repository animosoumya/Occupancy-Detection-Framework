#Import svm model
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score


# Importing the dataset
dataset = pd.read_csv(r"C:\Users\animo\Downloads\MS\Pi_4B\MLX\data\dc_0\frame_features.csv")
X = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print("Naive Bayes score: ",nb.score(X_test, y_test))

scores = cross_val_score(nb, X, y, cv=10)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(f1_score(y_test, y_pred, average='macro'))