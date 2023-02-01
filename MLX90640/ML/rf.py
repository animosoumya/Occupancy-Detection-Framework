import pandas as pd
from sklearn import metrics
from numpy import std, mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score


# Importing the dataset
dataset = pd.read_csv(r"C:\Users\animo\Downloads\MS\Pi_4B\MLX\data\dc_4\frame_features.csv")
X = dataset.iloc[:, [1,2,3,4]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100) 

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(f1_score(y_test, y_pred, average='macro'))