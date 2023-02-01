import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split


# Importing the dataset
dataset = pd.read_csv(r"C:\Users\animo\Downloads\MS\Pi_4B\MLX\data\dc_0\frame_features.csv")
X = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the K-NN model on the Training set
classifier = KNeighborsClassifier(n_neighbors = 20, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print('ac', accuracy_score(y_test, y_pred))

#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=20)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(f1_score(y_test, y_pred, average='macro'))