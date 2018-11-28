import numpy as np
import pandas as pd

dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_1 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])

le_2 = LabelEncoder()
X[:, 2] = le_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
# avoiding dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# --- Building ANN ----
# Importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# --- Making predictions and evaluating the model -----

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Plotting confusion matrix
import matplotlib.pyplot as plt
plt.clf()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Wistia)
classNames=['Negative','Positive']
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]))

plt.show()


# Part Three

# Homework
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# Does this customer have a credit card ? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $50000
# So should we say goodbye to that customer ?

# This is my implementation for testing new case
data = pd.DataFrame(data=[[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
test = data.values
test[:, 2] = le_2.transform(test[:, 2])
test[:, 1] = le_1.transform(test[:, 1])
test = onehotencoder.transform(test).toarray()
# avoiding dummy
test = test[:, 1:]
test_pred = classifier.predict(sc.transform(test))

# Alternative method for testing which is not good since we have to check how label is encoded in previous data
# test_pred = classifier.predict(sc.transform([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))

test_pred = (test_pred > 0.5)
print(test_pred)


# Part Four

