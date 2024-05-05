#my project
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
digits = datasets.load_digits()

X = digits.data[:, :]
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42, stratify=y)
#This stratify parameter makes a split so that the proportion of values in
#the sample produced will be the same as the proportion of values provided
#by parameter stratify.
sc = StandardScaler() #z = (x - u) / s
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
ppn = Perceptron(eta0=0.1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print()
