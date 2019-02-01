import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()  # all DataFrame in NumpPy Array
print("====================", "all DataFrame in NumpPy Array", iris, sep="\n")
print("====================", "iris.data.shape", iris.data.shape, sep="\n")
print("====================", "List Unique Values In A pandas Column", pd.DataFrame(iris.target)[0].unique(), sep='\n')
print("====================", "les attribues", iris.feature_names, sep='\n')

# -----------------------------------------------
y = iris.target
x = iris.data

# ---------------------------------------------------------------
#   Classification supérvisé Knn
# ---------------------------------------------------------------
k = 5
model = KNeighborsClassifier(n_neighbors=k)
# model.fit(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=99)
print("====================", "taille base d'apprentissage", x_train.shape, y_train.shape, sep="\n")
print("====================", "taille base de test:\n", x_test.shape, "|", y_test.shape, sep="\n")
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print("====================", "taux de classification est:", accuracy_score(y_test, y_predict), sep="\n")

# selection des variables
x_new = SelectKBest(chi2, k=2).fit_transform(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=99)
model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
y_predict = model.predict(x_test)
print("====================", "resultat avec KNN",
      pd.DataFrame(confusion_matrix(y_test, y_predict), columns=["setosa", "versicolor", "virfinica"],
                   index=["0", "1", "2"]), sep="\n")
acc_succee = accuracy_score(y_test, y_predict)
print("====================", "taux de classification est:", acc_succee, sep="\n")
print("====================", "taux de classification est:", 1 - acc_succee, sep="\n")


