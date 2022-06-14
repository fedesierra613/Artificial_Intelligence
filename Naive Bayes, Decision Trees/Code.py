# -*- coding: utf-8 -*-
"""

# Tarea #5: Metodos adicionales de clasificación

Usar el conjunto de datos de Fisher e implementar en python los métodos 1 Naive Bayes, 2 Decision Trees
Evaluar los dos usando las métricas de MCC y F1 y accuracy y compararlos.


**Empezamos con el método de Decision Tree**
"""

# Librerias a utilizar
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn import tree

# Se cargan los datos 
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

# Se divide el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
X=iris.data 
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Se crea Crea el objeto clasificador de árbol de decisión
clf = DecisionTreeClassifier()

# Clasificador del árbol de decisión 
clf = clf.fit(X_train,y_train)

# Predecir la respuesta para el conjunto de datos de prueba
y_pred = clf.predict(X_test)

tree.plot_tree(clf)
[...]

#Evaluar el modelo
#Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#MCC
from sklearn.metrics import matthews_corrcoef
print("MCC:", metrics.matthews_corrcoef(y_test, y_pred))

#F1
from sklearn.metrics import f1_score
print("F1 Score:", metrics.f1_score(y_test, y_pred,average='macro'))

# Exportar el arbol en formato Graphviz para  colorear los nodos por su clase 
#y usar variables explícitas y nombres de clases
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file=None, 
...                      feature_names=iris.feature_names,  
...                      class_names=iris.target_names,  
...                      filled=True, rounded=True,  
...                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

"""**Ahora vamos con Naive Bayes**"""

# Librerias a utilizar
import pandas as pd
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Se cargan los datos 
from sklearn.datasets import load_iris
iris = load_iris()

X, y = load_iris(return_X_y=True)

# Se divide el conjunto de datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Número de puntos mal etiquetados de un total de %d puntos : %d"
...       % (X_test.shape[0], (y_test != y_pred).sum()))

#Evaluar el modelo
#Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#MCC
from sklearn.metrics import matthews_corrcoef
print("MCC:", metrics.matthews_corrcoef(y_test, y_pred))

#F1
from sklearn.metrics import f1_score
print("F1 Score:", metrics.f1_score(y_test, y_pred,average='macro'))