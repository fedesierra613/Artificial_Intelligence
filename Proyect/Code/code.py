# -*- coding: utf-8 -*-
"""ProyectoFinal.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lmSLpQvuhs3FqyBmaKfpywDg9STyMHqd
"""

# Commented out IPython magic to ensure Python compatibility.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import decomposition
from sklearn.decomposition import PCA, KernelPCA
from sklearn import metrics
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import cross_val_score

# %matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#Visualizacion de los datos del dataset
data = pd.read_csv(r"heart.csv") 
data.head(-5)

#Revisar si existen datos nulos o espacios de columnas sin rellenar 
print("Valores nulos en train: \n", data.isna().sum(),"\n")

#Se eliminar de las columnas de características aquellas que no son útiles para nuestro proyecto
data.drop(['time'], axis = 1, inplace = True) 
data.head(-5)

fig1 = make_subplots(rows=5, cols=3, subplot_titles=data.columns)

for i, col in enumerate(data.columns):
    fig1.add_trace(go.Histogram(x=data[col], name=col), row=(i//3)+1, col=(i%3)+1)
    
fig1.update_layout(height=200*5, showlegend=False)
    
fig1.show()

X = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking']].values
y = data['DEATH_EVENT'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.25)

scaler = StandardScaler() #Se escalizan los datos
scaler.fit(X_train) #El fit de los datos se hace con el conjunto de entrenamiento!
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#PCA solamente al conjunto de entrenamiento 
n = 11
pca = decomposition.PCA(n_components=n,whiten=True,svd_solver='auto')
pca.fit(X_train_scaled)
X_train_PCA = pca.transform(X_train_scaled)
X_test_PCA = pca.transform(X_test_scaled)
print("Pesos de PCA:",pca.explained_variance_ratio_)
print("Componentes:",pca.components_)

#Empieza el entrenanmiento

#Regresión logística
log = LogisticRegression(C=0.2, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=200,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=20, solver='liblinear', tol=0.1, verbose=0,
                   warm_start=False)

log.fit(X_train_PCA, y_train)

pred = log.predict(X_test_PCA)

print("F1 Score:", metrics.f1_score(y_test, pred,average='macro'))

print("MCC:", metrics.matthews_corrcoef(y_test, pred))

#SVM
kernels=['linear', 'poly', 'rbf', 'sigmoid']
#lineal
#Kernel=0
#msv = svm.SVC(kernel=kernels[Kernel])

#polinomial cuadrático
#Kernel=1
#msv = svm.SVC(kernel=kernels[Kernel],degree=2)

#polinomial cúbico
#Kernel=1
#msv = svm.SVC(kernel=kernels[Kernel],degree=3)
#rbf 
Kernel=2
msv = svm.SVC(kernel=kernels[Kernel],C=1)
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

msv.fit(X_train_PCA, y_train)

y_test_predicted = msv.predict(X_test_PCA)

print("F1 Score:", metrics.f1_score(y_test, y_test_predicted,average='macro'))

print("MCC:", metrics.matthews_corrcoef(y_test, y_test_predicted))