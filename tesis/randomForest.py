import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

featureMatrix_all = pd.read_csv('DATOS.csv')
sns.heatmap(featureMatrix_all.isnull()) #Para ver si hay valorees nulos
featureMatrix_all.columns #Para obtener los nombres de las columnas
Y = featureMatrix_all['status']
X = featureMatrix_all.drop(['status','subject'],axis=1)

#Se crean los set de entrenamientos y de pruebas
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators=80)
randomforest.fit(X_train, Y_train) #Función para entrenar el modelo
predicciones = randomforest.predict(X_test) #Función para hacer las predicciones


from sklearn.metrics import classification_report
 
print(classification_report(Y_test,predicciones))


from sklearn.metrics import confusion_matrix
 
confusion_matrix(Y_test,predicciones)

TP = confusion_matrix(Y_test,predicciones)[0][0] #True_positive
FN = confusion_matrix(Y_test,predicciones)[0][1] #False_negative
FP = confusion_matrix(Y_test,predicciones)[1][0] #False_positive
TN = confusion_matrix(Y_test,predicciones)[1][1] #True_negative

TPR = TP / (TP + FN) # true_positive_rate 
TPR

FPR = FP / (FP + TN) # false_positive_rate
FPR

TNR = TN / (TN + FP) # true_negative_rate
TNR

from sklearn import metrics
from sklearn import svm, datasets


print("Accuracy", metrics.accuracy_score(Y_test, predicciones))

Y_pred_proba = arbol.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(Y_test,  Y_pred_proba)