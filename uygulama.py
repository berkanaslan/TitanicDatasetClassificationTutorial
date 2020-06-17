#Küpüthaneler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
 
#Eğitim seti
egitim = pd.read_csv('train.csv',header=0)

egitim.info()

cikarilacaklar = ['Name','Ticket','Cabin']
egitim = egitim.drop(cikarilacaklar,axis=1)

egitim = egitim.dropna()

egitim.info()

yolcular = []
sutunlar = ['Pclass','Sex','Embarked']
for sutun in sutunlar:
    yolcular.append(pd.get_dummies(egitim[sutun]))

yolcular = pd.concat(yolcular, axis=1)
print(yolcular)

yolcular = yolcular.drop(['female'], axis=1)
egitim = pd.concat((egitim,yolcular),axis=1)
egitim = egitim.drop(['Pclass','Sex','Embarked'],axis=1)

print(egitim)

X = egitim.values
Y = egitim['Survived'].values

X = np.delete(X,1,axis=1)

X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.3, random_state=0)

from sklearn import tree
siniflama = tree.DecisionTreeClassifier(max_depth=5)
siniflama.fit(X_train,y_train)
skor = siniflama.score(X_test,y_test)

print("Başarı: ",skor)

from sklearn.metrics import accuracy_score
tahminler = siniflama.predict(X)
as_egitim = accuracy_score(tahminler, Y)

print("Doğruluk tablosu skoru: ", as_egitim)

confusion_matrix = pd.crosstab(Y, tahminler, rownames=['Gerçek'], colnames=['Tahmin'])
print (confusion_matrix)

