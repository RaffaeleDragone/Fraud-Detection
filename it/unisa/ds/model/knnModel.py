from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

from it.unisa.ds.util import dataUtility as du

dataFrame = du.load_paysim_data()

dataFrameFull = du.dataCleaning(dataFrame)

dataFrameWithoutLabel = dataFrameFull[0]
label = dataFrameFull[3]

X_train, X_test, y_train, y_test = train_test_split(dataFrameWithoutLabel, label, test_size=0.2, random_state=0)

def accuracy_knn(N):
    classifier = KNeighborsClassifier(n_neighbors=N)
    classifier.fit(X_train,y_train)

    # Apply model to validation data
    y_predict = classifier.predict(X_test)

    # Accuracy
    #print('Accuracy Score:', metrics.accuracy_score(y_test, y_predict))
    print('Accuracy Score:', metrics.classification_report(y_test, y_predict))

    from sklearn.metrics import confusion_matrix
    confusion_matx = confusion_matrix(y_test, y_predict)
    print(confusion_matx)



for i in range(3,5):
    print('********************** Decision Tree depth : ',i,'**************************\n')
    accuracy_knn(i)
