from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

from it.unisa.ds.util import dataUtility as du

dataFrame = du.load_paysim_data()

dataFrameFull = du.dataCleaning(dataFrame)

dataFrameWithoutLabel = dataFrameFull[0]
label = dataFrameFull[3]

X_train, X_test, y_train, y_test = train_test_split(dataFrameWithoutLabel, label, test_size=0.2, random_state=0)

def accuracy_dectree(depth):
    # Build and train model
    # criterion = "gini | default " - entropy ".
    # splitter = "best | default " - "random".
    # max_depth = "None | default " -
    model = DecisionTreeClassifier(max_depth=depth)

    model.fit(X_train, y_train)

    # Apply model to validation data
    y_predict = model.predict(X_test)

    # Accuracy
    print('Accuracy Score:', metrics.accuracy_score(y_test, y_predict))
    from sklearn.metrics import confusion_matrix
    confusion_matx = confusion_matrix(y_test, y_predict)
    print(confusion_matx)

    file_name='result_graph_depth'+str(i)+'.dot'
    export_graphviz(model, out_file=file_name, class_names=True,
                    feature_names=dataFrameWithoutLabel.columns, rounded=True, filled=True)

for i in range(1,15):
    print('********************** Decision Tree depth : ',i,'**************************\n')
    accuracy_dectree(i)

