from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import pandas as pd
from it.unisa.ds.util import dataUtility as du

dataFrame = du.load_paysim_data()

dataFrameFull = du.dataCleaning(dataFrame)

dataFrameWithoutLabel = dataFrameFull[0]
label = dataFrameFull[3]

X_train, X_test, y_train, y_test = train_test_split(dataFrameWithoutLabel, label, test_size=0.2, random_state=0)

def accuracy_dectree(depth):

    #DecisionTree

    # Build and train model
    # criterion = "gini | default " - entropy ".
    # splitter = "best | default " - "random".
    # max_depth = "None | default " -
    model = DecisionTreeClassifier(max_depth=depth)


    model.fit(X_train, y_train)



    # Apply model to validation data
    y_predict = model.predict(X_test)


    # Accuracy of Decision Tree
    acc = metrics.accuracy_score(y_test, y_predict)
    prec = metrics.precision_score(y_test, y_predict)
    rec = metrics.recall_score(y_test, y_predict)
    f1 = metrics.f1_score(y_test, y_predict)

    results = pd.DataFrame([['Decision tree', acc, prec, rec, f1]],
                           columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])


    #Random Forest

    randomForest = RandomForestClassifier(random_state=0, n_estimators=100,max_depth=depth)
    randomForest.fit(X_train, y_train)

    y_predictForest = randomForest.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_predictForest)
    prec = metrics.precision_score(y_test, y_predictForest)
    rec = metrics.recall_score(y_test, y_predictForest)
    f1 = metrics.f1_score(y_test, y_predictForest)
    model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
                                 columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    results = results.append(model_results, ignore_index=True)

    #ANN
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu', input_dim=29))

    # Adding the second hidden layer
    classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))

    # Adding the output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size=32, epochs=100)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    #y_pred = (y_pred > 0.5)
    score = classifier.evaluate(X_test, y_test)
    print(score)

def annClass(x):
    # ANN
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu', input_dim=29))

    # Adding the second hidden layer
    classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))

    # Adding the output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size=32, epochs=100)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # y_pred = (y_pred > 0.5)
    score = classifier.evaluate(X_test, y_test)
    print(score)

#for i in range(1,15):
    #print('********************** Decision Tree depth : ',i,'**************************\n')
    #accuracy_dectree(i)

annClass(1)

#accuracy_dectree(5)



#Confusion Matrix

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_predict)
#print(cm)

#export_graphviz(model, out_file='classifier_graph.dot', class_names=True,feature_names=dataFrameWithoutLabel.columns, rounded=True, filled=True)

'''
# Compare actual and predicted values
actual_vs_predict = pd.DataFrame({'Actual': y_val,
                                'Prediction': y_predict})
print(actual_vs_predict.sample(12))


prob= model.predict_proba([[1,1,181.0,181.0,0.0,21182.0,0.0]])
print("Prob: {}".format(prob))

pred= model.predict([[1,1,181.0,181.0,0.0,21182.0,0.0]])
print("Pred: {}".format(pred))

# Evaluate model
print('Classification metrics: \n', classification_report(y_val, y_predict))

dot_data = export_graphviz(model, out_file=None, class_names=True,feature_names=dataFrameWithoutLabel.columns, rounded=True, filled=True)

graph = pydotplus.graph_from_dot_data(dot_data)
#display(Image(data=graph.create_pdf()))
display(data=graph.create_pdf())
'''
'''
tree_clf= DecisionTreeClassifier(max_depth=2,random_state=42)
tree_clf.fit(dataFrameWithoutLabel, label)

export_graphviz(tree_clf, out_file="decisiontree.dot", class_names=True,feature_names=dataFrameWithoutLabel.columns, rounded=True, filled=True)
path = 'decisiontree.dot'
s = Source.from_file(path)
s.view()

prob= tree_clf.predict_proba([[1,1,181.0,181.0,0.0,21182.0,0.0]])
print("Prob: {}".format(prob))

pred= tree_clf.predict([[1,1,181.0,181.0,0.0,21182.0,0.0]])
print("Pred: {}".format(pred))
'''