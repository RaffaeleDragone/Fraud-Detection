import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from it.unisa.ds.test import util_cleaning as util
import time

import warnings
warnings.filterwarnings("ignore")




#Step2 : dataFrame step1 - isFlaggedFraud
#dataFrameStep2 = util.step_2(dataFrame)

#Step3 : dataFrame step2 - nameOrig
#dataFrameStep3 = util.step_3(dataFrame)

#Step4 : dataFrame step3 - nameDest
#dataFrameStep4 = util.step_4(dataFrame)

#Step5 : dataFrame step4 + features engineering - FINAL
#dataFrameStep5 = util.step_5(dataFrame)


def comparing_step1(df):

    #dataFrame vergine
    simpleDf = util.simple_df(df)

    X = simpleDf[0]
    Y = simpleDf[1]
    decision_tree_model(7,X,Y)

    # Step1 : dataFrame con soltanto CASH_OUT e TRASFER
    dataFrameStep1 = util.step_1(df)

    X2 = dataFrameStep1[0]
    Y2 = dataFrameStep1[1]
    decision_tree_model(7, X2, Y2)

def comparing_step2(df):

    # Step1 : dataFrame con soltanto CASH_OUT e TRASFER
    dataFrameStep1 = util.step_1(df)

    X = dataFrameStep1[0]
    Y = dataFrameStep1[1]
    decision_tree_model(7, X, Y)

    # Step2 : dataFrame con soltanto CASH_OUT e TRASFER - IsFlaggedFraud
    dataFrameStep2 = util.step_2(df)

    X2 = dataFrameStep2[0]
    Y2 = dataFrameStep2[1]
    decision_tree_model(7, X2, Y2)

def comparing_step3(df):

    # Step1 : dataFrame con soltanto CASH_OUT e TRASFER
    dataFrameStep2 = util.step_2(df)

    X = dataFrameStep2[0]
    Y = dataFrameStep2[1]
    decision_tree_model(7, X, Y)

    # Step2 : dataFrame con soltanto CASH_OUT e TRASFER - IsFlaggedFraud
    dataFrameStep3 = util.step_3(df)

    X2 = dataFrameStep2[0]
    Y2 = dataFrameStep2[1]
    decision_tree_model(7, X2, Y2)

def comparing_step4(df):

    # Step1 : dataFrame con soltanto CASH_OUT e TRASFER
    dataFrameStep3 = util.step_3(df)

    X = dataFrameStep3[0]
    Y = dataFrameStep3[1]
    decision_tree_model(7, X, Y)

    # Step2 : dataFrame con soltanto CASH_OUT e TRASFER - IsFlaggedFraud
    dataFrameStep4 = util.step_4(df)

    X2 = dataFrameStep4[0]
    Y2 = dataFrameStep4[1]
    decision_tree_model(7, X2, Y2)

def comparing_step5(df):
        # Step1 : dataFrame con soltanto CASH_OUT e TRASFER
        dataFrameStep4 = util.step_4(df)

        X = dataFrameStep4[0]
        Y = dataFrameStep4[1]
        decision_tree_model(7, X, Y)

        # Step2 : dataFrame con soltanto CASH_OUT e TRASFER - IsFlaggedFraud
        dataFrameStep5 = util.step_5(df)

        X2 = dataFrameStep5[0]
        Y2 = dataFrameStep5[1]
        decision_tree_model(7, X2, Y2)

def decision_tree_model(depth, dfX, dfY):
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, test_size=0.2, random_state=42,
                                                        stratify=dfY)

    print('Transazioni Fraudolente TRAINING SET ', len(y_train.loc[(y_train == 1)]))
    print('Transazioni Fraudolente TEST SET ', len(y_test.loc[(y_test == 1)]))

    print('Transazioni NON Fraudolente TRAIN SET ', len(y_train.loc[(y_train == 0)]))
    print('Transazioni NON Fraudolente TEST SET ', len(y_test.loc[(y_test == 0)]))

    # Build and train model
    # criterion = "gini | default " - entropy ".
    # splitter = "best | default " - "random".
    # max_depth = "None | default " -
    #model = DecisionTreeClassifier(max_depth=depth)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Applichiamo il modello ai dati
    y_predict = model.predict(X_test)

    # Accuratezza
    print('Accuracy Score:', metrics.accuracy_score(y_test, y_predict))
    print('Accuracy Score Decision Tree: \n', metrics.classification_report(y_test, y_predict))

    confusion_matx = confusion_matrix(y_test, y_predict)
    print(confusion_matx)

    export_graphviz(model, out_file='classifier_graph.dot', class_names=True,
                    feature_names=dfX.columns, rounded=True, filled=True)
    end_time = time.time()
    print(' - - - - %s seconds - - - ' % (end_time-start_time))

dataFrame = util.load_paysim_data()
#comparing_step1(dataFrame)

comparing_step2(dataFrame)
