from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from it.unisa.ds.util import dataUtility as du
from sklearn.model_selection import KFold, cross_val_score
dataFrame = du.load_paysim_data()

dataFrameFull = du.dataCleaningAndEngineering(dataFrame)

dataFrameWithoutLabel = dataFrameFull[0]
label = dataFrameFull[3]

X_train, X_test, y_train, y_test = train_test_split(dataFrameWithoutLabel, label, test_size=0.2, random_state=42, stratify= label)

print('Transazioni Fraudolente TRAINING SET ', len(y_train.loc[(y_train==1)]))
print('Transazioni Fraudolente TEST SET ', len(y_test.loc[(y_test==1)]))

print('Transazioni NON Fraudolente TRAIN SET ', len(y_train.loc[(y_train==0)]))
print('Transazioni NON Fraudolente TEST SET ', len(y_test.loc[(y_test==0)]))

def decision_tree_model(depth):
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
    print('Accuracy Score Decision Tree: \n', metrics.classification_report(y_test, y_predict))

    from sklearn.metrics import confusion_matrix
    confusion_matx = confusion_matrix(y_test, y_predict)
    print(confusion_matx)

    export_graphviz(model, out_file='classifier_graph.dot', class_names=True,
                    feature_names=dataFrameWithoutLabel.columns, rounded=True, filled=True)

def random_forest_model(depth,n_ests):

    randomForest = RandomForestClassifier(max_depth=depth)

    randomForest.fit(X_train,y_train)

    # Apply model to validation data
    y_predictForest = randomForest.predict(X_test)

    # Accuracy
    print('Accuracy Score Random Forest: \n ', metrics.classification_report(y_test, y_predictForest))
    print('Accuracy Score:', metrics.accuracy_score(y_test, y_predictForest))
    from sklearn.metrics import confusion_matrix
    confusion_matx = confusion_matrix(y_test, y_predictForest)
    print(confusion_matx)


#for i in range(1,15):
    #print('********************** Decision Tree depth : ',i,'**************************\n')
    #accuracy_dectree(i)




# function for fitting trees of various depths on the training data using cross-validation
def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores


# function for plotting cross-validation results
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean - 2 * cv_scores_std, cv_scores_mean + 2 * cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()
    plt.show()

def cross_validation_split(type_model):
    #type_model = 0 : decision tree | 1 : random forest
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    idx = type_model

    precision_values=[]
    recall_values=[]
    f1_values=[]
    accuracy_values=[]

    for train_index, test_index in kf.split(dataFrameWithoutLabel):


        data_train = dataFrameWithoutLabel.iloc[train_index]

        target_train = label.iloc[train_index]

        data_test = dataFrameWithoutLabel.iloc[test_index]
        target_test = label.iloc[test_index]

        if(type_model==0):
            model = DecisionTreeClassifier(max_depth=5)

            model.fit(data_train, target_train)

            # Apply model to validation data
            y_predict = model.predict(data_test)

            accuracy_values.append(metrics.accuracy_score(target_test, y_predict))

            precision_values.append(metrics.precision_score(target_test, y_predict))

            recall_values.append(metrics.recall_score(target_test, y_predict))

            f1_values.append(metrics.f1_score(target_test, y_predict))

            # Accuracy
            print('Accuracy Score:', metrics.accuracy_score(target_test, y_predict))




            print('Accuracy Score Decision Tree:', metrics.classification_report(target_test, y_predict))

            from sklearn.metrics import confusion_matrix
            confusion_matx = confusion_matrix(target_test, y_predict)
            print(confusion_matx)

            export_graphviz(model, out_file='classifier_graph'+str(idx+1)+'.dot', class_names=True,
                            feature_names=dataFrameWithoutLabel.columns, rounded=True, filled=True)


    print("Media Accuratezza : ",np.mean(accuracy_values))
    print("Media Precision : ", np.mean(precision_values))
    print("Media Recall : ", np.mean(recall_values))
    print("Media F1 Score : ", np.mean(f1_values))



decision_tree_model(5)
#cross_validation_split(0)
random_forest_model(5,50)
# fitting trees of depth 1 to 24
#sm_tree_depths = range(1, 20)
#sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, y_train,
#                                                                                        sm_tree_depths)
