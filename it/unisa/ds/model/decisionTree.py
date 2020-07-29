from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(dataFrameWithoutLabel, label, test_size=0.2, random_state=0, stratify= label)
'''
def accuracy_dectree(depth):
    # Build and train model
    # criterion = "gini | default " - entropy ".
    # splitter = "best | default " - "random".
    # max_depth = "None | default " -
    model = DecisionTreeClassifier(max_depth=depth)

    randomForest = RandomForestClassifier(max_depth=depth)

    model.fit(X_train, y_train)

    randomForest.fit(X_train,y_train)


    # Apply model to validation data
    y_predict = model.predict(X_test)
    y_predictForest = randomForest.predict(X_test)

    # Accuracy
    #print('Accuracy Score:', metrics.accuracy_score(y_test, y_predict))
    print('Accuracy Score Decision Tree:', metrics.classification_report(y_test, y_predict))

    print('Accuracy Score Random Forest:', metrics.classification_report(y_test, y_predictForest)) 
    from sklearn.metrics import confusion_matrix
    confusion_matx = confusion_matrix(y_test, y_predict)
    print(confusion_matx)

    file_name='result_graph_depth'+str(i)+'.dot'
    export_graphviz(model, out_file=file_name, class_names=True,
                    feature_names=dataFrameWithoutLabel.columns, rounded=True, filled=True)
    '''
#for i in range(1,15):
    #print('********************** Decision Tree depth : ',i,'**************************\n')
    #accuracy_dectree(i)

#accuracy_dectree(10)


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


# fitting trees of depth 1 to 24
sm_tree_depths = range(1, 20)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, y_train,
                                                                                        sm_tree_depths)

# plotting accuracy
plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores,
                               'Accuracy per decision tree depth on training data')

idx_max = sm_cv_scores_mean.argmax()
sm_best_tree_depth = sm_tree_depths[idx_max]
sm_best_tree_cv_score = sm_cv_scores_mean[idx_max]
sm_best_tree_cv_score_std = sm_cv_scores_std[idx_max]
print('The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(
      sm_best_tree_depth, round(sm_best_tree_cv_score*100,5), round(sm_best_tree_cv_score_std*100, 5)))

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