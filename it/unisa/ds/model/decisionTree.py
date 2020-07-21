from graphviz import Source
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from it.unisa.ds.util import dataUtility as du


dataFrame = du.load_paysim_data()

dataFrameFull = du.dataCleaning(dataFrame)

dataFrameWithoutLabel = dataFrameFull[0]
label = dataFrameFull[3]


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