#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[9]:


from sklearn.model_selection import train_test_split

X = data_breast_cancer.data
y = data_breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[10]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

X_train_sel = X_train[['mean texture', 'mean symmetry']]
X_test_sel = X_test[['mean texture', 'mean symmetry']]

clf_tree = DecisionTreeClassifier()
clf_log = LogisticRegression()
clf_knn = KNeighborsClassifier()

voting_clf_hard = VotingClassifier(estimators=[('lr', clf_log), ('dt', clf_tree), ('knn', clf_knn)], voting='hard')

voting_clf_soft = VotingClassifier(estimators=[('lr', clf_log), ('dt', clf_tree), ('knn', clf_knn)], voting='soft')


# In[11]:


from sklearn.metrics import accuracy_score
import pickle

results = []
clfs = [clf_tree, clf_log, clf_knn, voting_clf_hard, voting_clf_soft]

for clf in clfs:
    clf.fit(X_train_sel, y_train)
    y_train_pred = clf.predict(X_train_sel)
    y_test_pred = clf.predict(X_test_sel)
    results.append((accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)))

with open('acc_vote.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('vote.pkl', 'wb') as f:
    pickle.dump(clfs, f)

results


# In[12]:


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

base_clf = DecisionTreeClassifier()

bagging = BaggingClassifier(base_clf, n_estimators=30, bootstrap=True)
bagging_50 = BaggingClassifier(base_clf, n_estimators=30, max_samples=0.5, bootstrap=True)
pasting = BaggingClassifier(base_clf, n_estimators=30, bootstrap=False)
pasting_50 = BaggingClassifier(base_clf, n_estimators=30, max_samples=0.5, bootstrap=False)
rnd_for= RandomForestClassifier(n_estimators=30)
ada = AdaBoostClassifier(n_estimators=30)
gradient = GradientBoostingClassifier(n_estimators=30)

models = [bagging, bagging_50, pasting, pasting_50, rnd_for, ada, gradient]
results = []

for model in models:
    model.fit(X_train_sel, y_train)
    results.append((accuracy_score(y_train, model.predict(X_train_sel)), accuracy_score(y_test, model.predict(X_test_sel))))

with open("acc_bag.pkl", 'wb') as f:
    pickle.dump(results, f)

with open("bag.pkl", 'wb') as f:
    pickle.dump(models, f)

results


# In[17]:


bagging_fea = BaggingClassifier(n_estimators=30, max_features=2, bootstrap_features=False, max_samples=0.5, bootstrap=True)

bagging_fea.fit(X_train, y_train)
acc = [accuracy_score(y_train, bagging_fea.predict(X_train)), accuracy_score(y_test, bagging_fea.predict(X_test))]

with open("acc_fea.pkl", 'wb') as f:
    pickle.dump(acc, f)

with open("fea.pkl", 'wb') as f:
    pickle.dump([bagging_fea], f)

print(acc)
print(bagging_fea)


# In[14]:


import pandas as pd

feature_list = X_train.columns.to_numpy()
rows = []

for est, fea in zip(bagging_fea.estimators_, bagging_fea.estimators_features_):
    X_train_sel = X_train.iloc[:, fea]
    X_test_sel = X_test.iloc[:, fea]
    selected_features = feature_list[fea]
    rows.append({
        'acc_train': accuracy_score(y_train, est.predict(X_train_sel)),
        'acc_test': accuracy_score(y_test, est.predict(X_test_sel)),
        'features': list(selected_features)
    })

df = pd.DataFrame(rows)
df.sort_values(by=['acc_test', 'acc_train'], ascending=False, inplace=True)
#df.reset_index(drop=True, inplace=True)

with open('acc_fea_rank.pkl', 'wb') as f:
    pickle.dump(df, f)

df


# In[14]:




