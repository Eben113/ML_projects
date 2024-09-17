import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as OHE, OrdinalEncoder, StandardScaler, FunctionTransformer,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier as VC, BaggingClassifier as BG, RandomForestClassifier as RND_clf
from sklearn.tree import DecisionTreeClassifier as DTC
import nnet
import math

at = pd.read_csv('datasets/titanic/train.csv')
tt = pd.read_csv('datasets/titanic/test.csv')

a = at.drop((i for i in ('Name', 'Ticket', 'Cabin',   'PassengerId')), axis=1)
t = tt.drop((i for i in ('Name', 'Ticket', 'Cabin',  'PassengerId')), axis=1)


sss = SSS(n_splits = 1, random_state = 42, test_size= 0.1)
for train_index, test_index  in sss.split(a, a['Survived']):
    train_set, test_set = a.loc[train_index], a.loc[test_index]



labels = np.array(train_set['Survived'])
Tlabels = np.array(test_set['Survived'])
train_set.drop('Survived', axis = 1, inplace = True)
test_set.drop('Survived', axis = 1, inplace = True)
print(train_set)

t_num = at.select_dtypes(include = [np.number])

def make_deck(a):
    a['Deck'] = 'nan'
    for i in [[1, 5, [3,4,5,6,7]],[2, 4, [1,2,3,4]], [3, 4, [1,2,3,4]]]:
         c = a['Fare'].iloc[(j for j in np.where(a['Pclass'] == i[0])[0])]
         pd.Series(c, index =  (j for j in np.where(a['Pclass'] == i[0])[0]))
         rang = (c.max()-c.min())/i[1]
         d = pd.cut(c, np.arange(c.min(),c.max()+1,rang), labels = i[2], right = True, include_lowest = True)
    
         a.loc[d.index, 'Deck'] = d.values
    
    return(pd.DataFrame(a['Deck']))

pipe2 = Pipeline([ ('encoder', OHE(sparse_output = False))])

pipe3 =  Pipeline([('imputer', SimpleImputer(strategy = 'most_frequent')), ('encoder', OHE(sparse_output = False))])

def feat(func_trans, inputs):
	return(['Deck'])

def Cat_namer(X, cat):
	for i in X:
		i = str(cat) + str(i)
		return X

make = make_pipeline(KNNImputer(),StandardScaler())
pipe = make_pipeline(SimpleImputer(strategy = 'most_frequent'),OrdinalEncoder())

def add():
	return make_pipeline((FunctionTransformer(make_deck, feature_names_out = feat)), KNNImputer())

agecat = make_pipeline(KNNImputer() ,StandardScaler())

preproc = ColumnTransformer([('age', agecat, ['Age']), ('Sex', pipe, ['Sex']), ('POE', pipe3, ['Embarked']), ('make', make, ['SibSp', 'Parch', 'Pclass', 'Fare']), ('Deck', add(), ['Fare', 'Pclass'])], remainder = 'passthrough', verbose_feature_names_out = False)


d = preproc.fit_transform(train_set)
print(preproc.get_feature_names_out())
d = pd.DataFrame(d, columns = preproc.get_feature_names_out(), index = train_set.index)
run = preproc.transform(test_set)
runs = pd.DataFrame(run, columns = preproc.get_feature_names_out(), index = test_set.index)

run = runs.astype('float64')
d = d.astype('float64')
svc = SVC(C = 5, degree = 1, kernel = 'rbf')
svc.fit(d,labels)

sgd = SGD(loss = 'log_loss',alpha = 0.10792857142857143, max_iter= 50, penalty = 'elasticnet')
sgd.fit(d, labels)

'''vc = VC(estimators = [('svc', SVC(random_state = 42,C = 5, degree = 1, kernel = 'rbf', probability= True)),
                      ('sgd', SGD(loss = 'log_loss', alpha = 0.10792857142857143, max_iter= 50, penalty = 'elasticnet')),
                      ('dtc', DTC())])
vc.fit(d, labels)

dtc = DTC()
dtc.fit(d, labels)

bag = BG(SVC(C = 5, degree = 1, kernel = 'rbf'), n_estimators=1500, max_samples = 100, random_state=42)
bag1 = BG(SGD(loss = 'log_loss',alpha = 0.10792857142857143, max_iter= 50, penalty = 'elasticnet'), n_estimators=1000, max_samples = 100, random_state=42)
bag2 = BG(DTC(), n_estimators=1500, max_samples = 150, random_state=42)

bag.fit(d,labels)'''
rnd_clf = RND_clf(n_estimators = 200, max_leaf_nodes = 16, random_state= 42, max_features = 7)
bag = BG(SVC(C = 5, degree = 1, kernel = 'rbf'), n_estimators=1500, max_samples = 100, random_state = 42, max_features = 5)
rnd_clf.fit(d, labels)
bag.fit(d, labels)
rnd_clf.fit(d,labels)