
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pipeline
from roc_analysis import ROCAnalysisScorer


# ### Loading the Datasets

# In[197]:

df = pd.DataFrame.from_csv("../data/train_risk.csv", index_col=False)
test = pd.DataFrame.from_csv("../data/test_risk.csv", index_col=False)
X, y = df[df.columns[:-1]], df[df.columns[-1]]


# ## Analysing the Data
#
# Looking at the difference between the number of positive and negative samples in the dataset shows that there are more negative examples than positive examples. Only 28% of all samples are of the positive class.

# In[3]:

def class_balance_summary(y):
    """ Summarise the imbalance in the dataset"""
    total_size = y.size
    negative_class = y[y == 0].size
    positive_class = y[y > 0].size
    ratio = positive_class / float(positive_class + negative_class)

    print "Total number of samples: %d" % total_size
    print "Number of positive samples: %d" % positive_class
    print "Number of negative samples: %d" % negative_class
    print "Ratio of positive to total number of samples: %.2f" % ratio


class_balance_summary(y)


# Some initial observations about the data before it is preprocessed:
#  - PRE32 is all zeros. This can be removed
#  - PRE14 looks catagorical. Should be split into multiple binary variables
#  - DGN looks catagorical. As above.
#  - PRE5 looks to have some outliers. See box plot below. Potentially remove or split into two extra variable?

# In[4]:

X.head()


# Box plot below shows the outliers in PRE5. It is worth noting that all of these outliers are of the negative class. This variable is the volume that can be exhaled in one second given full inhilation. It is likely that these values are therefore errors in reporting as it is unlikely that humans can exhale such a large volume so quickly.

# In[5]:

# X.PRE5.plot(kind='box')
X.PRE5.plot(kind='box')
print y[X.PRE5 > 30 ]


# ## Preprocessing
#
# Create a new matrix of preprocessed features. This will encode catagorical data as one hot vectors, remove outliers, and normalise the data.

# In[198]:

from sklearn import preprocessing

def encode_onehot(x_data, column_name, digitize=False):
    """ Encode a catagorical column from a data frame into a data frame of one hot features"""
    data = x_data[[column_name]]

    if digitize:
        data = np.digitize(data, np.arange(data.min(), data.max(), 10))

    enc = preprocessing.OneHotEncoder()
    features = enc.fit_transform(data).toarray()
    names = ['%s_%d' % (column_name, i) for i in enc.active_features_]
    features = pd.DataFrame(features, columns=names, index=x_data.index)
    return features


def preprocess(x_data, y_data=None):
    # drop zero var PRE32
    Xp = x_data.drop("PRE32", axis=1)

    # remove outliers
    if y_data is not None:
        mask = Xp.PRE5 < 30
        Xp = Xp.loc[mask]
        Yp = y_data.copy()
        Yp = Yp.loc[mask]
    else:
        Yp = None

    # encode catagorical data as one hot vectors
    one_hot_names = ["DGN"]
    encoded = map(lambda name: encode_onehot(Xp, name), one_hot_names)
    #combine into a single data frame
    new_features = pd.concat(encoded, axis=1)

    # drop the catagorical variables that have been encoded
    Xp.drop(["DGN"], inplace=True, axis=1)
    # add new features
    Xp = pd.concat([Xp, new_features], axis=1)

    return Xp, Yp

Xp, Yp = preprocess(X, y)
Xp.head()


# Measure the effectiveness of each feature using the variable importance measure from a Random Forest

# In[10]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def measure_importance(x_data, y_data):
    rf_selector = RandomForestClassifier(criterion='gini', class_weight='balanced')
    rf_selector.fit(StandardScaler().fit_transform(x_data), y_data)
    feature_importance = pd.Series(rf_selector.feature_importances_, index=x_data.columns).sort_values(ascending=False)
    feature_importance.plot(kind='bar')
    return feature_importance

feature_importance = measure_importance(Xp, Yp)
Xp.drop(feature_importance[feature_importance == 0].index, inplace=True, axis=1)


# In[11]:

feature_importance.plot(kind='barh')
plt.xlabel('Gini Impurity')
plt.tight_layout()
plt.savefig("img/feature_importance.png")


# The numerical features appear to be the most important ones. Plot a scatter plot matrix to see how the how the correlate with each other

# In[12]:

pd.tools.plotting.scatter_matrix(Xp[['PRE4', 'PRE5', 'AGE']], c=Yp)


# ## Tuning Model Parameters
#
# Given the current status of the data tune the model parameters to it before we evalute the overall performance. Note that all of the tuning presented here is orientated towards obtaining the highest AUC score. Other metrics might be more desirable given the problem domain, but AUC is the measurement used for assignment points.

# In[13]:

from sklearn import cross_validation
skf = cross_validation.StratifiedKFold(Yp, n_folds=5)


# ### Random Forest Tuning
# Run a grid search over a range of parameters for a Random Forest. The dataset is small enough that we can do them all at once. ```n_estimators``` is neglected because this should always improve as it is increased so we should attempt to make it as large as possible subject to lack of improvement

# In[3835]:

param_grid = {"max_depth": range(2, 20, 3),
              "max_features": range(2, 20, 3),
              "min_samples_split": range(1, 5),
              "min_samples_leaf": range(1, 5),
             }

rf = RandomForestClassifier(class_weight='balanced', n_estimators=50, random_state=50)
rf_clf = grid_search.GridSearchCV(rf, param_grid, n_jobs=-1, cv=skf, scoring='roc_auc')
rf_clf.fit(Xp, Yp)


# In[3836]:

print rf_clf.best_params_


# Now take a look at the number of estimators and see where performance begins to level off.

# In[3838]:

param_grid = {"n_estimators": range(50, 500, 50)}
const_params = {'max_features': 1, 'min_samples_split': 1, 'max_depth': 16, 'min_samples_leaf': 1}

rf = RandomForestClassifier(class_weight='balanced', n_estimators=50, random_state=50, **const_params)
rf_clf2 = grid_search.GridSearchCV(rf, param_grid, n_jobs=-1, cv=skf, scoring='roc_auc')
rf_clf2.fit(Xp, Yp)


# The best parameters for ```n_estimators``` levels off after around 300 estimators

# In[3840]:

plt.plot([d[0]['n_estimators'] for d in rf_clf2.grid_scores_], [d[1] for d in rf_clf2.grid_scores_])
print rf_clf2.best_params_
print rf_clf2.best_score_


# In[3725]:

rf_clf2.best_estimator_.get_params()


# ### Gradient Boosting Tuning
#
# Gradient boosting is difficult to tune effectively. [This guide](http://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/) suggests starting by fixing the learning rate and number of estimators to a relatively low number in order to tune the other hyperparameters. After they are optimised the learning rate is gradually lowered and the number of estimators increased until we find convergance on the optimum parameters

# In[3587]:

param_grid = [
   {'n_estimators': range(20,150,10)}
]

const_params = {'learning_rate': 0.1, 'min_samples_split': 1, 'min_samples_leaf': 3, 'max_depth': 8, 'max_features': 'sqrt', 'subsample': 0.8}
gbc = GradientBoostingClassifier(random_state=50, **const_params)

gbc_clf = grid_search.GridSearchCV(gbc, param_grid, cv=skf, scoring='roc_auc')
gbc_clf.fit(Xp, Yp)


# ```n_estimators``` plateaus at around 100, so we'll use this instead of the optimum as less trees == quicker training and we'll need to decrease the learning rate and increase the number of trees later in the tuning anyway.

# In[3588]:

plt.plot([d[0]['n_estimators'] for d in gbc_clf.grid_scores_], [d[1] for d in gbc_clf.grid_scores_])
print gbc_clf.best_params_


# Now tune the ```max_depth``` and the ```min_samples_split``` parameters.

# In[3594]:

const_params = {'n_estimators':100,
                'learning_rate': 0.1,
                'min_samples_leaf': 3,
                'max_features': 'sqrt',
                'subsample': 0.8
               }

param_grid = [
    {'max_depth':range(5,16,2), 'min_samples_split':range(1, 20, 3)}
]


gbc = GradientBoostingClassifier(random_state=50, **const_params)
gbc_clf = grid_search.GridSearchCV(gbc, param_grid, cv=skf, scoring='roc_auc')
gbc_clf.fit(Xp, Yp)


# In[3595]:

print gbc_clf.best_params_
gbc_clf.grid_scores_


# Now train ```max_features```:

# In[3600]:

const_params = {'n_estimators':100,
                'learning_rate': 0.1,
                'min_samples_leaf': 3,
                'max_features': 'sqrt',
                'max_depth': 9,
                'min_samples_split': 7,
                'subsample': 0.8
               }

param_grid = [
    {'max_features':range(5,20,2)}
]


gbc = GradientBoostingClassifier(random_state=50, **const_params)
gbc_clf = grid_search.GridSearchCV(gbc, param_grid, cv=skf, scoring='roc_auc')
gbc_clf.fit(Xp, Yp)


# In[3601]:

plt.plot([d[0]['max_features'] for d in gbc_clf.grid_scores_], [d[1] for d in gbc_clf.grid_scores_])
print gbc_clf.best_params_


# Now train to tune the ```subsample``` rate.

# In[3603]:

const_params = {'n_estimators':100,
                'learning_rate': 0.1,
                'min_samples_leaf': 3,
                'max_features': 'sqrt',
                'max_depth': 9,
                'min_samples_split': 7,
                'max_features': 11,
                'subsample': 0.8
               }

param_grid = [
    {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
]


gbc = GradientBoostingClassifier(random_state=50, **const_params)
gbc_clf = grid_search.GridSearchCV(gbc, param_grid, cv=skf, scoring='roc_auc')
gbc_clf.fit(Xp, Yp)


# In[3604]:

plt.plot([d[0]['subsample'] for d in gbc_clf.grid_scores_], [d[1] for d in gbc_clf.grid_scores_])
print gbc_clf.best_params_


# Now cross validate with all the parameters set:

# In[3614]:

const_params = {
                'min_samples_leaf': 1,
                'min_samples_split': 7,
                'max_depth': 9,
                'max_features': 11,
                'subsample': 0.8
               }

param_grid = [
    {'n_estimators': [100], 'learning_rate': [0.1]},
    {'n_estimators': [200], 'learning_rate': [0.05]},
    {'n_estimators': [1000], 'learning_rate': [0.01]},
    {'n_estimators': [1500], 'learning_rate': [0.005]},
]

gbc = GradientBoostingClassifier(random_state=50, **const_params)

gbc_clf = grid_search.GridSearchCV(gbc, param_grid, cv=skf, scoring='roc_auc')
gbc_clf.fit(Xp, Yp)


# In[3718]:

print gbc_clf.best_params_
gbc_clf.grid_scores_


# In[3854]:

p = pd.DataFrame(gbc_clf.best_estimator_.get_params(), index=['Value']).T
p.index.name = "Parameter"
print p.to_latex()


# ### AdaBoost Tuning
# Perhaps the easiest train due to a fairly limited number of parameters. Adjusting the ```max_depth``` suggests that 4 appears to be roughly the best option for the depth of the decision trees.

# In[3764]:

param_grid = {"n_estimators": range(50, 1000, 50), 'learning_rate': [0.1, 0.5, 0.01, 0.005]}

dt = DecisionTreeClassifier(class_weight='balanced', max_depth=4)
adb = AdaBoostClassifier(dt)
adb_clf = grid_search.GridSearchCV(adb, param_grid, n_jobs=-1, cv=skf, scoring='roc_auc')
adb_clf.fit(Xp, Yp)


# In[3781]:

print adb_clf.best_params_
print adb_clf.best_score_
adb_clf.grid_scores_


# ### Extremely Random Trees Tuning
#
# This is very similar to Random Forests. In fact we will start with the same parameter set for the grid search.

# In[3800]:

param_grid = {"max_depth": range(2, 20, 3),
              "max_features": range(2, 20, 3),
              "min_samples_split": range(1, 5),
              "min_samples_leaf": range(1, 5),
             }
etc = ExtraTreesClassifier(class_weight='balanced', bootstrap=True, n_estimators=50, random_state=50)
etc_clf = grid_search.GridSearchCV(etc, param_grid, n_jobs=-1, cv=skf, scoring='roc_auc')
etc_clf.fit(Xp, Yp)


# In[3801]:

print etc_clf.best_params_
print etc_clf.best_score_


# Now check increasing the number of estimators and find the drop off point

# In[3805]:

param_grid = {"n_estimators": range(50, 500, 50)}
const_params = {'max_features': 16, 'min_samples_split': 1, 'max_depth': 19, 'min_samples_leaf': 1}

etc = ExtraTreesClassifier(class_weight='balanced', bootstrap=True, random_state=50, **const_params)
etc_clf2 = grid_search.GridSearchCV(etc, param_grid, n_jobs=-1, cv=skf, scoring='roc_auc')
etc_clf2.fit(Xp, Yp)


# In[3806]:

plt.plot([d[0]['n_estimators'] for d in etc_clf2.grid_scores_], [d[1] for d in etc_clf2.grid_scores_])
print etc_clf2.best_params_
print etc_clf2.best_score_


# In[3807]:

etc_clf2.best_estimator_.get_params()


# ## Model Performance
# Test the performance of each of the models on the preprocessed dataset before trying any more complicated feature engineering/resampling. This should give us some rough baseline AUC measures to work with. Firstly, set up the models. This creates a set of pipelines for each of the models we want to use.

# In[178]:

scaler = preprocessing.StandardScaler()

# set up classifier objects
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
dct = DecisionTreeClassifier(class_weight='balanced', max_depth=4)
abt = AdaBoostClassifier(dct, n_estimators=400, learning_rate=0.5)

gbc_params = {
    'min_samples_leaf': 1,
    'min_samples_split': 7,
    'max_depth': 9,
    'max_features': 11,
    'subsample': 0.8,
    'n_estimators': 1000,
    'learning_rate': 0.01
}
gbc = GradientBoostingClassifier(**gbc_params)

exf_params = {
    'bootstrap': False,
    'class_weight': 'balanced',
    'criterion': 'gini',
    'max_depth': 19,
    'max_features': 16,
    'max_leaf_nodes': None,
    'min_samples_leaf': 1,
    'min_samples_split': 1,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 200,
    'n_jobs': 1,
    'oob_score': False,
    'random_state': 50,
    'verbose': 0,
    'warm_start': False
}


exf = ExtraTreesClassifier(**exf_params)

rf_params = {
    'bootstrap': True,
    'class_weight': 'balanced',
    'criterion': 'gini',
    'max_depth': 16,
    'max_features': 1,
    'max_leaf_nodes': None,
    'min_samples_leaf': 1,
    'min_samples_split': 1,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 300
}
rf_balanced = RandomForestClassifier(**rf_params)

# create pipelines for each model
abt_pipe = Pipeline([('scaler', scaler), ('AdaBoost', abt)])
exf_pipe = Pipeline([('scaler', scaler), ('ExtraTrees', exf)])
gbc_pipe = Pipeline([('scaler', scaler), ('GradientBoostingClassifer', gbc)])
rfs_pipe = Pipeline([('scaler', scaler), ('RandomForest', rf_balanced)])

# create list of model data
models = [
    {'name': 'AdaBoost', 'model': abt_pipe},
    {'name': 'ExtraTrees', 'model': exf_pipe},
    {'name': 'RandomForest', 'model': rfs_pipe},
    {'name': 'GradientBoost', 'model': gbc_pipe},
]

# set the same training set for all models.
# this is just the preprocessed dataset.
for model in models:
    model['train_data'] = (Xp, Yp)


# Define some useful helper functions for summarising the results of k-fold/monte carlo cross validation

# In[150]:

def f_score_summary(scorers):
    """ Create a summary of the average f-scores for all folds/trials"""
    series = []
    columns = []
    for key, scorer in scorers.iteritems():
        f_scores = [np.mean(scorer.f1scores_), np.mean(scorer.f2scores_), np.mean(scorer.fhalf_scores_)]
        s = pd.Series(f_scores, index=['F1', 'F2', 'F0.5'])
        series.append(s)
        columns.append(key)

    frame = pd.concat(series, axis = 1)
    frame.columns = columns
    return frame

def summarise_scorers(scorers):
    """ Create a summary of the scorers AUCs for all folds/trials"""
    names = [name for name in scorers.keys()]
    aucs = [scorer.aucs_ for scorer in scorers.values()]
    aucs = pd.DataFrame(np.array(aucs).T, columns=names)
    return aucs.describe()


# Perform n iterations of k fold cross validation. Here I am using 10 iterations and 5 folds at each iteration.

# In[151]:

scorers = pipeline.repeated_cross_fold_validation(models, n=10, k=5)


# Plot an ROC curve and the mean AUCs.

# In[153]:

for key, scorer in scorers.iteritems():
    scorer.plot_roc_curve(mean_label=key, mean_line=True, show_all=False)

plt.plot(np.arange(0,1.1, 0.1), np.arange(0,1.1, 0.1), '--')
plt.savefig("img/roc_cv.png")


# Plot bar chart of the F2 scores

# In[154]:

f_scores = f_score_summary(scorers)
ax = f_scores.loc['F2'].plot(kind='barh', title='F2 Measure for All Classifiers', color=['b', 'r', 'g', 'y'])
ax.set_xlabel('F2 Score')
plt.tight_layout()
plt.savefig('img/f2_score.png')


# Summarise the F scores

# In[155]:

f_scores = f_score_summary(scorers)
print f_scores.to_latex()
f_scores


# ## Feature Engineering
#
# Test creating some new features based on combinations of existing ones in the dataset. Cross validate each set of new features to see if it improves performance.

# ### Binary Features

# In[156]:

import itertools

def binary_combinations(x_data, names):
    name_pairs = itertools.combinations(names, 2)
    features = []
    for a_name, b_name in name_pairs:
        a, b = x_data[a_name], x_data[b_name]
        features.append(np.logical_xor(a, b).astype(int))
        features.append(np.logical_and(a, b).astype(int))
        features.append(np.logical_or(a, b).astype(int))

    return pd.DataFrame(np.array(features).T, index=x_data.index)

binary_features = binary_combinations(Xp, ['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE30'])
Xp_binary = pd.concat([Xp, binary_features], axis=1)
feature_importance = measure_importance(Xp_binary, Yp)
Xp_binary.drop(feature_importance[feature_importance == 0].index, inplace=True, axis=1)


# In[157]:

for model in models:
    model['train_data'] = (Xp_binary, Yp)

scorers = pipeline.repeated_cross_fold_validation(models, n=10, k=5)


# In[158]:

get_ipython().magic(u'matplotlib inline')
for key, scorer in scorers.iteritems():
    scorer.plot_roc_curve(mean_label=key, mean_line=True, show_all=False)

plt.plot(np.arange(0,1.1, 0.1), np.arange(0,1.1, 0.1), '--')
plt.savefig("img/roc_binary_features.png")


# In[159]:

f_scores = f_score_summary(scorers)
print f_scores.to_latex()
f_scores


# ### Spirometry Based Features

# In[160]:

def create_spiro_features(x_data):
    # create new feature FER
    # this is the raito of FEV1 and FVC
    FER = (x_data.PRE5 / x_data.PRE4) * 100
    FER.index = x_data.index

    # create a new feature OBS
    # this is whether the instance has a FER below 70%
    # which implies an obstructive disease.
    OBS = pd.Series(np.zeros(x_data.AGE.shape))
    OBS.index = x_data.index
    OBS.loc[FER < 70] = 1.0

    spiro = pd.concat([FER, OBS], axis=1)
    spiro.columns = ['FER', 'OBS']
    return spiro


# In[161]:

spiro_features = create_spiro_features(Xp)
Xp_spiro = pd.concat([Xp, spiro_features], axis=1)
feature_importance = measure_importance(Xp_spiro, Yp)
Xp_spiro.drop(feature_importance[feature_importance == 0].index, inplace=True, axis=1)


# In[162]:

for model in models:
    model['train_data'] = (Xp_spiro, Yp)

scorers = pipeline.repeated_cross_fold_validation(models, n=10, k=5)


# In[163]:

get_ipython().magic(u'matplotlib inline')
for key, scorer in scorers.iteritems():
    scorer.plot_roc_curve(mean_label=key, mean_line=True, show_all=False)

plt.plot(np.arange(0,1.1, 0.1), np.arange(0,1.1, 0.1), '--')
plt.savefig("img/roc_spiro_features.png")


# In[164]:

feature_importance.plot(kind='barh')
plt.xlabel('Gini Impurity')
plt.tight_layout()
plt.savefig("img/importance_spiro_features.png")


# In[165]:

f_scores = f_score_summary(scorers)
print f_scores.to_latex()
f_scores


# ### Polynomial Combinations

# In[166]:

def create_poly_features(x_data, names):
    # create new features base on Polynomials of the original best two predictors
    poly = preprocessing.PolynomialFeatures(2, include_bias=False, interaction_only=True)
    poly_features = pd.DataFrame(poly.fit_transform(x_data[names]), index=x_data.index)
    poly_features.columns = ["POLY_%d" % i for i in poly_features.columns]
    return poly_features

poly_features = create_poly_features(Xp, ['PRE4', 'PRE5'])
Xp_poly = pd.concat([Xp, poly_features], axis=1)
feature_importance = measure_importance(Xp_poly, Yp)
Xp_poly.drop(feature_importance[feature_importance == 0].index, inplace=True, axis=1)


# In[167]:

for model in models:
    model['train_data'] = (Xp_poly, Yp)

scorers = pipeline.repeated_cross_fold_validation(models, n=10, k=5)


# In[168]:

get_ipython().magic(u'matplotlib inline')
for key, scorer in scorers.iteritems():
    scorer.plot_roc_curve(mean_label=key, mean_line=True, show_all=False)

plt.plot(np.arange(0,1.1, 0.1), np.arange(0,1.1, 0.1), '--')
plt.savefig("img/roc_poly_features.png")


# In[169]:

feature_importance.plot(kind='barh')
plt.xlabel('Gini Impurity')
plt.tight_layout()
plt.savefig("img/importance_poly_features.png")


# In[170]:

f_scores = f_score_summary(scorers)
print f_scores.to_latex()
f_scores


# ## Resampling the Dataset
#
# Testing whether using resampling improves performance

# ### Testing with regular Over/Under sampling

# In[43]:

splitter = pipeline.OverUnderSplitter(test_size=0.2, under_sample=0.4, over_sample=0.8)
overunder_scorers = pipeline.monte_carlo_validation(Xp, Yp, models, splitter, n=50)


# In[45]:

for key, scorer in overunder_scorers.iteritems():
    scorer.plot_roc_curve(mean_label=key, mean_line=True, show_all=False)


# In[46]:

f_score_summary(overunder_scorers)


# In[48]:

summarise_scorers(overunder_scorers)


# ### Testing with SMOTE + Undersampling

# In[171]:

smote_params = {'kind': 'regular', 'k':3, 'ratio': 0.8, 'verbose': 1}
splitter = pipeline.SMOTESplitter(test_size=0.2, under_sample=1.0, smote_params=smote_params)
smote_scorers = pipeline.monte_carlo_validation(Xp, Yp, models, splitter, n=50)


# In[172]:

for key, scorer in smote_scorers.iteritems():
    scorer.plot_roc_curve(mean_label=key, mean_line=True, show_all=False)

plt.plot(np.arange(0,1.1, 0.1), np.arange(0,1.1, 0.1), '--')
plt.savefig("img/roc_smote.png")


# In[173]:

smote_f_scores = f_score_summary(smote_scorers)
print smote_f_scores.to_latex()
smote_f_scores


# ## Best Classifier

# In[199]:

spiro_features = create_spiro_features(Xp)
poly_features = create_poly_features(Xp, ['PRE4', 'PRE5'])
Xp_all = pd.concat([Xp, poly_features, spiro_features], axis=1)
Xp_all.drop(['DGN_1', 'DGN_8'], axis=1, inplace=True)
for model in models:
    model['train_data'] = (Xp_all, Yp)


# In[200]:

scorers = pipeline.repeated_cross_fold_validation(models, n=10, k=5)


# In[201]:

for key, scorer in scorers.iteritems():
    scorer.plot_roc_curve(mean_label=key, mean_line=True, show_all=False)

plt.plot(np.arange(0,1.1, 0.1), np.arange(0,1.1, 0.1), '--')


# In[202]:

f_scores = f_score_summary(scorers)
f_scores


# ## Predicton on Test Set
#
# Finally, based on the best combination of techniques used in the preceeding sections, and using the classifier with the best AUC performance, make probalistic predictions based on the unlabelled test data.

# In[207]:

Xtest, _ = preprocess(test, y_data=None)
Xtest = Xtest.drop(['test_id'], axis=1)

test_spiro_features = create_spiro_features(Xtest)
test_poly_features = create_poly_features(Xtest, ['PRE4', 'PRE5'])
Xtest = pd.concat([Xtest, test_spiro_features, test_poly_features], axis=1)

print Xtest.columns.size
print Xp_all.columns.size


# In[224]:

final_model = models[3]['model']
final_model.fit(Xp_all, Yp)
predicted_prob = pd.Series(final_model.predict_proba(Xtest)[:, 1])
predicted_label = pd.Series(final_model.predict(Xtest))


# In[225]:

final_submission = pd.concat([test.test_id, predicted_label, predicted_prob], axis=1)
final_submission.columns = ['test_id', 'predicted_label', 'predicted_output']
class_balance_summary(predicted_label)
final_submission

