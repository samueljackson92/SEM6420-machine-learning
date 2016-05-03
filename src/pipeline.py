from roc_analysis import ROCAnalysisScorer
from sklearn import cross_validation
from sklearn.metrics import make_scorer
from unbalanced_dataset import SMOTE, UnderSampler

def cv_pipeline(model, x_data, y_data, cv=None):
    roc_data = ROCAnalysisScorer()
    roc_data_scorer = make_scorer(roc_data, greater_is_better=True, needs_proba=True, average='weighted')
    cross_validation.cross_val_score(model, x_data, y_data, cv=cv, scoring=roc_data_scorer)
    return roc_data

def test_pipeline(model, x_data, y_data, x_test, y_test):
    model.fit(x_data, y_data)
    y_hat = model.predict_proba(x_test)

    test_result = ROCAnalysisScorer()
    test_result.auc_score(y_test, y_hat)
    return test_result
   

def score_pipeline(data, cv=None):
    model = data['model']
        
    cv_results = None
    test_results = None

    x_data, y_data = data['train_data']
    
    if cv is not None:
        cv_results = cv_pipeline(model, x_data, y_data, cv=cv)
    
    if 'test_data' in data:
        x_test, y_test = data['test_data']
        test_results = test_pipeline(model, x_data, y_data, x_test, y_test)

    return (cv_results, test_results)


def repeated_cross_fold_validation(models, n=10, k=5):
    scorers = {}

    for i in range(n):
        # create a new cross validation set for each iteration & test.
        skf = cross_validation.StratifiedKFold(models[0]['train_data'][1], n_folds=k)

        for model in models:
            model_name = model['name']
            if model_name not in scorers:
                scorers[model_name] = ROCAnalysisScorer()
             
            results = score_pipeline(model, cv=skf)

            # for each model collect the results into a single scorer.
            # note: no average is made at this stage. The results of each
            # of the k folds is collected into a single k * n list for
            # the model.
            scorers[model_name].rates_ += results[0].rates_
            scorers[model_name].aucs_ += results[0].aucs_

    return scorers


def monte_carlo_validation(x_data, y_data, models, splitter, n=10):
    scorers = {}
    
    for i in range(n):
        x_train, y_train, x_valid, y_valid = splitter.split(x_data, y_data)

        for model in models:
            model_name = model['name']
            if model_name not in scorers:
                scorers[model_name] = ROCAnalysisScorer()
            
            model['train_data'] = (x_train, y_train)
            model['test_data'] = (x_valid, y_valid)

            results = score_pipeline(model)

            # for each model collect the results into a single scorer.
            # note: no average is made at this stage. The results of each
            # of the k folds is collected into a single k * n list for
            # the model.
            scorers[model_name].rates_ += results[1].rates_
            scorers[model_name].aucs_ += results[1].aucs_

    return scorers


class TestSplitter(object):

    def __init__(self, test_size=0.2):
        self._test_size = test_size

    def split(self, x_data, y_data):
        Xtest, Xvalid = cross_validation.train_test_split(x_data, test_size=self._test_size, stratify=y_data)
        Ytest, Yvalid = y_data.loc[Xtest.index], y_data.loc[Xvalid.index]
        return Xtest, Ytest, Xvalid, Yvalid


class SMOTESplitter(TestSplitter):

    def __init__(self, under_sample=1.0, smote_params={}, **kwargs):
        super(SMOTESplitter, self).__init__(**kwargs)
        self._under_sample = under_sample
        self._smote_params = smote_params

    def split(self, x_data, y_data):
        Xt, Yt, Xv, Yv = super(SMOTESplitter, self).split(x_data, y_data)
        Xt_smote, Yt_smote = SMOTE(**self._smote_params).fit_transform(Xt.as_matrix(), Yt.as_matrix())
        Xt_smote, Yt_smote = UnderSampler(ratio=self._under_sample).fit_transform(Xt_smote, Yt_smote)
        return Xt_smote, Yt_smote, Xv, Yv

