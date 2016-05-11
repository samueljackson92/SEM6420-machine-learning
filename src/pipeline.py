from roc_analysis import ROCAnalysisScorer
from sklearn import cross_validation
from sklearn.metrics import make_scorer
from unbalanced_dataset import SMOTE, UnderSampler, OverSampler


def cv_pipeline(model, x_data, y_data, cv=None):
    """ Cross Validate a model pipeline

    Args:
        model: A sklearn Pipeline object to cross validate
        x_data: Feature matrix
        y_data: Class labels
        cv: A predefined sklearn cross validation object
    Returns:
        A ROCAnalysisScorer object with the true/false positive rates and AUCs
        for all folds
    """
    roc_data = ROCAnalysisScorer()
    roc_data_scorer = make_scorer(roc_data, greater_is_better=True, needs_proba=True, average='weighted')
    cross_validation.cross_val_score(model, x_data, y_data, cv=cv, scoring=roc_data_scorer)
    return roc_data


def test_pipeline(model, x_data, y_data, x_test, y_test):
    """ Test a model on a data

    Args:
        model: A sklearn Pipeline object to test
        x_data: Feature matrix
        y_data: Class labels
    Returns:
        A ROCAnalysisScorer object with the true/fals positive rates and AUCS
        for the test
    """
    model.fit(x_data, y_data)
    y_hat = model.predict_proba(x_test)

    test_result = ROCAnalysisScorer()
    test_result.auc_score(y_test, y_hat)
    return test_result


def score_pipeline(data, cv=None):
    """ Score a pipline using either cross validation (if a cross validation
    object is provided) or using a training/test split

    Args:
        data: A dictionary object with the model and training or testing data
        cv: Optional cross validation object to use
    Returns:
        A tuple of cross validation results and testing results
    """

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
    """ Run cross validation on a set of models n times

    All models are tested using the same cross validation splits
    at each iteration.

    Args:
        models: List of dictionaries containing the model
        and training or testing data.
        n: number of iterations to repeat cross validation (default 10)
        k: number of folds to use at each iteration (default 5)
    Returns:
        A list of scorer objects of type ROCAnalysisScorer, one for each model
        passed.
    """

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
            scorers[model_name].f1scores_ += results[0].f1scores_
            scorers[model_name].f2scores_ += results[0].f2scores_
            scorers[model_name].fhalf_scores_ += results[0].fhalf_scores_
            scorers[model_name].rates_ += results[0].rates_
            scorers[model_name].aucs_ += results[0].aucs_

    return scorers


def monte_carlo_validation(x_data, y_data, models, splitter, n=10):
    """ Run Monte Carlo cross validation on a set of models n times.

    This will randomly split the training and test data n times
    and evaluate the performance of each model on each split.

    Args:
        x_data: Feature matrix
        y_data: Class labels
        models: List of dictionaries containing the model and
        training or testing data.
        splitter: A test splitter object that creates random training
        test splits.
        n: number of iterations to perform (default 10)
    Returns:
        A list of scorer objects of type ROCAnalysisScorer, one for each
        model passed.
    """
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
            scorers[model_name].f1scores_ += results[1].f1scores_
            scorers[model_name].f2scores_ += results[1].f2scores_
            scorers[model_name].fhalf_scores_ += results[1].fhalf_scores_
            scorers[model_name].rates_ += results[1].rates_
            scorers[model_name].aucs_ += results[1].aucs_

    return scorers


class TestSplitter(object):
    """ TestSplitter base class.

    This splits a feature matrix and class labels vector
    into a training and testing split. This can be used to
    create more complicated splitters for resampling.
    """
    def __init__(self, test_size=0.2):
        self._test_size = test_size

    def split(self, x_data, y_data):
        Xtest, Xvalid = cross_validation.train_test_split(x_data, test_size=self._test_size, stratify=y_data)
        Ytest, Yvalid = y_data.loc[Xtest.index], y_data.loc[Xvalid.index]
        return Xtest, Ytest, Xvalid, Yvalid


class SMOTESplitter(TestSplitter):
    """ Test splitter for SMOTE datasets.

    This splitter will apply smote the the training portion of the dataset
    but will leave the testing part of the split untouched.
    """
    def __init__(self, under_sample=1.0, smote_params={}, **kwargs):
        super(SMOTESplitter, self).__init__(**kwargs)
        self._under_sample = under_sample
        self._smote_params = smote_params

    def split(self, x_data, y_data):
        Xt, Yt, Xv, Yv = super(SMOTESplitter, self).split(x_data, y_data)
        Xt_smote, Yt_smote = SMOTE(**self._smote_params).fit_transform(Xt.as_matrix(), Yt.as_matrix())
        Xt_smote, Yt_smote = UnderSampler(ratio=self._under_sample).fit_transform(Xt_smote, Yt_smote)
        return Xt_smote, Yt_smote, Xv, Yv


class OverUnderSplitter(TestSplitter):
    """ Test splitter for under and/or over sampling datasets.

    This splitter will apply under and/or over sampling the the training
    portion of the dataset but will leave the testing part of the split
    untouched.
    """
    def __init__(self, under_sample=1.0, over_sample=1.0, **kwargs):
        super(OverUnderSplitter, self).__init__(**kwargs)
        self._under_sample = under_sample
        self._over_sample = over_sample

    def split(self, x_data, y_data):
        Xt, Yt, Xv, Yv = super(OverUnderSplitter, self).split(x_data, y_data)
        Xt_smote, Yt_smote = OverSampler(ratio=self._over_sample).fit_transform(Xt.as_matrix(), Yt.as_matrix())
        Xt_smote, Yt_smote = UnderSampler(ratio=self._under_sample).fit_transform(Xt_smote, Yt_smote)
        return Xt_smote, Yt_smote, Xv, Yv

