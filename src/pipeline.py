from roc_analysis import ROCAnalysisScorer
from sklearn import cross_validation
from sklearn.metrics import make_scorer

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
        
def score_pipelines(models, cv=None):
    results = {}
    for data in models:
        name = data['name']
        model = data['model']
        
        cv_results = None
        test_results = None

        x_data, y_data = data['train_data']
        
        if cv is not None:
            cv_results = cv_pipeline(model, x_data, y_data, cv=cv)
        
        if 'test_data' in data:
            x_test, y_test = data['test_data']
            test_results = test_pipeline(model, x_data, y_data, x_test, y_test)

        results[name] = (cv_results, test_results)
    return results

