from sklearn import metrics
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

class ROCData():
    def __init__(self):
        self._j = None
        self._area = None
        self._fpr = None
        self._tpr = None
        
    @property
    def j_stat(self):
        return self._j
    
    @property
    def auc(self):
        return self._area
    
    @property
    def fpr(self):
        return self._fpr
    
    @property
    def tpr(self):
        return self._tpr
    
def sensitivity_and_specificity(cm):
    tp, tn, fp, fn = cm[0,0], cm[1,1], cm[0,1], cm[1,0]
    specificity = tn / float(tn + fp)
    sensitivity = tp / float(tp + fn)
    return sensitivity, specificity

def youden_index(cm):
    sensitivity, specificity = sensitivity_and_specificity(cm)
    return sensitivity + specificity - 1

def roc_metrics(x_data, y_data, model):
    preds = model.predict_proba(x_data)[:, 1]
    preds_discrete = model.predict(x_data)

    fpr, tpr, _ = metrics.roc_curve(y_data, preds)
    area = metrics.auc(fpr, tpr)
    
    cm = metrics.confusion_matrix(y_data, preds_discrete)
    j_stat = youden_index(cm)
    
    data = ROCData()
    data.j_stat = j_stat
    data.fpr = fpr
    data.tpr = tpr
    data.auc = area
    return data

def mean_roc_metrics(mets):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, data in enumerate(mets):
        mean_tpr += interp(mean_fpr, data.fpr, data.tpr) 

    mean_tpr = mean_tpr / len(mets)
    area =  metrics.auc(mean_fpr, mean_tpr)
    return mean_fpr, mean_tpr, area

def plot_roc(mets, title="ROC Curve"):
    """ Plot an ROC cruve for the given y and yhat"""
    
    for i, data in enumerate(mets):
        plt.plot(data.fpr, data.tpr, label="ROC Split %d, AUC: %.2f, J: %.2f" % ((i+1), data.auc, data.j_stat))
    
    mean_fpr, mean_tpr, area = mean_roc_metrics(mets)
    plt.plot(mean_fpr, mean_tpr, "--", label="Mean, AUC: %.2f" % area)
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), "--", label="Chance")

    plt.title(title)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right", prop={'size':8})
    plt.show()
