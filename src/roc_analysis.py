from sklearn import metrics
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

class ROCAnalysisScorer:
    def __init__(self):
        self.rates_ = []
        self.aucs_ = []

    def __call__(self, ground_truth, predictions, **kwargs):
        return self.auc_score(ground_truth, predictions, **kwargs)

    def auc_score(self, ground_truth, predictions, **kwargs):
        fpr, tpr, _ = metrics.roc_curve(ground_truth, predictions[:, 1])
        area = metrics.auc(fpr, tpr)

        self.rates_.append((fpr, tpr))
        self.aucs_.append(area)
        return area
    
    def mean_roc_metrics(self):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for fpr, tpr in self.rates_:
            mean_tpr += interp(mean_fpr, fpr, tpr) 

        mean_tpr = mean_tpr / len(self.rates_)
        area =  metrics.auc(mean_fpr, mean_tpr)
        return mean_fpr, mean_tpr, area

    
    def plot_roc_curve(self, title=None, labels=None, chance_line=False, mean_line=False):
        for i, ((fpr, tpr), area) in enumerate(zip(self.rates_, self.aucs_)):
            if labels is None:
                name = "ROC %d" % (i+1)
            elif isinstance(labels, list):
                name = labels[i]
            elif isinstance(labels, str):
                name = labels
            plt.plot(fpr, tpr, label='%s AUC = %0.4f' % (name, area))

        if mean_line and len(self.rates_) > 1:
            mean_fpr, mean_tpr, area = self.mean_roc_metrics()
            plt.plot(mean_fpr, mean_tpr, "--", label="Mean, AUC: %.4f" % area)

        if chance_line:
            line = np.arange(0, 1.1, 0.1)
            plt.plot(line, line, "--", label="Chance")

        if title is None:
            title = 'Receiver operating characteristic' 
        
        plt.title(title)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", prop={'size': 10})


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    width, height = cm.shape
    for x in xrange(width):
        for y in xrange(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center', size=20)

    plt.title(title)
    plt.colorbar()
    plt.show()
    
