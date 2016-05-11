from sklearn import metrics
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from scipy.stats import threshold

class ROCAnalysisScorer:
    """ Custom scorer to capute both the AUC score and the ROC curves.

    A normal scorer for sklearn cannot capture multiple scores at once.
    This object allows use to calculate multiple quantiies in one pass
    of a cross validation object.

    This will also capture the F scores
    """
    def __init__(self):
        self.rates_ = []
        self.aucs_ = []
        self.fhalf_scores_ = []
        self.f2scores_ = []
        self.f1scores_ = []

    def __call__(self, ground_truth, predictions, **kwargs):
        """ Custom __call__ function to make the object look like a
        function thanks to python's duck typing.
        """
        return self.auc_score(ground_truth, predictions, **kwargs)

    def auc_score(self, ground_truth, predictions, **kwargs):
        """ Calculate the AUC score for this particular trial.

        This will also calculate the F scores and ROC curves

        Args:
            ground_truth: vector of class labels
            predictions: vector of predicted class labels

        Returns:
            AUC score for this trial
        """

        # calculate f scores
        thresholded = threshold(predictions[:, 1], threshmin=0.5)
        thresholded = threshold(thresholded, threshmax=0.5, newval=1.0).astype(int)
        fhalf_score = metrics.fbeta_score(ground_truth.astype(int), thresholded, beta=0.5)
        f2_score = metrics.fbeta_score(ground_truth.astype(int), thresholded, beta=2)
        f1_score = metrics.fbeta_score(ground_truth.astype(int), thresholded, beta=1)

        # calculate ROC curve and AUC
        fpr, tpr, _ = metrics.roc_curve(ground_truth, predictions[:, 1])
        area = metrics.auc(fpr, tpr)

        self.fhalf_scores_.append(fhalf_score)
        self.f2scores_.append(f2_score)
        self.f1scores_.append(f1_score)
        self.rates_.append((fpr, tpr))
        self.aucs_.append(area)
        return area

    def mean_roc_metrics(self):
        """ Compute the mean AUC and mean ROC curve """
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for fpr, tpr in self.rates_:
            mean_tpr += interp(mean_fpr, fpr, tpr)

        mean_tpr = mean_tpr / len(self.rates_)
        area =  metrics.auc(mean_fpr, mean_tpr)
        return mean_fpr, mean_tpr, area

    def plot_roc_curve(self, title=None, labels=None, show_all=True, chance_line=False, mean_line=False, mean_label="Mean"):
        """ Plot ROC curves for all trials attached to this object

        Args:
            title: the title for the plot
            labels: the labels to use for each line. Either list, string or None
            show_all: show all plots or only the mean line
            chance_line: show the chance line
            mean_line: show the mean line
            mean_label: label for the mean line
        """
        if show_all:
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
            plt.plot(mean_fpr, mean_tpr, label="%s, AUC: %.4f" % (mean_label, area))

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
    """ Plot a confusion matrix

    Args:
        cm: square matrix representing the confusion matrix
        title: title to show on the plot
        cmap: the colour map to use
    """
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

