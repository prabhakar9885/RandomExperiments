import sklearn
import sklearn.metrics

def print_f1_score( actual, predictions ):
    f1_score_type = [ 'macro', 'micro', 'weighted', None ]
    for t in f1_score_type:
        print( "{}: {}".format(t, sklearn.metrics.f1_score( actual, predictions, average=t ),) )


def get_metrics( actual, predictions ):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.values.sum() - (FP + FN + TP)
    return { 'FP': FP, 'FN':FN, 'TP':TP, 'TN':TN };

