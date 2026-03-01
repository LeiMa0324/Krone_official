from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def test_metrics(predictions, labels):
    auc = round(roc_auc_score(labels, predictions), 4)
    P = round(precision_score(labels, predictions), 4)
    R = round(recall_score(labels, predictions), 4)
    F1 = round(f1_score(labels, predictions), 4)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    metrics = {"auc": auc, "f1": F1,"p": P,"r": R, "tp": tp, "tn":tn, "fp": fp, "fn": fn}
    return metrics

