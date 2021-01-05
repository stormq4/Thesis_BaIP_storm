from sklearn.metrics import roc_auc_score, roc_curve

Cost_Reduction = 5100

def classifier(X_test_total, Y, w, b):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    i = 0
    pred = []
    for z in X_test_total:
        z = z[None, :]

        clasi = z @ w + b

        if clasi > 0 and Y[i] == 1:
            TP += 1
            pred.append(1)

        if clasi < 0 and Y[i] == 1:
            FN += 1
            pred.append(-1)

        if clasi > 0 and Y[i] == -1:
            FP += 1
            pred.append(1)

        if clasi < 0 and Y[i] == -1:
            TN += 1
            pred.append(-1)
        i += 1
    fpr, tpr, thres = roc_curve(Y, pred, pos_label=1)
    AUC_ROC = roc_auc_score(Y, pred)
    CR_QALY = Cost_Reduction * TP

    auc = fpr, tpr

    return CR_QALY, auc, AUC_ROC