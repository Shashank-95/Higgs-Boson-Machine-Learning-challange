import utils

def AMS(cum_weights, data_weights, y, pred ):
    data_weights = data_weights * cum_weights / sum(data_weights)
    s = data_weights * (y == 1) * (pred == 1)
    b = data_weights * (y == 0) * (pred == 1)
    signal = utils.np.sum(s)
    background = utils.np.sum(b)
    br = 10.0
    AMS_val = utils.math.sqrt(2 * ((signal + background + 10) * utils.math.log(1.0 + signal/(background + 10)) - signal))
    return AMS_val


def evaluation(pred_train, true_labels_train, pred_val, true_labels_val, sum_weights, train_weights, val_weights):
    fpr, tpr, thresholds = utils.metrics.roc_curve(true_labels_val, pred_val)
    fpr_tr, tpr_tr, thresholds_tr = utils.metrics.roc_curve(true_labels_train, pred_train)
    auc_val = utils.metrics.auc(fpr, tpr)
    auc_train = utils.metrics.auc(fpr_tr, tpr_tr)
    sum_weights = sum(train_weights)+sum(val_weights)
    AMS_score_val  = AMS(sum_weights, val_weights, true_labels_val, pred_val)
    AMS_score_tr  = AMS(sum_weights, train_weights, true_labels_train, pred_train)

    return auc_val, auc_train, AMS_score_val, AMS_score_tr 
