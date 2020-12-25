import utils
from Evaluation import evaluation, AMS

def NaiveBayesClassifier(PCAtrain, PCAVal, y_train_enc, y_val_enc,train_weights, val_weights):
    naive_bayes = utils.GaussianNB()
    nb= naive_bayes.fit(PCAtrain, y_train_enc)
    predictions_nb = nb.predict(PCAVal)
    predictions_nb_tr = nb.predict(PCAtrain)

    sum_weights= sum(train_weights) + sum(val_weights)
    auc_val, auc_train, AMS_score_val, AMS_score_tr  = evaluation(predictions_nb_tr, y_train_enc, predictions_nb, y_val_enc, sum_weights, train_weights, val_weights)
  
    print("Gaussian Naive Bayes: AUC-Train  - ", auc_train)
    print("Gaussian Naive Bayes: AUC-validation  - ", auc_val)  
    print("Gaussian Naive Bayes: AMS-train  - ", AMS_score_tr) 
    print("Gaussian Naive Bayes: AMS-Val  - ", AMS_score_val) 
    print("====================================================================\n")

    return nb

def LogRegression(PCAtrain, PCAVal, y_train_enc, y_val_enc,train_weights, val_weights):
    logistic_reg = utils.LogisticRegression(random_state=0, max_iter= 100, C = 0.01, solver='saga').fit(PCAtrain, y_train_enc)
    predictions = logistic_reg.predict(PCAVal)
    prediction_tr = logistic_reg.predict(PCAtrain)

    sum_weights = sum(train_weights) + sum(val_weights)
    auc_val, auc_train, AMS_score_val, AMS_score_tr  = evaluation(prediction_tr, y_train_enc, predictions, y_val_enc, sum_weights, train_weights, val_weights)  
    print("Logistic Regression: AUC-Train  - ", auc_train)
    print("Logistic Regression: AUC-validation  - ", auc_val)
    print("Logistic Regression: AMS-train  - ", AMS_score_tr) 
    print("Logistic Regression: AMS-Val  - ", AMS_score_val) 
    print("====================================================================\n")
    
    return logistic_reg

def DecTrees(train_preprocessed_final, val_preprocessed_final, y_train_enc, y_val_enc,train_weights, val_weights):
    dec_tree = utils.DecisionTreeClassifier(max_depth = 10, max_features= 'sqrt', random_state=0)
    tree_classifier = dec_tree.fit(train_preprocessed_final, y_train_enc)
    prediction_dectree = tree_classifier.predict(val_preprocessed_final)
    predictions_dectree_tr = tree_classifier.predict(train_preprocessed_final)

    sum_weights = sum(train_weights) + sum(val_weights)
    auc_val, auc_train, AMS_score_val, AMS_score_tr  = evaluation(predictions_dectree_tr, y_train_enc, prediction_dectree, y_val_enc, sum_weights, train_weights, val_weights)
 
    print("Decision Tree: AUC-Train  - ", auc_train)
    print("Decision Tree: AUC-validation  - ", auc_val)  
    print("Decision Tree: AMS-train  - ", AMS_score_tr)
    print("Decision Tree: AMS-Val  - ", AMS_score_val) 
    print("====================================================================\n")

    return tree_classifier

def RandForest(train_preprocessed_final, val_preprocessed_final, y_train_enc, y_val_enc,train_weights, val_weights):
    RFModel = utils.RandomForestClassifier(n_estimators = 250, max_depth=12, max_features = 5, random_state = 0)
    rf= RFModel.fit(train_preprocessed_final, y_train_enc)
    predictions_rf = rf.predict(val_preprocessed_final)
    predictions_rf_tr = rf.predict(train_preprocessed_final)

    sum_weights = sum(train_weights) + sum(val_weights)
    auc_val, auc_train, AMS_score_val, AMS_score_tr  = evaluation(predictions_rf_tr, y_train_enc, predictions_rf, y_val_enc, sum_weights, train_weights, val_weights)
    print("Random Forest: AUC-Train  - ", auc_train) 
    print("Random Forest: AUC-validation  - ", auc_val)
    print("Random Forest: AMS-train  - ", AMS_score_tr) 
    print("Random Forest: AMS-Val  - ", AMS_score_val) 
    print("====================================================================\n")

    return rf

def AdaBoost(train_preprocessed_final, val_preprocessed_final, y_train_enc, y_val_enc,train_weights, val_weights):
    adaboost = utils.AdaBoostClassifier(n_estimators = 250, learning_rate = 0.5, random_state = 0)
    ada= adaboost.fit(train_preprocessed_final, y_train_enc)
    predictions_ada = ada.predict(val_preprocessed_final)
    predictions_ada_tr = ada.predict(train_preprocessed_final)

    sum_weights = sum(train_weights) + sum(val_weights)
    auc_val, auc_train, AMS_score_val, AMS_score_tr  = evaluation(predictions_ada_tr, y_train_enc, predictions_ada, y_val_enc, sum_weights, train_weights, val_weights)

    print("Ada Boost: AUC-Train  - ", auc_train)
    print("Ada Boost: AUC-validation  - ", auc_val) 
    print("Adaboost: AMS-train  - ", AMS_score_tr)
    print("Ada Boost: AMS-Val  - ", AMS_score_val)
    print("====================================================================\n")

    return ada
def GradientBoost(train_preprocessed_final, val_preprocessed_final, y_train_enc, y_val_enc,train_weights, val_weights):
    gradient_boost = utils.GradientBoostingClassifier(loss = 'deviance', learning_rate= 0.5,  n_estimators= 250, subsample = 0.8, max_depth = 3, max_features = 6, random_state=0)
    gbm= gradient_boost.fit(train_preprocessed_final, y_train_enc)
    predictions_gbm = gbm.predict(val_preprocessed_final)
    predictions_gbm_tr = gbm.predict(train_preprocessed_final)

    sum_weights = sum(train_weights) + sum(val_weights)
    auc_val, auc_train, AMS_score_val, AMS_score_tr  = evaluation(predictions_gbm_tr, y_train_enc, predictions_gbm, y_val_enc, sum_weights, train_weights, val_weights)
    print("Gradient Boost: AUC-Train  - ", auc_train)
    print("Gradient Boost: AUC-validation  - ", auc_val)
    print("Gradient Boost: AMS-train  - ", AMS_score_tr)
    print("Gradient Boost: AMS-Val  - ", AMS_score_val) 
    print("====================================================================\n")

    return gbm