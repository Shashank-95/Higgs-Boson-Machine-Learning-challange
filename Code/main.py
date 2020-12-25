import utils
from preprocessing import PreProcessing
from DimensionalityAdjustment import DimensionalityReduction
from Models import NaiveBayesClassifier, LogRegression, DecTrees, RandForest, AdaBoost, GradientBoost


#Read the train data
print("=======Reading the Training data========")
train_data = utils.pd.read_csv('./training.csv')
train_data = train_data.replace(-999.000,utils.np.nan)

#Split the 250K samples for training and validation
labels = train_data['Label']
FeatureMatrix = train_data
FeatureMatrix = FeatureMatrix.drop(['EventId'], axis=1)
FeatureMatrix = FeatureMatrix.drop(['Label'], axis=1)
X_train, X_val, y_train, y_val = utils.train_test_split(FeatureMatrix, labels, test_size=0.2, random_state=42)
train_weights = X_train['Weight'].values
val_weights = X_val['Weight'].values

X_train = X_train.drop(['Weight'], axis=1)
X_val = X_val.drop(['Weight'], axis=1)


#Preprocess
print("=======Preprocessing In progress========")
train_preprocessed_final, val_preprocessed_final, y_train_enc, y_val_enc = PreProcessing(X_train, X_val, y_train, y_val)

#Dimension Reduction
print("=======Dimensionality Reduction In progress========")
PCAtrain, PCAVal = DimensionalityReduction(train_preprocessed_final, val_preprocessed_final)

#Training
print("=======Training starts========")
nb = NaiveBayesClassifier(PCAtrain, PCAVal, y_train_enc, y_val_enc,train_weights, val_weights)
logistic_reg= LogRegression(PCAtrain, PCAVal, y_train_enc, y_val_enc,train_weights, val_weights)
tree_classifier = DecTrees(train_preprocessed_final, val_preprocessed_final, y_train_enc, y_val_enc,train_weights, val_weights)
rf = RandForest(train_preprocessed_final, val_preprocessed_final, y_train_enc, y_val_enc,train_weights, val_weights)
ada = AdaBoost(train_preprocessed_final, val_preprocessed_final, y_train_enc, y_val_enc,train_weights, val_weights)
gmb = GradientBoost(train_preprocessed_final, val_preprocessed_final, y_train_enc, y_val_enc,train_weights, val_weights)
print("=======Training Finished========")

#Process the test set, get the predictions on testset using final model and prepare submission.csv file

print("=======Reading the Test data========")
test_data = utils.pd.read_csv('./test.csv')
test_data = test_data.replace(-999.000,utils.np.nan)
test_data = test_data.drop(['EventId'], axis=1)
print(test_data.shape)

#passing y_train and y_val is redundant here.
print("=======Predicting the labels on test data========")
train_preprocessed_final, test_preprocessed_final,y_train_enc, y_val_enc = PreProcessing(X_train, test_data, y_train, y_val)
print(test_preprocessed_final.shape)
pred = rf.predict(test_preprocessed_final)


#Create submission file
sub = utils.pd.read_csv('./random_submission.csv')
test_predict = utils.pd.Series(pred)

test_predict = utils.pd.DataFrame({"EventId":sub['EventId'],"RankOrder":sub['RankOrder'],"Class":test_predict})
test_predict = test_predict.replace(1,'s')
test_predict = test_predict.replace(0,'b')
test_predict['RankOrder'] = test_predict['Class'].argsort().argsort() + 1 
test_predict.to_csv("submission_rf.csv",index=False)
print("=======Submission file is ready in the root directory========")
