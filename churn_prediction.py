#Import Packages
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

#Read CSV files using Pandas
churn_df = pd.read_csv('./churn.all')

#Print Info
print(churn_df.info())
print(churn_df.head())
print("\n")

#Count Rows and Columns from the dataframe:
print("Num of Rows: " + str(churn_df.shape[0])) 
print("Num of Columns: " + str(churn_df.shape[1]))

#Data Cleaning:
churn_df['voice_mail_plan'] = churn_df['voice_mail_plan'].map(lambda x: x.strip())
churn_df['intl_plan'] = churn_df['intl_plan'].map(lambda x: x.strip())
churn_df['churned'] = churn_df['churned'].map(lambda x: x.strip())

plt1 = sns.distplot(churn_df['total_intl_charge'], kde = False)

corr = churn_df[["account_length", "number_vmail_messages", "total_day_minutes",
                    "total_day_calls", "total_day_charge", "total_eve_minutes",
                    "total_eve_calls", "total_eve_charge", "total_night_minutes",
                    "total_night_calls", "total_intl_minutes", "total_intl_calls",
                    "total_intl_charge"]].corr()

sns.heatmap(corr)

#Show the Plot
# plt.show()

from scipy.stats import pearsonr
print("The pearson coefficient is: ", pearsonr(churn_df['total_day_minutes'], churn_df['number_vmail_messages'])[0])

#Feature Preprocessing:

#1. Get groundtruth:
y = np.where(churn_df['churned'] == 'True.', 1, 0)

#2. Drop useless columns:
to_drop = ['state', 'area_code', 'phone_number', 'churned']
churn_new_df = churn_df.drop(to_drop, axis = 1)

#3.Convert Yes/No to Boolean values:
yes_no = ["intl_plan", "voice_mail_plan"]
churn_new_df[yes_no] = churn_df[yes_no] == 'yes'
x = churn_new_df

#Scale the data:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
print("Feature Space holds %d observations and %d different features" % x.shape)
print("unique target labels:", np.unique(y))

#Model Training:
from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2)
print('training data has %d observations with %d different features' % x_train.shape)
print('testing data has %d observations with %d different features' % x_test.shape)
print("\n")

#Model training and selection:
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

classifier_logistic = LogisticRegression()
classifier_KNN = KNeighborsClassifier()
classifier_RF = RandomForestClassifier()
classifier_SVC = SVC()

model_names = ['Logistic Regression', 'KNN', 'Random Forest','SVM']
model_list = [classifier_logistic, classifier_KNN, classifier_RF, classifier_SVC]
count  = 0
print("Accuracy of different Model Classifiers: ")
for classifier in model_list:
	cv_score = model_selection.cross_val_score(classifier, x_train, y_train, cv = 5)
	print('Model Accuracy of %s is: %.3f'%(model_names[count], cv_score.mean()))
	count += 1
print("\n")

#Use Grid Search to find optimal hyperparameters:
from sklearn.model_selection import GridSearchCV
def print_grid_search_metric(gs):
	print("Best Score: %0.3f" % gs.best_score_)
	print("Best parameters set:")
	best_para = gs.best_params_
	for para_name in sorted(parameters.keys()):
		print("\t%s: %r" % (para_name, best_para[para_name]))

#Find the best hyperparameters for Logistic Regression using Grid Search
parameters = {
	'penalty': ('l1', 'l2'),
	'C': (1, 5, 10)
}

Grid_LR = GridSearchCV(LogisticRegression(), parameters, cv =5)
Grid_LR.fit(x_train, y_train)
print(print_grid_search_metric(Grid_LR))
best_LR_model =  Grid_LR.best_estimator_
print(best_LR_model)
print("\n")

#Find Optimal Hyperparameters for KNN using Grid Search
parameters = {
	'n_neighbors' : [2,3,4,5,6,7,8,10]
}

Grid_KNN = GridSearchCV(KNeighborsClassifier(), parameters, cv = 5)
Grid_KNN.fit(x_train, y_train)
print(print_grid_search_metric(Grid_KNN))
best_KNN_model =  Grid_KNN.best_estimator_
print(best_KNN_model)
print("\n")

#Find Optimal Hyperparameters for Random Forest using Grid Search:
parameters = {
	'n_estimators' : [40, 60 ,80]
}

Grid_RF = GridSearchCV(RandomForestClassifier(), parameters, cv = 5)
Grid_RF.fit(x_train, y_train)
print(print_grid_search_metric(Grid_RF))
best_RF_model = Grid_RF.best_estimator_
print(best_RF_model)
print("\n")

#Model Evaluation - Confusion Matrix:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def cal_evaluation(classifier, cm):
	tn = cm[0][0]
	fp = cm[0][1]
	fn = cm[1][0]
	tp = cm[1][1]
	accuracy = (tp+tn) / (tp+fp+fn+tn+0.0)
	precision = tp / (tp+fp+0.0)
	recall = tp / (tp+fn+0.0)
	print(classifier,":")
	print("Accuracy is: %0.3f" %accuracy)
	print("Precision is: %0.3f" %precision)
	print("Recall is: %0.3f" %recall)
	print("\n")

def draw_confusion_matrices(confusion_matrices):
	class_names = ['Not', 'Churn']
	for cm in confusion_matrices:
		classifier, cm = cm[0], cm[1]
		cal_evaluation(classifier, cm)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm, interpolation = 'nearest', cmap = plt.get_cmap('Reds'))
		plt.title('Confusion matrix for %s' % classifier)
		fig.colorbar(cax)
		ax.set_xticklabels([''] + class_names)
		ax.set_yticklabels([''] + class_names)
		plt.xlabel('Predicted')
		plt.ylabel('True')
		plt.show()

confusion_matrices = [
("Random Forest", confusion_matrix(y_test, best_RF_model.predict(x_test))),
("Logistic Regression", confusion_matrix(y_test, best_LR_model.predict(x_test))),
]

draw_confusion_matrices(confusion_matrices)

#Draw ROC curve of the Random Forest Model:
from sklearn.metrics import roc_curve
from sklearn import metrics
y_pred_rf = best_RF_model.predict_proba(x_test)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

#Plot it:
plt.figure(1)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr_rf, tpr_rf, label = 'RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve - RF Model')
plt.legend(loc = 'best')
plt.show()

#Output RF AUC metrics:
print("The AUC of Random Forest Classifier is: ", metrics.auc(fpr_rf, tpr_rf))

#ROC of a LR Model:
y_pred_lr = best_LR_model.predict_proba(x_test)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)

#Plot it:
plt.figure(1)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr_lr, tpr_lr, label = 'LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve - LR Model')
plt.legend(loc = 'best')
plt.show()

#Output LR AUC Metrics:
print("The AUC of Logistic Regression Classifier is:", metrics.auc(fpr_lr, tpr_lr))
print("\n")

#Logistic Regression Feature Selection:
#L1 Model:
LRmodel_l1 = LogisticRegression(penalty = 'l1')
LRmodel_l1.fit(x,y)
LRmodel_l1.coef_[0]
print("Logistic Regression (L1) Coefficients")
for k, v in sorted(zip(map(lambda x: round(x,4), LRmodel_l1.coef_[0]),churn_new_df.columns), key = lambda k_v: (-abs(k_v[0]),k_v[1])):
	print(v + ":" + str(k))
print("\n")

#L2 Model:
LRmodel_l2 = LogisticRegression(penalty = 'l2')
LRmodel_l2.fit(x,y)
LRmodel_l2.coef_[0]
print("Logistic Regression (L2) Coefficients")
for k, v in sorted(zip(map(lambda x: round(x,4), LRmodel_l2.coef_[0]),churn_new_df.columns), key = lambda k_v: (-abs(k_v[0]),k_v[1])):
	print(v + ":" + str(k))
print("\n")

#Random Forest Model- Feature Importance:
classifier_RF.fit(x,y)
importances = classifier_RF.feature_importances_

#Print Feature Ranking:
print("Feature importance ranking by Random Forest Model:")
for k, v in sorted(zip(map(lambda x: round(x, 4), importances), churn_new_df.columns), reverse = True):
	print(v + ":" + str(k))