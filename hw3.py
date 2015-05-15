
import csv
from time import time
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#preprocessing
from sklearn.preprocessing import StandardScaler
# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import BaggingClassifier as BG
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
#evaluation
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

features = ['RevolvingUtilizationOfUnsecuredLines','age',
'NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans',
'NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse',
'NumberOfDependents']

dependent_var = ['SeriousDlqin2yrs']


#logistic regression, k-nearest neighbor, decision trees, svm, random forrest, bagging and boosting
#classifiers = [LogisticRegression, KNN, DT, LinearSVC, RF, BG, AdaBoost]
classifiers = [DT]

# 1. Read Data
def read_data(fname):
	df = pd.DataFrame.from_csv(fname)
	return df

# 2. Explore Data
def sum_stats(df):
	moderow = df.mode().T 
	summary = df.describe().T
	summary["mode"] = moderow
	summary = summary.rename(columns={'50%':'median'}) 
	return summary

#find out numbers of missing values
def print_null_freq(df):
	df_long = pd.melt(df)
	nullval = df_long.value.isnull()
	return pd.crosstab(df_long.variable,nullval)

#plot histogram
def plot_data(df,col_name):
	plt.figure()
	data = list(df[col_name].dropna())
	plt.hist(data)
	plt.xlabel(col_name)
	plt.ylabel("Frequency")
	plt.savefig(col_name + "hist.png")

#plot categorical variables
def groupby_plot(df,col_name):
	plt.figure()
	ax = df.groupby(col_name).size().plot(kind = 'bar')
	plt.title(col_name)
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	gh = ax.get_figure()
	gh.savefig(col_name + "hist.png")

#plot a quantile of data
def plot_percentile(df, col_name, quantile_percent):
	plt.figure()
	plt.title(col_name + "using data within the" + quantile_percent + "quantile")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	column = df[col_name]
	q = column.quantile(quantile_percent)
	data = column[column < q]
	plt.hist(list(data))
	plt.savefig(col_name + "hist.png")

# 3. Pre-Process Data: Fill in misssing values (MonthlyIncome and NumberOfDependents)

# general function to fill in missing value with a specified value
def filling_missing_num(df, missing_colName, num):
	df[missing_colName] = df[missing_colName].fillna(num)
	return df

# general function to impute a missing column with reference columns
def impute(df,missing_col,ref_col):
	train = df[df[missing_col].isnull()==False]
	topredict = df[df[missing_col].isnull()]
	imputer = KNN(n_neighbors=1)
	imputer.fit(train[ref_col],train[missing_col])
	new_values = imputer.predict(topredict[ref_col])
	topredict[missing_col] = new_values
	fill = train.append(topredict)
	fill.sort_index(inplace = True)
	return fill, imputer

def impute_median(train_df, test_df, missing_col):
	median = train_df[missing_col].median()
	train_filled = filling_missing_num(train_df, missing_col, median)
	test_filled = filling_missing_num(test_df, missing_col, median)
	return train_filled, test_filled

# def impute_median(df, train_index, test_index):
# 	median = df.iloc[train_index].median()
# 	df.iloc[train_index] = df.iloc[train_index].fillna(median)
# 	df.iloc[test_index] = df.iloc[test_index].fillna(median)
# 	return df

# 4. Generate Features
def create_binary_helper(x, condition):
	if x >= condition:
		return 1
	else:
		return 0

def create_binary(df,colname,condition):
	df[str(colname +'_binary')] = df[colname].apply(lambda x: create_binary_helper(x,condition))

#discretize : specify the quantile range to cut off begin with 0 and end with 1: eg:[0,0.2,0.5,1]
def discritize_quantile(df,colname,quantile_range):
	bins = []
	for q in quantile_range:
		bins.append(df[colname].quantile(q))
	col_binned = pd.cut(df[colname], bins=bins)
	return col_binned

#determine the most important features using random forrest
def feature_importance(df):
	cols = list(df.columns.values)
	y = cols[0]
	features = np.array(cols[1:])
	clf = RF()
	clf.fit(df[features],df[y])
	importances = clf.feature_importances_
	sorted_idx = np.argsort(importances)
	return features[sorted_idx], importances[sorted_idx]


# 5. Build Classifier : training and testing with the training data

#return binary and probability results
def run_cv_binary(train_df, features, dependent_var, clf_class, **kwargs):
	X = np.array(train_df[features].as_matrix())
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	y = np.ravel(train_df[dependent_var])
	kf = KFold(len(y), n_folds = 5)
	y_pred = y.copy()
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train = y[train_index]
		clf = clf_class(**kwargs)
		begin_train = time()
		clf.fit(X_train, y_train)
		end_train = time()
		begin_test = time()
		y_pred = clf.predict(X_train)
		end_test = time()
		y_pred_proba = clf.predict_proba(X_train)
		train_time = begin_train - end_train
		test_time = begin_test - end_test
	return y_pred, y_pred_proba, clf, train_time, test_time

# 6. Evaluate Classifier

# baseline
def baseline(train_df, dependent_var):
	y = train_df[dependent_var]
	y_0 = float(len(np.where(y == 0)[0]))
	baseline = (y_0/float(len(y)))
	return baseline

# accuracy
def accuracy(y_true, y_pred):	
	score = accuracy_score(y_true,y_pred)
	return score

#precision
def precision(y_true, y_pred):
 	precision = precision_score(y_true, y_pred)
 	return precision

#recall
def recall(y_true, y_pred):
 	recall = recall_score(y_true, y_pred)
 	return recall
#F1
def f1(y_true, y_pred):
 	f1 = f1_score(y_true, y_pred)
 	return f1

#Area under curve
def ROCCurve(y_true, y_pred, clfname): 
	fpr, tpr, threshold = roc_curve(y_true, y_pred)
 	auc = roc_auc_score(y_true, y_pred)
 	plt.figure()
 	plt.plot(fpr, tpr)
 	plt.xlabel('False Positive Rate')
 	plt.ylabel('True Positive Rate')
 	plt.title('Receiver Operating Charateristics: AUC = {0:0.2f}'.format(auc))
 	plt.savefig(clfname + "png")
 	return auc

#Precision-recall curve
def PRCurve(y_true, y_pred_proba,clfname):
	precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
	average_precision = average_precision_score(y_true, y_pred_proba)
	plt.figure()
	plt.plot(recall, precision, label = 'precision_recall_curve')
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('Precision-Recall Curve : AUC = {0:0.2f}'.format(average_precision))
	plt.savefig(clfname + "png")
	return average_precision


if __name__ == "__main__":
	train = read_data("cs-training.csv")
	test = read_data("cs-test.csv")
	# preprocessing: 
	# filling in an arbitrary value assuming the missing value just means there is no dependent
	train_fill = filling_missing_num(train, "NumberOfDependents", 0).copy()
	test_fill = filling_missing_num(test, "NumberOfDependents", 0).copy()
	#fill in other missing values
	train_filled, test_filled = impute_median(train_fill,test_fill,"MonthlyIncome")
	# compare classifier
	clf_dict = {'accuracy':[],'precision':[],'recall':[],'f1':[],'auc':[],'average_precision':[], 'training time':[], 'testing time':[]}
	y_true = train_filled[dependent_var]
	for classifier in classifiers:
		name = str(classifier)
		y_pred, y_pred_proba, clf, traintime, testtime = run_cv_binary(train_filled, features, dependent_var, classifier)
		accuracy_ = accuracy(y_true, y_pred)
		precision_ = precision(y_true, y_pred)
		recall_ = recall(y_true, y_pred)
		f1_ = f1(y_true, y_pred)
		auc_ = ROCCurve(y_true, y_pred, name)
		average_precision_ = PRCurve(y_true, y_pred_proba, name)
		clf_dict['accuracy'].append(accuracy_)
		clf_dict['precision'].append(precision_)
		clf_dict['recall'].append(recall_)
		clf_dict['f1'].append(f1_)
		clf_dict['auc'].append(auc_)
		clf_dict['average_precision'].append(average_precision_)
		clf_dict['training time'].append(traintime)
		clf_dict['test_time'].append(testtime)
	df = pd.DataFrame(clf_dict, index = ['DT'])

	# df = pd.DataFrame(clf_dict, index = ['LogisticRegression', 'KnearestNeighbor', 
	# 	'Decision Tree', 'Linear SVM', 'Random Forrest', 'Bagging', 'Boosting'])
	df.to_csv('comparison_table.csv')
	# f = open('comparison_table.csv', 'w')
	# base = baseline(train_filled, dependent_var)
	# f.write('baseline: ' + str(base))
