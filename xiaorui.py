# Machine Learning Homework 4
# Xiaorui Tang, 449972

import pandas as pd
import numpy as np
import re
from scipy.stats import mode
import itertools as iter
from pandas import isnull, cut, Series, concat
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC as LSVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import ExtraTreesClassifier as ETC


def histogram_helper(key):
	plt.title("Probability Distribution for "+key)
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.savefig("output/summary/"+key+".png")

def discretize(df, var, num_groups):
	df[var+"_cate"]=cut(df[var], num_groups)
	df.drop(var, axis=1, inplace=True)

def create_dummy(df, var):
	for i in df[var].unique():
		df[var+"_"+str(i)]=df[var].apply(lambda x: 1 if x == i else 0)
	df.drop(var, axis=1, inplace=True)

def combination(l):
	comb = (iter.combinations(l, i) for i in range(len(l)-2, len(l)+1))
	return list(iter.chain.from_iterable(comb))

def value_combinations(dic):
	'''
	Takes a dictionary which maps from parameter name to a list of possible
	values, and returns a list of all possible dictionaries. Each dictionary 
	maps from parameter name to one particular value.
	'''
	rv = []
	key = []
	vals = []
	for item in dic.items():
		key.append(item[0])
		vals.append(item[1])
	combs = iter.product(*vals)
	for comb in combs:
		temp = {}
		for i in range(len(key)):
			temp[key[i]] = comb[i]
		rv.append(temp)
	return rv

def train_test_split(length):
    return KFold(length,n_folds=5,shuffle=True)

def fill_missing_median(df, train_index):
	return df.fillna(df.iloc[train_index].median())

def feature_importance(df, x_cols, y_col):
	model = ETC()
	model.fit(df[x_cols], df[y_col])
	return model.feature_importances_

def summary_statistics(data, cols):
	f = open("output.txt", "w")
	f.write("Output for Homework 4\nXiaorui Tang, 449972\n\nA. Summary Statistics & Histograms\n\nNote: For binary data SeriousDlqin2yrs, the mean shows the percentage distribution of Yes; (1-mean) is the percentage of No.\n\n")
	summary=data.describe().T
	summary.rename(columns={'50%': 'median'}, inplace=True)
	summary['mode']=Series(np.array([mode(data[var])[0][0] for var in cols]), index=summary.index)
	summary['cnt_missing']=len(data.index)-summary['count']
	to_drop = ['25%','75%','count']
	summary.drop(to_drop, axis=1, inplace=True)
	summary=summary.T
	for var in summary.index.values:
		if var not in ['mean','cnt_missing']:
			summary['SeriousDlqin2yrs'][var]="NA for categorical variable"
	f.write(str(summary)+'\n\n\nHistograms:\n\nSee output/summary directory.\nNote: To show the distributions more clearly, for RevolvingUtilizationOfUnsecuredLines, DebtRatio and MonthlyIncome, only present the data below 99th percentile.\n\n')
	f.close()

def plot_distribution(data, cols):
	for key in cols:
		plt.clf()
		if key in ['age','NumberOfOpenCreditLinesAndLoans']:
			data[key].hist(bins=10, range=(data[key].min(),data[key].max()))
		elif key in ['RevolvingUtilizationOfUnsecuredLines','DebtRatio','MonthlyIncome']:
			data[key].hist(bins=10, range=(data[key].min(),data[key].quantile(0.99)))
		else:
			data.groupby(key).size().plot(kind='bar')
		histogram_helper(key)

def model(X, y, train_index, test_index, clf_class, **kwargs):
	X_train, X_test = X[train_index], X[test_index]
	y_train = y[train_index]
	clf = clf_class(**kwargs)
	train_start = time()
	clf.fit(X_train,y_train)
	train_end = time()
	train_time = train_end - train_start
	test_start = time()
	y_pred = clf.predict(X_test)
	test_end = time()
	test_time = test_end - test_start
	y_pred_prob = clf.predict_proba(X_test)
	return y_pred, y_pred_prob, train_time, test_time

def cross_valid(data, classifier, x_cols, y_col, **kwargs):
	# Do train-test split for cross-validation
	size = len(data)
	kf = train_test_split(size)
	y_pred = np.zeros(size)
	y_pred_prob = np.zeros(size)
	y = data[y_col].as_matrix().astype(np.float)
	totaltime_train = 0
	totaltime_test = 0
	for train_index, test_index in kf:
		# Fill in missing values
		df = data.copy()
		df = fill_missing_median(df, train_index)
		# Transform and normalize
		X = df[x_cols].as_matrix().astype(np.float)
		scaler = StandardScaler()
		X = scaler.fit_transform(X)
		# Build classifier and yield predictions
		y_pred[test_index], y_pred_prob[test_index], train_time, test_time \
		= model(X, y, train_index, test_index, classifier, **kwargs)
		totaltime_train += train_time
		totaltime_test += test_time
	avgtime_train = train_time/len(kf)
	avgtime_test = test_time/len(kf)
	return y, y_pred, y_pred_prob, avgtime_train, avgtime_test

def plot_eval_curve(list1, list2, classifier, plot_type):
	plt.clf()
	plt.plot(list1, list2)
	if plot_type == "prc":
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Precision-Recall Curve for '+classifier)
	elif plot_type == "roc":
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC Curve for '+classifier)
	plt.savefig("output/evaluation/"+classifier+"_"+plot_type+".png")

def evaluate(name, y, y_pred, y_pred_prob, plot):
	rv = {}
	rv["accuracy"] = str(np.mean(y == y_pred))
	rv["precision"] = str(precision_score(y, y_pred))
	rv["recall"] = str(recall_score(y, y_pred))
	rv["f1"] = str(f1_score(y, y_pred))
	fpr, tpr, _ = roc_curve(y, y_pred_prob)
	if plot == True:
		plot_eval_curve(fpr, tpr, name, "roc")
	rv["auc_roc"] = str(auc(fpr, tpr))
	precision_c, recall_c, _ = precision_recall_curve(y, y_pred_prob)
	if plot == True:
		plot_eval_curve(recall_c, precision_c, name, "prc")
	rv["auc_prc"] = str(auc(recall_c, precision_c))
	return pd.Series(rv), confusion_matrix(y, y_pred)

def select_parameter(data, classifier, x_cols, y_col, dic_param_vals, criterion, **kwargs):
	temp = []
	combs = value_combinations(dic_param_vals)
	metrics = pd.Series(["accuracy","precision","recall","f1","auc_roc","auc_prc"])
	results = pd.DataFrame(columns=metrics)
	for comb in combs:
		y, y_pred, y_pred_prob, avgtime_train, avgtime_test = cross_valid(data, classifier, x_cols, y_col, **comb)
		evaluation_result, _ = evaluate("test", y, y_pred, y_pred_prob, False)
		results.loc[str(classifier)+"_"+str(comb)] = evaluation_result
		temp.append((evaluation_result[criterion], comb, y, y_pred, y_pred_prob, avgtime_train, avgtime_test))
	temp.sort(reverse=True)
	with open('param_select_outcomes.csv', 'a') as f:
		results.to_csv(f)
	return temp[0][1:]


# Put it all together
def run(data, classifiers, evaluation_result, x_cols, y_col):
	comb = -1
	f = open("output.txt", "a")
	for classifier in classifiers:
		if classifier == LR:
			dic_param_vals = {"C":[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0]}
		elif classifier == KNC:
			dic_param_vals = {"n_neighbors":[3, 5, 7, 10, 13, 15]}
		elif classifier == LSVC:
			dic_param_vals = {"C":[0.1, 1.0, 10.0]}
		elif classifier == RFC:
			dic_param_vals = {"n_estimators":[5, 10, 15], "max_features":["auto","log2"], "max_depth":[3, 6, None]}
		elif classifier == DTC:
			dic_param_vals = {"criterion":["gini","entropy"],"max_features":["auto","log2",None], "max_depth":[3, 6, None]}
		elif classifier == BC:
			dic_param_vals = {"n_estimators":[5, 10, 15], "max_samples":[0.5, 0.7, 1.0], "max_features":[0.5, 0.7, 1.0]}
		else:
			dic_param_vals = {"learning_rate":[0.05, 0.1, 0.3], "n_estimators":[100, 150, 200]}
		comb, y, y_pred, y_pred_prob, avgtime_train, avgtime_test = select_parameter(data, classifier, x_cols, y_col, dic_param_vals, "auc_prc")
		
		# Evaluate the classifier
		name = reduce(lambda x,y: x+y, re.findall('[A-Z][^a-z]*', str(classifier).strip("'>")))
		dic, conf_matrix = evaluate(name, y, y_pred, y_pred_prob, True)
		dic["train_time"] = str(avgtime_train)
		dic["test_time"] = str(avgtime_test)
		evaluation_result.loc[name] = dic
		f.write("Model: "+name+"\nParameters: ")
		if comb == -1:
			f.write("Default")
		else:
			f.write(str(comb))
		f.write("\nPerformance According to Evaluation Metrics: See evaluation result table\nConfusion Matrix:\n"+str(conf_matrix)+\
			"\nPrecision-Recall Curve & ROC Curve: See output/evaluation directory\n\n")
	f.close()


def evaluation_output(evaluation_result):
	f = open("output.txt", "a")
	f.write("Evaluation Result Table: \n\n"+str(evaluation_result)+"\n\n")
	# Compare classifiers
	max_min = concat([evaluation_result.idxmax(),evaluation_result.idxmin()],axis=1)
	max_min.columns = ["Max", "Min"]
	f.write("Best and worst performing classifier according to each metric: \n"+str(max_min)+"\n\n")
	f.close()


if __name__ == '__main__' :

	# Read data
	data=pd.read_csv("cs-training.csv", na_values=["NA"], sep=',', encoding='latin1', index_col='Unnamed: 0')

	# Explore data
	cols=list(data.columns.values)	
	summary_statistics(data, cols)
	plot_distribution(data, cols)

	# Generate features
	# Can generate features here; for this task, don't need to generate new features
	# discretize(df, "MonthlyIncome", 4)
	# create_dummy(df, "MonthlyIncome_cate")

	# Set dependent and independent variables
	y_col = 'SeriousDlqin2yrs'
	x_cols = data.columns[1:]

	# Feature selection
	# combs = combination(x_cols)
	# print feature_importance(data.fillna(data.median()), x_cols, y_col)

	# Loop through classifiers
	classifiers = [LR, KNC, LSVC, RFC, DTC, BC, GBC]
	f = open("output.txt", "a")
	baseline = str(1-data.describe()[y_col]["mean"])
	f.write("B. Models & Evaluations\n\nBase Line: "+baseline+"\n\nThreshold: 0.5\n\n")
	f.close()
	metrics = pd.Series(["accuracy","precision","recall","f1","auc_roc","auc_prc","train_time","test_time"])
	evaluation_result = pd.DataFrame(columns=metrics)
	# Can call run for multiple times here before reporting evaluation output
	run(data, classifiers, evaluation_result, x_cols, y_col)
	evaluation_output(evaluation_result)