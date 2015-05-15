'''
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def readcsv_funct(input_csv):
	''' Takes input_csv file name as a string and returns DataFrame
	'''
	df = pd.DataFrame.from_csv(input_csv)
	return df

#This is just a comment to myself for the next assignment/dataset
#Do extra row convert/add to true false
#for not numeric values should return fraction so do df['non_numeric_colum'] = df.Graduated == "Yes", 
#df['non_numeric_colum'] = df.Gender == "Female"

def missing(dataframe):
	'''Finds number of missing entries for columns in df'''
	count_nan = len(df) - df.count()
	missing_data = count_nan
	return missing_data

def stats(dataframe):
	''' Generates data summaries and distributions from DataFrame input'''
	df = dataframe.copy()
	summary_stats = df.describe()
	mode = df.mode()
	summary_stats = summary_stats.append(mode)
	count_nan = pd.Series(len(df) - df.count(), name = "missing_vals").T
	summary_stats = summary_stats.append(count_nan).T
	new_stats = summary_stats.rename(columns = {'50%':'median', 0:'mode'})
	summary_stats=np.round(new_stats, decimals=2)
	summary_stats.to_csv("summary_stats") 
	return  summary_stats

def dist(dataframe, number_of_bins, name):
	df = dataframe
	ax_list = df.hist(bins=number_of_bins)
	#plt.savefig(name)
	return ax_list

def split_to_test_train(df):
	'''splits df into two df one for testing and one training'''
	is_test = np.random.uniform(0, 1, len(df)) > 0.75
	train = df[is_test==False]
	test = df[is_test==True]
	return test, train

def fill_mean(df, column_name):
	'''fills NaNs with mean of the column'''
	new_df = df
	new_df[new_df[column_name].isnull()] = new_df[new_df[column_name].isnull()].fillna(new_df[column_name].mean())
	return new_df 

def impute_cat(df, column_name):
	#This imputes values based on mean of a category (not very useful in this case but worth keeping!)
	for v in df[column_name].unique():
		p = df[column_name] == v 
		df.ix[p] = df.ix[p].fillna(df[p].mean())
	return df

def cap_values(x, cap):
    if x > cap:
        return cap
    else:
        return x

def cont_var_to_disc(df, column_name, max_value, number_of_bins):
	'''function that can discretize a continuous variable'''
	df[column_name] = df[column_name].apply(lambda x: cap_values(x, max_value))
	variable_name = column_name + "_bins"
	df[variable_name] = pd.cut(df[column_name], bins=number_of_bins, labels=False)
	print pd.value_counts(df[variable_name])
	#This is useful if you take all of the features to do the model but not if specify features
	#df.drop(column_name, axis=1, inplace=True)
	return df

def cat_var_to_binary(df, column_name):
	''' function that can take a categorical variable and create binary variables from it. '''
	dummies = pd.get_dummies(df[column_name], prefix=column_name)
	df.join(dummies.ix[:,:])
	#This is useful if you take all of the features for model but not if specify features
	#df.drop(column_name, axis=1, inplace=True) 
	return df 

def run_cv(X,y,clf,number):#clf_class, **kwargs):
	from sklearn.cross_validation import KFold
	from time import time
	kf = KFold(len(y),n_folds=number,shuffle=True)
	y_pred = y.copy()
	testing_time = []
	training_time = []
	for train_index, test_index in kf:
		#print train_index, test_index
		X_train, X_test = X[train_index], X[test_index]
		y_train = y[train_index]
		#clf = clf_class(**kwargs)
		toc = time()
		model = clf.fit(X_train,y_train)
		tic = time()
		train_time = tic - toc
		toc2 = time()
		y_pred[test_index] = clf.predict(X_test)
		tic2 = time()
		test_time = tic2-toc2
		training_time.append(train_time)
		testing_time.append(test_time)
	training_time = sum(training_time)/float(len(training_time))
	testing_time = sum(testing_time)/float(len(testing_time))
	#can do with different initializations  random_state = integer value and iterate i from 0 to 9)
	return y_pred, training_time, testing_time

def classifier(features, target, classifier):
	from time import time
	toc = time()
	train_cols = features.copy()
	model = classifier.fit(train_cols, target)
	tic = time()
	train_time = tic-toc
	return model, train_time

def predict_func(model, test_cols):
	from time import time
	toc = time()
	preds = model.predict(test_cols)
	tic = time()
	test_time = tic-toc
	return preds, test_time 

def evaluate(preds, true_values, train_time, test_time):
	from sklearn.metrics import classification_report, confusion_matrix
	from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
	confusion = confusion_matrix(true_values, preds)
	report = classification_report(true_values, preds, labels=[0, 1])
	accuracy = accuracy_score(true_values, preds)
	precision = precision_score(true_values, preds)
	recall = recall_score(true_values, preds)
	f1 = f1_score(true_values, preds, labels=[0, 1])
	auc_score = roc_auc_score(true_values, preds)
	metrics_list = [accuracy, precision, recall, f1, auc_score, train_time, test_time]
	columns = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'train_time', 'test_time']
	metrics_dict = dict(zip(columns,metrics_list))
	return confusion, report, metrics_dict
	
def metrics_dataframe(metrics_df,dict_metrics, model_name):	
	metrics_df.loc[model_name] = pd.Series(dict_metrics)
	return metrics_df

def prec_recall_curve(preds, true_values):
	from sklearn.metrics import precision_recall_curve
	precision, recall, thresholds = precision_recall_curve(preds, true_values)
	plt.figure()
	plt.ylabel("precision")
	plt.xlabel("recall")
	p_r = plt.plot(recall,precision)
	return p_r


#THIS PART OF THE SCRIPT RUNS THE CODE/ANALYSIS
input_file = "cs-training.csv"
df = readcsv_funct(input_file)

#print summary_stats and makes histograms
df["DebtRatio"] = df["DebtRatio"].apply(lambda x: cap_values(x, 5.0))
summary_stat= stats(df)
#print "stats", summary_stat
first_graph = df[["DebtRatio", "NumberOfDependents", "age"]]
second_graph = df[["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]]
third_graph = df[["NumberRealEstateLoansOrLines", "NumberOfOpenCreditLinesAndLoans"]]
bin_no = 40
first = dist(first_graph, bin_no, "dist_1.png")
first[0][0].set_xlim((0,6))
first[0][1].set_xlim((0, 8))
plt.savefig("dist_1.png")

bin_no = np.linspace(0,4, 20)
second = dist(second_graph, bin_no, "dist_2.png")
second[0][0].set_xlim((0, 4))
second[0][1].set_xlim((0, 4))
second[1][0].set_xlim((0, 4))
plt.savefig("dist_2.png")

bin_no = 50
#third_graph[["NumberOfOpenCreditLinesAndLoans"]].plot(kind = 'bar') 
third = dist(third_graph, bin_no, "dist_3.png")
third[0][0].set_xlim((0, 40))
third[0][1].set_xlim((0, 8))
plt.savefig("dist_3.png")

monthlyincome = df[["MonthlyIncome"]]
income_bins = np.linspace(0,50000,30)
month = dist(monthlyincome, income_bins, "monthly_income.png")
plt.savefig("Monthly_Income.png")


#SHOULD USE MEAN OF TRAING FOR BOTH TRAING AND TESTING!!!
#adds missing data 
#df["MonthlyIncome"] = df["MonthlyIncome"].apply(np.log1p)
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)
filled_dependents = df
column_name = "MonthlyIncome"
filled_df = (fill_mean(filled_dependents,column_name)).reset_index()
#print missing(filled_df)

#creates features
max_value = 100000
bin_number = 10
name_column = 'MonthlyIncome'
cont_var_to_disc(filled_df, name_column , max_value, bin_number)
name_column = "age"
max_value = 110
bin_number = 10
cont_var_to_disc(filled_df, name_column, max_value, bin_number )
#print filled_df


########### TRAINING AND TESTING WITH JUST TWO FOLDS --> IGNORE AS USING CV ###########

from sklearn.preprocessing import StandardScaler
features_list = ["MonthlyIncome_bins", "age_bins", "NumberOfTimes90DaysLate", "DebtRatio", "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]
target = filled_df["SeriousDlqin2yrs"]
true_values = target
all_features = filled_df[["MonthlyIncome_bins", "age_bins", "DebtRatio", "NumberOfDependents", "MonthlyIncome","age","NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate","NumberRealEstateLoansOrLines", "NumberOfOpenCreditLinesAndLoans"]]
scaler = StandardScaler()
X = filled_df[features_list].as_matrix()
features = scaler.fit_transform(X)
all_features = scaler.fit_transform(all_features.as_matrix())


from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC #linearsvc try
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT 
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.ensemble import GradientBoostingClassifier as Boost

###############list of models for parameter testing
'''
neighb_1 = KNN(n_neighbors=1,weights='uniform')
neighb_3 = KNN(n_neighbors=3,weights='uniform')
neighb_5 = KNN(n_neighbors=5,weights='uniform')
neighb_9 = KNN(n_neighbors=9,weights='uniform')
neighb_11 = KNN(n_neighbors=11,weights='uniform')
neighb_15 = KNN(n_neighbors=15,weights='uniform')
neighb_5_dist = KNN(n_neighbors=5,weights='distance')
neighb_15_dist = KNN(n_neighbors=15,weights='distance')

rf_n20_d5 = RF(n_estimators=20,criterion='gini', max_depth=5)
rf_n100_d5 = RF(n_estimators=100,criterion='gini', max_depth=5)
rf_n5_d5 = RF(n_estimators=5,criterion='gini', max_depth=5)
rf_n20_d10 = RF(n_estimators=20,criterion='gini', max_depth=10)
rf_n50_d10 = RF(n_estimators=50,criterion='gini', max_depth=10)
rf_n100_d15 = RF(n_estimators=100,criterion='gini', max_depth=15)
rf_ent_n20_d5 = RF(n_estimators=20,criterion='entropy', max_depth=5)
rf_ent_n100_d10 = RF(n_estimators=100,criterion='entropy', max_depth=10)

bagging_lr_10 = BC(base_estimator=LR(fit_intercept=False),n_estimators=10)
bagging_lr_20 = BC(base_estimator=LR(fit_intercept=False),n_estimators=20)
bagging_10 = BC(base_estimator=None,n_estimators=10)
bagging_20 = BC(base_estimator=None,n_estimators=20)

boosting_exp_r01_e100_d3 = Boost(loss='exponential',learning_rate=0.1,n_estimators=100, max_depth=3)
boosting_exp_r015_e100_d3 = Boost(loss='exponential',learning_rate=0.15,n_estimators=100, max_depth=3)
boosting_exp_r01_e100_d10 = Boost(loss='exponential',learning_rate=0.1,n_estimators=100, max_depth=10)
boosting_exp_r01_e50_d10 = Boost(loss='exponential',learning_rate=0.1,n_estimators=50, max_depth=10)
boosting_dev_r01_e100_d3 = Boost(loss='deviance',learning_rate=0.1,n_estimators=100, max_depth=3)
boosting_dev_r015_e100_d10 = Boost(loss='deviance',learning_rate=0.15,n_estimators=100, max_depth=10)

#model_params = [neighb_1, neighb_3, neighb_5, neighb_9, neighb_11, neighb_15, neighb_5_dist, neighb_15_dist]
#list_of_models = ["neighb_1", "neighb_3", "neighb_5", "neighb_9", "neighb_11", "neighb_15", "neighb_5_dist", "neighb_15_dist"]
#model_params = [rf_n20_d5, rf_n100_d5,rf_n5_d5, rf_n20_d10, rf_n50_d10, rf_n100_d15, rf_ent_n20_d5, rf_ent_n100_d10]
#list_of_models = ["rf_n20_d5", "rf_n100_d5","rf_n5_d5", "rf_n20_d10", "rf_n50_d10", "rf_n100_d15", "rf_ent_n20_d5", "rf_ent_n100_d10"]
#model_params = [bagging_lr_10, bagging_lr_20, bagging_10, bagging_20]
#list_of_models = ["bagging_lr_10", "bagging_lr_20", "bagging_10", "bagging_20"]
#model_params = [boosting_exp_r01_e100_d3, boosting_exp_r015_e100_d3, boosting_exp_r01_e100_d10, boosting_exp_r01_e50_d10, boosting_dev_r01_e100_d3, boosting_dev_r015_e100_d10]
#list_of_models = ["boosting_exp_r01_e100_d3", "boosting_exp_r015_e100_d3", "boosting_exp_r01_e100_d10", "boosting_exp_r01_e50_d10", "boosting_dev_r01_e100_d3", "boosting_dev_r015_e100_d10"]
'''

#actual list of models
logit = LR(fit_intercept=False)
neighb = KNN(n_neighbors=15,weights='uniform')#'distance', experiment with n is odd
svm = SVC(C=1.0, kernel='linear')#kernel='rbf' or 'linear' or 'poly' C=1.0 is default
randomforest = RF(n_estimators=20,criterion='gini',max_depth=15) #n is 10 default criterion='gini' or 'entropy'
decisiontree = DT(criterion='gini')#can also be 'entropy'
bagging = BC(base_estimator=None,n_estimators=40)#pass in base estimator as logit maybe? Not trained tho! 
boostin = Boost(loss='deviance',learning_rate=0.15,n_estimators=100,max_depth=3)#loss='exponential', learning_rate=0.1 which is default
model_params = [logit, neighb, svm, randomforest, decisiontree, boostin, bagging] 
list_of_models = ["LR", "KNN", "SVM", "RF", "DT", "Boosting", "Bagging"]

metrics_df = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'train_time', 'test_time'], index = [list_of_models])

baseline_predictions = pd.Series(0,index=np.arange(len(true_values)))
confusion_matrix, report, dict_metrics = evaluate(baseline_predictions, true_values, 0, 0)
metrics_dataframe(metrics_df,dict_metrics, "baseline")

i = 0
for model in model_params: 
	if model == logit:
		predictions, training_time, testing_time = run_cv(features, target, model, 5)
		#model_out, training_time = classifier(features,target,model)
		#predictions, testing_time = predict_func(model_out, test_col)
		clf_name = list_of_models[0]
		confusion_matrix, report, dict_metrics = evaluate(predictions, true_values, training_time, testing_time)
		metrics_dataframe(metrics_df,dict_metrics, clf_name)
		graph = prec_recall_curve(predictions, true_values)
		name = "precision_recall_" + clf_name + ".png"
		plt.savefig(name)
		i += 1
	elif model == svm:
		predictions, training_time, testing_time = run_cv(all_features, target, model, 2)
		confusion_matrix, report, dict_metrics = evaluate(predictions, true_values, training_time, testing_time)
		clf_name = list_of_models[i]
		metrics_dataframe(metrics_df,dict_metrics, clf_name)
		graph = prec_recall_curve(predictions, true_values)
		name = "precision_recall_" + clf_name + ".png"
		plt.savefig(name)
		i += 1
	else:	
		predictions, training_time, testing_time = run_cv(all_features, target, model, 5)
		#model_out, training_time = classifier(all_features, target,model)
		#redictions, testing_time = predict_func(model_out, test_all)
		confusion_matrix, report, dict_metrics = evaluate(predictions, true_values, training_time, testing_time)
		clf_name = list_of_models[i]
		metrics_dataframe(metrics_df,dict_metrics, clf_name)
		graph = prec_recall_curve(predictions, true_values)
		name = "precision_recall_" + clf_name + ".png"
		plt.savefig(name)
		i += 1
	#print confusion_matrix, report
	print metrics_df
metrics_df.to_csv("metrics.csv")






