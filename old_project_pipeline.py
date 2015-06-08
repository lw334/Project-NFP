''' MAIN PROJECT PIPELINE TO PUT ALL FUNCTIONS IN FOR NFP PROJECT
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re

def readcsv_funct(input_csv):
	''' Takes input_csv file name as a string and returns DataFrame
	'''
	df = pd.DataFrame.from_csv(input_csv)
	return df

def missing(dataframe):
	'''Finds number of missing entries for columns in df'''
	missing_data = dataframe.isnull().sum()
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
	summary_stats = new_stats
	#summary_stats=np.round(new_stats, decimals=2)
	summary_stats.to_csv("summary_stats.csv") 
	return  summary_stats

def dist(dataframe, number_of_bins, name):
	df = dataframe
	ax_list = df.hist(bins=number_of_bins)
	plt.savefig(name)
	return ax_list

def bar_chart(dataframe,col_title):
	df = dataframe
	dataframe_cols = df[col_title]
	bar = pd.value_counts(dataframe_cols).plot(kind='bar',title=col_title)
	name = col_title + ".png"
	plt.savefig(name)
	plt.rcParams.update({'font.size': 12})
	return bar

	# data preprocessing
def binary_transform(df, cols):
	''' Transform True/False to 1/0'''
	df[cols] = df[cols].applymap(lambda x: 1 if x else 0)
	return df

def cat_var_to_binary_helper(df, column_name):
	''' function that can take a categorical variable and create binary variables from it. '''
	dummies = pd.get_dummies(df[column_name], prefix=column_name)
	df = df.join(dummies.ix[:,:])
	df.drop(column_name, axis=1, inplace=True) 
	return df

def cat_var_to_binary(df, cols):
	''' apply cat_var_to_binary_helper() to a list of columns'''
	for col in cols:
		df = cat_var_to_binary_helper(df, col)
	return df

def cat_to_bi_test_helper(df, col_name, train_cols):
	for i in range(len(train_cols)): 
		df[col_name+"_"+str(train_cols[i])] = (df[col_name] == train_cols[i]).astype(int)
	df.drop(col_name, axis=1, inplace=True) 
	return df

def cat_to_bi_test(df, cols, df_train):
	for col in cols:
		df = cat_to_bi_test_helper(df, col, pd.get_dummies(df_train[col]).columns)
	return df

def missing_indicator(df, column_name):
	""" add a missing indicator for a feature to the dataframe, 1 if missing and 0 otherwise. """
	nul = df[[column_name]].isnull()
	nul = nul.applymap(lambda x: 1 if x else 0)
	name = column_name + "_missing"
	df[name] = nul
	return df

def run_missing_indicator(df, cols):
	for col in cols:
		df = missing_indicator(df,col)
	return df

def fill_nans(df, column_name, value):
	'''fill NaNs with value'''
	new_df = df.copy()
	new_df[column_name] = new_df[column_name].fillna(value)
	return new_df 

def fill_str(df, col, value):
	''' filling in arbitrary value '''
	df[col] = df[col].fillna(value)
	return df

def fill_mode(df, column_name):
	'''fills NaNs with mode of the column for categorical and true/false'''
	new_df = fill_nans(df, column_name,df[column_name].mode())
	return new_df 

def fill_median(df, column_name):
	'''fills NaNs with median of the column for numerical'''
	new_df = fill_nans(df, column_name,df[column_name].median())
	return new_df 

def change_time_var(df, datelabel):
	'''fill in na before conversion'''
	for label in datelabel:
		df[label] = df[label].astype('datetime64[ns]')
	return df

def get_interval(df, startdate, enddate, labelinterval):
	'''convert into datetime objects before using the function'''
	df[labelinterval] = df[enddate] - df[startdate]
	df[labelinterval] = df[labelinterval].astype('int64')

def get_dummy_dates(df, labelinterval):
	df[labelinterval] = df[labelinterval].apply(lambda x: 1 if x < 0 else 0)

def get_year(df, datelabel):
	for date in datelabel:
		df[date + '_yr'] = df[date].dt.year

def get_month(df, datelabel):
	for date in datelabel:
		df[date + '_mth'] = df[date].dt.month

def applythreshold(array, threshold):
	return np.where(array >= threshold, 1, 0)

def train_test_split(df,column_name,last_train_yr):
	'''split function for main train and testing, according to last_train_yr'''
	train_df = df.loc[(df[column_name] <= last_train_yr)]
	test_df = df.loc[(df[column_name] > last_train_yr)]
	return train_df, test_df

def cv_split(df,column_name,last_train_yr, last_test_yr):
	'''split function for main train and testing, according to start_time and end_time'''
	train_df = df.loc[(df[column_name] <= last_train_yr)]
	test_df = df.loc[(df[column_name] > last_train_yr) & (df[column_name] <= last_test_yr)]
	return train_df, test_df

def sequential_cv_index(df_len, num_moves, total_pct, train_pct):
	'''return train-test split indexes for (num_moves) folds, overlapping allowed'''
	index = 0
	rv = []
	temp = range(df_len)
	train_size = int(np.round((df_len*total_pct)*train_pct))
	test_size = int(np.round(df_len*total_pct)-train_size)
	overlap = int(np.round((num_moves*(train_size+test_size)-df_len)/(num_moves-1)))
	step_size = train_size+test_size-overlap
	if(step_size<test_size):
		print "Warning: test on a single individual for multiple times!"
	for i in range(num_moves):
		if i != num_moves-1:
			train_index = temp[index:index+train_size]
			test_index = temp[index+train_size:index+train_size+test_size]
		else:
			train_index = temp[-(train_size+test_size):-test_size]
			test_index = temp[-test_size:]
		index += step_size
		rv.append((train_index, test_index))
	return rv

def run_cv(train_df, test_df, x_cols, y_col, clf_class, **kwargs):
	'''train and test the model'''
	from sklearn.preprocessing import StandardScaler
	from time import time
	# normalization
	X_train = np.array(train_df[x_cols].as_matrix().astype(np.float))
	X_test = np.array(test_df[x_cols].as_matrix().astype(np.float))
	# scaler = StandardScaler()
	# X_train = scaler.fit_transform(X_train)
	# X_test = scaler.fit_transform(X_test)
	y_train = np.ravel(train_df[y_col].astype(np.float))
	y_test = np.ravel(test_df[y_col].astype(np.float))
	clf = clf_class(**kwargs)
	# train and test
	begin_train = time()
	clf.fit(X_train, y_train)
	end_train = time()
	begin_test = time()
	y_pred = clf.predict(X_test)
	end_test = time()
	y_pred_proba = clf.predict_proba(X_test)
	train_time = end_train - begin_train
	test_time = end_test - begin_test
	return y_pred, y_pred_proba[:,1], y_test, train_time, test_time

def evaluate(name, y, y_pred, y_pred_prob, train_time, test_time, threshold):
	#LETS FIX THIS - PUT PRECISION RECALL INTO SEPARATE FUNCTION
	'''generate evaluation results'''
	from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
	rv = {}
	y_pred_new = applythreshold(y_pred_prob, threshold)
	rv["accuracy"] = str(np.mean(y == y_pred_new))
	rv["precision"] = str(precision_score(y, y_pred_new))
	rv["recall"] = str(recall_score(y, y_pred_new))
	rv["f1"] = str(f1_score(y, y_pred_new))
	rv["auc_roc"] = str(roc_auc_score(y, y_pred_new))
	#fpr, tpr, _ = roc_curve(y, y_pred_prob)
	# plot_eval_curve(fpr, tpr, name, "roc")
	#rv["auc_roc"] = str(auc(fpr, tpr))
	#precision_c, recall_c, _ = precision_recall_curve(y, y_pred_prob)
	# plot_eval_curve(recall_c, precision_c, name, "prc")
	#rv["auc_prc"] = str(auc(recall_c, precision_c))
	rv["train_time"] = str(train_time)
	rv["test_time"] = str(test_time)
	return pd.Series(rv), confusion_matrix(y, y_pred_new)

def precision_recall_curve(y_true, y_pred_prob):
	from sklearn.metrics import precision_recall_curve
	p, r, th = precision_recall_curve(y_true, y_pred_prob)
	plot(r, p)
	xlabel('Recall')
	ylabel('Precision')
	plt.show()
	return 

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

def select_parameter(train_df, test_df, classifier, x_cols, y_col, dic_param_vals, criterion, **kwargs):
	temp = []
	combs = value_combinations(dic_param_vals)
	metrics = pd.Series(["accuracy","precision","recall","f1","auc_roc"])
	results = pd.DataFrame(columns=metrics)
	for comb in combs:
		y_pred, y_pred_prob, y_test, time_train, time_test = run_cv(train_df, test_df, x_cols, y_col, classifier, **comb)
		evaluation_result, _ = evaluate("test", y_test, y_pred, y_pred_prob, time_train, time_test, 0.4)
		results.loc[str(classifier)+"_"+str(comb)] = evaluation_result
		temp.append((evaluation_result[criterion], comb, y_test, y_pred, y_pred_prob, time_train, time_test))
	temp.sort(reverse=True)
	return temp[0][1:]

def select_parameter_2(train_df, test_df, classifier, x_cols, y_col, dic_param_vals, list_threshold, criterion, **kwargs):
	temp = []
	classifier_name = reduce(lambda x,y: x+y, re.findall('[A-Z][^a-z]*', str(classifier).strip("'>")))
	combs = value_combinations(dic_param_vals)
	metrics = pd.Series(["accuracy","precision","recall","f1","auc_roc","train_time","test_time"])
	results = pd.DataFrame(columns=metrics)
	for comb in combs:
		y_pred, y_pred_prob, y_test, time_train, time_test = run_cv(train_df, test_df, x_cols, y_col, classifier, **comb)
		for threshold in list_threshold:
			name = classifier_name+"_"+str(comb)+"_"+str(threshold)
			evaluation_result, conf_matrix = evaluate(name, y_test, y_pred, y_pred_prob, time_train, time_test, threshold)
			print name
			print conf_matrix
			results.loc[name] = evaluation_result
	results.sort(columns=criterion,ascending=False)
	return results


if __name__ == '__main__':

	TIME = ["client_enrollment", "client_dob", "client_edd", "NURSE_0_FIRST_HOME_VISIT_DATE", "EarliestCourse",
	"EndDate","HireDate"] 

	missing_cols = ["CLIENT_ABUSE_AFRAID_0_PARTNER", "CLIENT_ABUSE_EMOTION_0_PHYSICAL_",
	"CLIENT_ABUSE_FORCED_0_SEX", "CLIENT_ABUSE_HIT_0_SLAP_LAST_TIM", "CLIENT_ABUSE_HIT_0_SLAP_PARTNER",
	"CLIENT_ABUSE_TIMES_0_ABUSE_WEAPO","CLIENT_ABUSE_TIMES_0_BURN_BRUISE",
	"CLIENT_ABUSE_TIMES_0_HEAD_PERM_I","CLIENT_ABUSE_TIMES_0_HURT_LAST_Y",
	"CLIENT_ABUSE_TIMES_0_PUNCH_KICK_", "CLIENT_ABUSE_TIMES_0_SLAP_PUSH_P",
	"CLIENT_WORKING_0_CURRENTLY_WORKI", "English", "INCOME", "PREPGBMI",
	"Spanish", "highest_educ","NURSE_0_YEAR_COMMHEALTH_EXPERIEN", "NURSE_0_YEAR_MATERNAL_EXPERIENCE",
	"NURSE_0_YEAR_NURSING_EXPERIENCE","nurse_English","nurse_hispanic",
	"nurse_Spanish","nurserace_americanindian_alaskanative","nurserace_asian","nurserace_black",
	"nurserace_nativehawaiian_pacificislander","nurserace_white","other_diseases"]

	NANCOLS_CAT_BINARY = ["MomsRE", "HSGED", "INCOME", "MARITAL", 
	"CLIENT_ABUSE_TIMES_0_HURT_LAST_Y", "CLIENT_ABUSE_TIMES_0_SLAP_PUSH_P",
	"CLIENT_ABUSE_TIMES_0_PUNCH_KICK_", "CLIENT_ABUSE_TIMES_0_BURN_BRUISE",
	"CLIENT_ABUSE_TIMES_0_HEAD_PERM_I", "CLIENT_ABUSE_TIMES_0_ABUSE_WEAPO",
	"CLIENT_BIO_DAD_0_CONTACT_WITH", "CLIENT_LIVING_0_WITH", "CLIENT_WORKING_0_CURRENTLY_WORKI",
	"CLIENT_ABUSE_HIT_0_SLAP_PARTNER","highest_educ", "educ_currently_enrolled_type",
	"Highest_Nursing_Degree","Highest_Non_Nursing_Degree","NurseRE","PrimRole","SecRole","CLIENT_ABUSE_EMOTION_0_PHYSICAL_", "CLIENT_ABUSE_EMOTION_0_PHYSICAL_",
	"CLIENT_ABUSE_FORCED_0_SEX", "CLIENT_ABUSE_HIT_0_SLAP_LAST_TIM", 
	"CLIENT_ABUSE_AFRAID_0_PARTNER", "educ_currently_enrolled",
	"English", "Spanish", "disease","heart_disease","high_blood_pressure","diabetes","kidney_disease",
	"epilepsy","sickle_cell_disease","chronic_gastrointestinal_disease",
	"asthma_chronic_pulmonary","chronic_urinary_tract_infection",
	"chronic_vaginal_infection_sti","genetic_disease_congenital_anomalies",
	"mental_health","other_diseases","nurse_English","nurse_hispanic",
	"nurse_Spanish","nurserace_americanindian_alaskanative","nurserace_asian","nurserace_black",
	"nurserace_nativehawaiian_pacificislander","nurserace_white","other_diseases"]

	NUMERICAL = ["PREPGKG", "PREPGBMI", "age_intake_years", 
	"edd_enrollment_interval_weeks", "gest_weeks_intake","NURSE_0_YEAR_COMMHEALTH_EXPERIEN", "NURSE_0_YEAR_MATERNAL_EXPERIENCE",
	"NURSE_0_YEAR_NURSING_EXPERIENCE"]

	CATEGORICAL = ["MomsRE", "HSGED", "INCOME", "MARITAL", 
	"CLIENT_ABUSE_TIMES_0_HURT_LAST_Y", "CLIENT_ABUSE_TIMES_0_SLAP_PUSH_P",
	"CLIENT_ABUSE_TIMES_0_PUNCH_KICK_", "CLIENT_ABUSE_TIMES_0_BURN_BRUISE",
	"CLIENT_ABUSE_TIMES_0_HEAD_PERM_I", "CLIENT_ABUSE_TIMES_0_ABUSE_WEAPO",
	"CLIENT_BIO_DAD_0_CONTACT_WITH", "CLIENT_LIVING_0_WITH", "CLIENT_WORKING_0_CURRENTLY_WORKI",
	"CLIENT_ABUSE_HIT_0_SLAP_PARTNER","highest_educ", "educ_currently_enrolled_type",
	"SERVICE_USE_0_OTHER1_DESC","SERVICE_USE_0_OTHER2_DESC",
	"SERVICE_USE_0_OTHER3_DESC","SERVICE_USE_0_TANF_CLIENT",
	"SERVICE_USE_0_FOODSTAMP_CLIENT","SERVICE_USE_0_SOCIAL_SECURITY_CL",
	"SERVICE_USE_0_UNEMPLOYMENT_CLIEN",
	"SERVICE_USE_0_IPV_CLIENT","SERVICE_USE_0_CPS_CHILD",
	"SERVICE_USE_0_MENTAL_CLIENT","SERVICE_USE_0_SMOKE_CLIENT",
	"SERVICE_USE_0_ALCOHOL_ABUSE_CLIE","SERVICE_USE_0_DRUG_ABUSE_CLIENT",
	"SERVICE_USE_0_MEDICAID_CLIENT","SERVICE_USE_0_MEDICAID_CHILD",
	"SERVICE_USE_0_SCHIP_CLIENT","SERVICE_USE_0_SCHIP_CHILD",
	"SERVICE_USE_0_SPECIAL_NEEDS_CHIL","SERVICE_USE_0_PCP_CLIENT","SERVICE_USE_0_PCP_WELL_CHILD",
	"SERVICE_USE_0_DEVELOPMENTAL_DISA","SERVICE_USE_0_WIC_CLIENT","SERVICE_USE_0_CHILD_CARE_CLIENT",
	"SERVICE_USE_0_JOB_TRAINING_CLIEN","SERVICE_USE_0_HOUSING_CLIENT",
	"SERVICE_USE_0_TRANSPORTATION_CLI","SERVICE_USE_0_PREVENT_INJURY_CLI",
	"SERVICE_USE_0_BIRTH_EDUC_CLASS_C","SERVICE_USE_0_LACTATION_CLIENT",
	"SERVICE_USE_0_GED_CLIENT","SERVICE_USE_0_HIGHER_EDUC_CLIENT",
	"SERVICE_USE_0_CHARITY_CLIENT","SERVICE_USE_0_LEGAL_CLIENT","SERVICE_USE_0_OTHER1",
	"SERVICE_USE_0_OTHER2","SERVICE_USE_0_OTHER3",
	"SERVICE_USE_0_PRIVATE_INSURANCE_","SERVICE_USE_0_PRIVATE_INSURANCE1",
	"Highest_Nursing_Degree","Highest_Non_Nursing_Degree","NurseRE",
	"PrimRole","SecRole"]

	BOOLEAN = ["nicu", "premature", "lbw", "CLIENT_ABUSE_EMOTION_0_PHYSICAL_",
	"CLIENT_ABUSE_FORCED_0_SEX", "CLIENT_ABUSE_HIT_0_SLAP_LAST_TIM", "CLIENT_ABUSE_AFRAID_0_PARTNER", "educ_currently_enrolled",
	"live_with_mother", "live_with_FoC_not_spouse", "live_with_spouse_not_FoC",
	"live_with_other_family_members", "live_with_infant_or_child","live_with_other_adults", "income_employment",
	"income_socialsecurity", "income_disability","income_other_public_benefits", "income_other", 
	"English", "Spanish","disease","heart_disease","high_blood_pressure","diabetes",
	"kidney_disease","epilepsy","sickle_cell_disease","chronic_gastrointestinal_disease","asthma_chronic_pulmonary",
	"chronic_urinary_tract_infection","chronic_vaginal_infection_sti","genetic_disease_congenital_anomalies",
	"mental_health","other_diseases","nurse_English","nurse_hispanic",
	"nurse_Spanish","nurserace_americanindian_alaskanative",
	"nurserace_asian", "nurserace_black","nurserace_nativehawaiian_pacificislander","nurserace_white"]

	#upload data
	input_file = "project_data10.csv"
	df_in = readcsv_funct(input_file)
	#drop rows where premature values are missing
	df = df_in.dropna(subset = ['premature'])
	# maybe delete this variable 
	df["edd_enrollment_interval_weeks"]=df["edd_enrollment_interval_weeks"].str.replace(',', '').astype(float)
	summary_stat= stats(df)
	#print "stats", summary_stat

	#saves distributions
	pd.value_counts(df.premature).plot(kind='bar')
	col_names = ["premature","MomsRE", "HSGED", "INCOME", "MARITAL","highest_educ", "educ_currently_enrolled_type"]
	for col in col_names:
		bar_chart(df,col)
	first_graph = df[NUMERICAL]
	bin_no = 40
	first = dist(first_graph, bin_no, "dist_1.png")
	plt.savefig("dist_1.png")
	#plt.show()

	# filling in missing dates with mode and get years and months
	df = fill_str(df, "client_enrollment", "2009-04-08 00:00:00")
	df = fill_str(df, "client_dob", "1990-08-04 00:00:00")
	df = fill_str(df, "client_edd", "2009-09-15 00:00:00")
	df = fill_str(df, "NURSE_0_FIRST_HOME_VISIT_DATE","2001-06-18 00:00:00")
	df = fill_str(df, "EarliestCourse", "2014-06-13 00:00:00")
	df = fill_str(df, "EndDate", "2010-12-06 00:00:00")
	df = fill_str(df, "HireDate", "2008-01-02 00:00:00")
	df = fill_str(df, "NURSE_0_BIRTH_YEAR", "1973")
	change_time_var(df,TIME)
	get_year(df, TIME)
	get_month(df, TIME)
	#generated
	get_interval(df, "client_edd", "EndDate", "leftbeforebirth")
	get_interval(df, "client_enrollment", "client_edd", "enrollment_duration")
	get_interval(df, "client_dob_yr", "client_enrollment_yr", "age")
	get_interval(df, "HireDate", "EndDate", "nurse_work_duration")
	get_dummy_dates(df,"leftbeforebirth")
	GENERATED = ["leftbeforebirth", "enrollment_duration", "age", "nurse_work_duration"]
	df.drop(TIME, axis=1, inplace=True)


	################################ if split by year #################################################
	#split data into training and test
	last_train_year = 2012 #so means test_df starts from 2010
	column_name = "client_enrollment_yr"
	train_df, test_df = train_test_split(df,column_name,last_train_year)

	#split train_df into the various train and testing splits for CV
	last_train_year = 2009
	last_test_year = 2012
	column_name = "client_enrollment_yr"
	cv_train, cv_test = cv_split(train_df,column_name,last_train_year, last_test_year)

	#impute 
	#make dummy indicators for columns with large numbers of missing values
	df_mind_train = run_missing_indicator(cv_train,missing_cols)
	df_mind_test = run_missing_indicator(cv_test,missing_cols)


	#NEED DATE COLUMNS OR OTHER RANDOM ONES 
	#fill_nans
	for col_name in NANCOLS_CAT_BINARY:
		df_mind_train= fill_mode(df_mind_train,col_name)
		df_mind_test= fill_mode(df_mind_test,col_name)

	for col_name in NUMERICAL:
		df_mind_train = fill_median(df_mind_train,col_name)
		df_mind_test = fill_median(df_mind_test,col_name)

	# Transforming features 
	df_train = cat_var_to_binary(df_mind_train,CATEGORICAL)
	df_test = cat_to_bi_test(df_mind_test,CATEGORICAL,df_mind_train)

	df_train = binary_transform(df_train, BOOLEAN)
	df_test = binary_transform(df_test, BOOLEAN)

	# df_train = pd.DataFrame.from_csv("train_1.csv")
	# df_test = pd.DataFrame.from_csv("test_1.csv")

	# Models
	# Set dependent and independent variables
	cols_to_drop = ["Nurse_ID", "NURSE_0_BIRTH_YEAR"]
	for col in cols_to_drop:
		df_train.drop(col, axis=1, inplace=True)
		df_test.drop(col, axis=1, inplace=True) 

	number_train = (missing(df_train)>0).sum()
	number_test = (missing(df_test)>0).sum()
	print "NUMBER OF COLS with missing values in df_train", number_train
	print "NUMBER OF COLS with missing values in df_test", number_test

	y_col = 'premature'
	x_cols = df_train.columns[4:]

	# Build classifier and yield predictions
	from sklearn.svm import LinearSVC as LSVC
	from sklearn.ensemble import RandomForestClassifier as RFC
	from sklearn.neighbors import KNeighborsClassifier as KNC
	from sklearn.tree import DecisionTreeClassifier as DTC
	from sklearn.linear_model import LogisticRegression as LR
	from sklearn.ensemble import BaggingClassifier as BC
	from sklearn.ensemble import GradientBoostingClassifier as GBC
	
	# logit = LR(fit_intercept=False)
	# neighb = KNC(n_neighbors=15,weights='uniform')#'distance', experiment with n is odd
	# svm = LSVC(C=1.0)#kernel='rbf' or 'linear' or 'poly' C=1.0 is default
	# randomforest = RFC(n_estimators=300,criterion='gini',max_depth=500) #n is 10 default criterion='gini' or 'entropy'
	# decisiontree = DTC(criterion='gini')#can also be 'entropy'
	# bagging = BC(base_estimator=None,n_estimators=40)#pass in base estimator as logit maybe? Not trained tho! 
	# boosting = GBC(loss='deviance',learning_rate=0.15,n_estimators=100,max_depth=3)#loss='exponential', learning_rate=0.1 which is default
	
	#classifiers = [logit, neighb, svm, randomforest, decisiontree, boostin, bagging] 
	classifiers = [LR]#, KNC, RFC, DTC, BC, GBC]
	
	import itertools as iter
	dic_param_vals = {
		LR:{"C":[0.01, 0.1, 1.0, 10.0]},
		KNC:{"n_neighbors":[5, 10, 15]},
		LSVC:{"C":[0.1, 1.0, 10.0]},
		RFC:{"n_estimators":[5, 10, 15], "max_features":["auto","log2"], "max_depth":[3, 6, None]},
		DTC:{"criterion":["gini","entropy"],"max_features":["auto","log2",None], "max_depth":[3, 6, None]},
		BC:{"n_estimators":[5, 10, 15], "max_samples":[0.5, 0.7, 1.0], "max_features":[0.5, 0.7, 1.0]},
		GBC:{"learning_rate":[0.05, 0.1, 0.3], "n_estimators":[100, 150, 200]}
	}
	
	list_threshold = np.arange(0.3,0.4,0.05)
	metrics = pd.Series(["accuracy","precision","recall","f1","auc_roc","train_time","test_time"])#"auc_prc"
	evaluation_result = pd.DataFrame(columns=metrics)
	for classifier in classifiers:	
		# y_pred, y_pred_prob, y_true, train_time, test_time = run_cv(df_train, df_test, x_cols, y_col, classifier)
		results = select_parameter_2(df_train, df_test, classifier, x_cols, y_col, dic_param_vals[classifier], list_threshold, "f1")
		evaluation_result = evaluation_result.append(results)
	baseline = str(1-df_test.describe()[y_col]["mean"])
	baseline_dict = dict(zip(metrics,pd.Series([baseline,0,0,0,0,0,0])))
	evaluation_result.loc["baseline"] = baseline_dict
	### OUTPUT EVALUATION TABLE
	print evaluation_result

	'''
	################################ if split the sorted dataframe evenly ##################################
	# Build classifier and yield predictions
	from sklearn.svm import LinearSVC as LSVC
	from sklearn.ensemble import RandomForestClassifier as RFC
	from sklearn.neighbors import KNeighborsClassifier as KNC
	from sklearn.tree import DecisionTreeClassifier as DTC
	from sklearn.linear_model import LogisticRegression as LR
	from sklearn.ensemble import BaggingClassifier as BC
	from sklearn.ensemble import GradientBoostingClassifier as GBC
	
	logit = LR(fit_intercept=False)
	neighb = KNC(n_neighbors=15,weights='uniform')#'distance', experiment with n is odd
	svm = LSVC(C=1.0, kernel='linear')#kernel='rbf' or 'linear' or 'poly' C=1.0 is default
	randomforest = RFC(n_estimators=20,criterion='gini',max_depth=15) #n is 10 default criterion='gini' or 'entropy'
	decisiontree = DTC(criterion='gini')#can also be 'entropy'
	bagging = BC(base_estimator=None,n_estimators=40)#pass in base estimator as logit maybe? Not trained tho! 
	boostin = GBC(loss='deviance',learning_rate=0.15,n_estimators=100,max_depth=3)#loss='exponential', learning_rate=0.1 which is default
	classifiers = [logit, neighb, svm, randomforest, decisiontree, boostin, bagging] 

	metrics = pd.Series(["accuracy","precision","recall","f1","auc_roc","train_time","test_time"])#"auc_prc"
	evaluation_result = pd.DataFrame(columns=metrics)

	# Sequential cross-validation
	split_index = sequential_cv_index(len(df), 5, 0.4, 0.8)
	
	for classifier in classifiers:
		name = reduce(lambda x,y: x+y, re.findall('[A-Z][^a-z]*', str(classifier).strip("'>")))
		id_num = 1
		for train_index, test_index in split_index:
			cv_train = df.iloc[train_index]
			cv_test = df.iloc[test_index]

			#impute 
			#make dummy indicators for columns with large numbers of missing values
			df_mind_train = run_missing_indicator(cv_train,missing_cols)
			df_mind_test = run_missing_indicator(cv_test,missing_cols)

			#fill_nans
			for col_name in NANCOLS_CAT_BINARY:
				df_mind_train= fill_mode(df_mind_train,col_name)
				df_mind_test= fill_mode(df_mind_test,col_name)

			for col_name in NUMERICAL:
				df_mind_train = fill_median(df_mind_train,col_name)
				df_mind_test = fill_median(df_mind_test, col_name)

			# Transforming features 
			df_train = cat_var_to_binary(df_mind_train,CATEGORICAL)
			df_test = cat_var_to_binary(df_mind_test,CATEGORICAL)

			df_train = binary_transform(df_train, BOOLEAN)
			df_test = binary_transform(df_test, BOOLEAN)

			number_train = (missing(df_train)>0).sum()
			number_test = (missing(df_test)>0).sum()
			print "NUMBER OF COLS with missing values in df_train", number_train
			print "NUMBER OF COLS with missing values in df_test", number_test
	
			# Model
			# Set dependent and independent variables / feature selection
			cols_to_drop = ["Nurse_ID", "NURSE_0_BIRTH_YEAR"]
			for col in cols_to_drop:
				df_train.drop(col, axis=1, inplace=True)
				df_test.drop(col, axis=1, inplace=True) 
			y_col = 'premature'
			x_cols = df_train.columns[3:100]

			y_pred, y_pred_prob, y_true, train_time, test_time = run_cv(df_train, df_test, x_cols, y_col, classifier)
			dic, conf_matrix = evaluate(name+str(id_num), y_true, y_pred, y_pred_prob, train_time, test_time)
			# print name
			# print conf_matrix
			evaluation_result.loc[name+str(id_num)] = dic
			id_num += 1
	baseline = str(1-df_test.describe()[y_col]["mean"])
	baseline_dict = dict(zip(metrics,pd.Series([baseline,0,0,0,0,0,0])))
	evaluation_result.loc["baseline"] = baseline_dict
	### OUTPUT EVALUATION TABLE
	# print evaluation_result
	'''
