import pandas as pd

# preprocessing of data
def binary_transform(df, cols):
	'''
	Transform True/False to 1/0.
	'''
	df[cols] = df[cols].applymap(lambda x: 1 if x else 0)
	return df

def cat_var_to_binary_helper(df, column_name):
	''' function that can take a categorical variable and create binary variables from it. '''
	dummies = pd.get_dummies(df[column_name], prefix=column_name)
	df = df.join(dummies.ix[:,:])
	df.drop(column_name, axis=1, inplace=True) 
	return df

def cat_var_to_binary(df, cols):
	""" apply cat_var_to_binary_helper() to a list of cols"""
	for col in cols:
		df = cat_var_to_binary_helper(df, col)
	return df

def missing_indicator(df, column_name):
	nul = df[[column_name]].isnull()
	nul = nul.applymap(lambda x: 1 if x else 0)
	name = column_name + "_missing"
	df[name] = nul
	return df
