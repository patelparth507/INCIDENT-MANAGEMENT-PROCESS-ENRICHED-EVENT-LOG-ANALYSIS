import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt

IEL = pd.read_csv("C:/Users/admin/Desktop/Incident_event_log_Master_V4.csv")
IEL.head()
IEL.describe()
#IEL.dtypes
#IEL.columns
#IEL.dtypes

#dropping columns that is cleaned through excel and has a new updated version in dataset 
IEL.drop('caller_id',axis=1,inplace=True)
IEL.drop('opened_by',axis=1,inplace=True)
IEL.drop('sys_created_by',axis=1,inplace=True)
IEL.drop('sys_updated_by',axis=1,inplace=True)
IEL.drop('location',axis=1,inplace=True)
IEL.drop('category',axis=1,inplace=True)
IEL.drop('assigned_to',axis=1,inplace=True)
IEL.drop('subcategory',axis=1,inplace=True)
IEL.drop('closed_code',axis=1,inplace=True)
IEL.drop('resolved_by',axis=1,inplace=True)

#dropping u_symptom and subcategory_updated_v2 as of now 
IEL.drop('u_symptom',axis=1,inplace=True)
IEL.drop('subcategory_updated_v2',axis=1,inplace=True)
IEL.drop('assignment_group',axis=1,inplace=True)


# Removing all the duplicate incidents 
duplicates = IEL[IEL.duplicated(subset='number', keep='first')]
# After exploring duplicates it seems like reassignment_count,reopen_count,sys_mod_count are most up to date in the last record so keeping all the last occurance of the duplicate incidents
IEL = IEL[~IEL.duplicated(subset='number', keep='last')]


# We are left with 24918 unique incidents. Now dropping incident numbers (not required)
IEL.drop('number',axis=1,inplace=True)


# checking out distribution of each of the continuous variables
IEL.iloc[:,0:9].hist(bins=100, figsize=(15,20))
IEL.iloc[:,10:18].hist(bins=100, figsize=(15,20))
IEL.iloc[:,19:27].hist(bins=100, figsize=(15,20))
IEL.iloc[:,20:36].hist(bins=100, figsize=(15,20))
plt.show()

# Cleaning impact with proper labels
IEL.impact.value_counts()
IEL.impact.replace('1 - High',1,inplace=True)
IEL.impact.replace('2 - Medium',2,inplace=True)
IEL.impact.replace('3 - Low',3,inplace=True)
IEL.impact = IEL.impact.astype('category')

# Cleaning urgency with proper labels
IEL.urgency.value_counts()
IEL.urgency.replace('1 - High',1,inplace=True)
IEL.urgency.replace('2 - Medium',2,inplace=True)
IEL.urgency.replace('3 - Low',3,inplace=True)
IEL.urgency = IEL.urgency.astype('category')

# Cleaning priority with proper labels
IEL.priority.value_counts()
IEL.priority.replace('1 - Critical',1,inplace=True)
IEL.priority.replace('2 - High',2,inplace=True)
IEL.priority.replace('3 - Moderate',3,inplace=True)
IEL.priority.replace('4 - Low',4,inplace=True)
IEL.priority = IEL.priority.astype('category')

# only 701 incidents are taking between 61 and 341 days to get resolved so bucketing all this incidents into a single number 61
IEL["Days_Elapsed_for_close"] = np.where((IEL.Days_Elapsed_for_close>60), 61, IEL.Days_Elapsed_for_close)
IEL.Days_Elapsed_for_close.describe()
IEL['Days_Elapsed_for_close'].max()

# Mean encoding Caller_id as it contains huge number of categories/labels
x = IEL.groupby('Caller_id updated_v2').Days_Elapsed_for_close.mean()
x= pd.DataFrame(x)
x.columns
x = x.reset_index()

IEL = pd.merge(IEL, x, how='left', left_on='Caller_id updated_v2', right_on='Caller_id updated_v2')
IEL = IEL.drop('Caller_id updated_v2',axis=1)
IEL['Caller_id_mean_encoded'] = IEL.Days_Elapsed_for_close_y 
IEL = IEL.drop('Days_Elapsed_for_close_y',axis=1)
IEL.Caller_id_mean_encoded.hist(bins=100)

# Mean encoding opened_by as it contains huge number of categories/labels
x = IEL.groupby('opened_by updated_v2').Days_Elapsed_for_close_x.mean()
x= pd.DataFrame(x)
x.columns
x = x.reset_index()

IEL = pd.merge(IEL, x, how='left', left_on='opened_by updated_v2', right_on='opened_by updated_v2')
IEL = IEL.drop('opened_by updated_v2',axis=1)
IEL['opened_by_mean_encoded'] = IEL.Days_Elapsed_for_close_x_y 
IEL = IEL.drop('Days_Elapsed_for_close_x_y',axis=1)
IEL.opened_by_mean_encoded.hist(bins=100)

# Mean encoding location as it contains huge number of categories/labels
x = IEL.groupby('location_updated_v3').Days_Elapsed_for_close_x_x.mean()
x= pd.DataFrame(x)
x.columns
x = x.reset_index()

IEL = pd.merge(IEL, x, how='left', left_on='location_updated_v3', right_on='location_updated_v3')
IEL = IEL.drop('location_updated_v3',axis=1)
IEL['location_mean_encoded'] = IEL.Days_Elapsed_for_close_x_x_y 
IEL = IEL.drop('Days_Elapsed_for_close_x_x_y',axis=1)
IEL.location_mean_encoded.hist(bins=100)

# Mean encoding category as it contains huge number of categories/labels
x = IEL.groupby('category_updated_v2').Days_Elapsed_for_close_x_x_x.mean()
x= pd.DataFrame(x)
x.columns
x = x.reset_index()

IEL = pd.merge(IEL, x, how='left', left_on='category_updated_v2', right_on='category_updated_v2')
IEL = IEL.drop('category_updated_v2',axis=1)
IEL['category_mean_encoded'] = IEL.Days_Elapsed_for_close_x_x_x_y 
IEL = IEL.drop('Days_Elapsed_for_close_x_x_x_y',axis=1)
IEL.category_mean_encoded.hist(bins=100)

# Mean encoding closed_code as it contains huge number of categories/labels
x = IEL.groupby('closed_code_updated_v2').Days_Elapsed_for_close_x_x_x_x.mean()
x= pd.DataFrame(x)
x.columns
x = x.reset_index()

IEL = pd.merge(IEL, x, how='left', left_on='closed_code_updated_v2', right_on='closed_code_updated_v2')
IEL = IEL.drop('closed_code_updated_v2',axis=1)
IEL['closed_code_mean_encoded'] = IEL.Days_Elapsed_for_close_x_x_x_x_y 
IEL = IEL.drop('Days_Elapsed_for_close_x_x_x_x_y',axis=1)
IEL.closed_code_mean_encoded.hist(bins=100)

# Mean encoding resolved_by as it contains huge number of categories/labels
x = IEL.groupby('resolved_by_updated_v2').Days_Elapsed_for_close_x_x_x_x_x.mean()
x= pd.DataFrame(x)
x.columns
x = x.reset_index()

IEL = pd.merge(IEL, x, how='left', left_on='resolved_by_updated_v2', right_on='resolved_by_updated_v2')
IEL = IEL.drop('resolved_by_updated_v2',axis=1)
IEL['resolved_by_mean_encoded'] = IEL.Days_Elapsed_for_close_x_x_x_x_x_y 
IEL = IEL.drop('Days_Elapsed_for_close_x_x_x_x_x_y',axis=1)
IEL.resolved_by_mean_encoded.hist(bins=100)

IEL.columns

#Modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
#from scipy.stats import randint


X = np.array(IEL.drop('Days_Elapsed_for_close_x_x_x_x_x_x',axis=1))
y= np.array(IEL.Days_Elapsed_for_close_x_x_x_x_x_x)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=21, stratify=y)


# Model 1 - Linear Regression

reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rsquare = reg.score(x_test,y_test)

# Model 2 - Ridge

ridge = Ridge(alpha=0.1, normalize=True)    #normalize to set all variables on same scale
ridge.fit(x_train,y_train)
y_pred = ridge.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rsquare = ridge.score(x_test,y_test)

# Model 3 - Lasso

Lasso = Lasso(alpha=0.1, normalize=True)    #normalize to set all variables on same scale
Lasso.fit(x_train,y_train)
y_pred = Lasso.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rsquare = Lasso.score(x_test,y_test)

# Model 4 - SVM

svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rsquare = svc.score(x_test,y_test)

# Model 5 - Decision Tree

param_dist = {"max_depth": [3, None]}
                    #,"max_features": randint(1, 9),
                    #"min_samples_leaf": randint(1, 9),
                    #"criterion": ["gini", "entropy"]}
Tree = DecisionTreeRegressor()  
Tree_cv = GridSearchCV(Tree, param_dist, cv=5)
Tree_cv.fit(x_train,y_train)
y_pred = Tree_cv.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rsquare = Tree_cv.score(x_test,y_test)

# Model 6 - Neural Network

NN = MLPRegressor()
NN.fit(x_train,y_train)
y_pred = NN.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rsquare = Tree_cv.score(x_test,y_test)

#We select model 6 since it gives highest rquare and lowest rmse i.e we are getting best accuracy with neural networks 
on the hold out set.



