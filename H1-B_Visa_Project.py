#%%

import os
os.chdir('E:\ML Project\Project1')


# In[446]:

import pandas as pd
import numpy as np 
import warnings
import collections
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,LassoCV
from sklearn.metrics import r2_score , mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")
pd.pandas.set_option('display.max_columns', None)


# In[447]:


dataset = pd.read_csv('H-1BVisaApplications-2017.csv',na_values=['NA'],index_col=None)
print('Dimension: {}'.format(dataset.shape))
dataset.head()


# In[448]:


# Variables that contain missing values
dataset.isnull().sum()


# ## Selecting columns.
# -----

# In[449]:


dataset = dataset[['visa_class',
                   'case_status',
                   'employer_city',
                   'employer_state',
                   'employer_country',
                   'soc_name',
                   'soc_code',
                   'job_title',
                   'total_workers',
                   'employment_start_date',
                   'employment_end_date',
                   'full_time_position',
                   'pw_unit_of_pay',
                   'wage_rate_of_pay_from',
                   'wage_rate_of_pay_to',
                   'wage_unit_of_pay',
                   'worksite_city',
                   'worksite_county',
                   'new_employment',
                   'continued_employment',
                   'change_previous_employment',
                   'new_concurrent_employment',
                   'pw_wage_level',
                   'pw_source',
                   'pw_source_year',
                   'pw_source_other',
                   'prevailing_wage'
                  ]]


# In[450]:


dataset.head()


# In[451]:


dataset.describe()


# In[452]:


dataset.info()


# ## Missing Values.
# ----

# In[453]:


# make a list of the variables that contain missing values
dataset.isnull().sum()


# ---

# ### Analyzing Numerical variables (Discrete and continous).

# In[454]:


#numerical variables
dataset._get_numeric_data().columns


# In[455]:


#Checking Distribution on total_worker
dataset.total_workers.hist()
print(dataset.total_workers.describe())
#the distrubition is skewed and consist of only one value hence we'll not consider this column


# In[456]:


#Checking Distribution on wage rate of pay from
dataset.wage_rate_of_pay_from.plot()
print(dataset.wage_rate_of_pay_from.describe())
#There are variation in this column


# In[457]:


#Checking Distribution on wage_rate_of_pay_to
dataset.wage_rate_of_pay_to.plot()
print(dataset.wage_rate_of_pay_to.describe())
#the distrubition is skewed and consist of only one value hence we'll not consider this column


# In[458]:


#Checking Distribution on new_employment
dataset.new_employment.plot(kind='hist')
print(dataset.new_employment.describe())
#the distrubition is skewed and consist of only one value hence we'll not consider this column


# In[459]:


#Checking Distribution on continued_employment
dataset.continued_employment.plot(kind='hist')
print(dataset.continued_employment.describe())
#the distrubition is skewed and consist of only one value hence we'll not consider this column


# In[460]:


#Checking Distribution on change_previous_employment
dataset.change_previous_employment.plot(kind='hist')
print(dataset.change_previous_employment.describe())
#the distrubition is skewed and consist of only one value hence we'll not consider this column


# ### Categorical Variables.
# -----

# In[461]:


### Categorical variables
cat_vars = dataset.select_dtypes(include=['object']).columns
print(cat_vars)
print('Number of categorical variables: ', len(cat_vars))


# In[462]:


#checking for visa_class


#Majority of class belong to H1-B hence we'll remove this


# In[463]:


#checking for case_status
dataset.case_status.value_counts().plot(kind='bar')
print(dataset.case_status.describe())
#Majority of class belong to Certified but there are other variation hence we'll keep it for now


# In[464]:


#checking for employer_state
dataset.employer_state.value_counts().plot(kind='bar')
print(dataset.employer_state.describe())
#There are many unique values out here


# In[465]:


#checking for employer_country
dataset.employer_country.value_counts().plot(kind='bar')
print(dataset.employer_country.describe())
#Majority are united states hence we can't keep this in our model


# In[466]:


#Checking for soc_code
print(dataset.soc_code.describe())
#There are many unique value which requires grouping


# In[467]:


#Checking for job title full_time_position
print(dataset.job_title.describe())
dataset[dataset["pw_wage_level"]=='Level I']["job_title"].value_counts()[0:15].plot(kind='bar')
#There are many unique value which requires grouping
#Checking by job title doesn't give any information about the wage level,so going by wage rate , so checking with wage_rate_of_pay_from


# In[468]:


def WRPF(level):
    lst=dataset.wage_rate_of_pay_from[dataset["pw_wage_level"]=='Level I'].value_counts()
    X=pd.Series(lst.index[0:10]).apply(lambda x: str(x))
    Y=lst[lst.index[0:10]]
    plt.bar(X,Y)
    plt.show()


# In[469]:


WRPF('Level I')


# In[470]:


WRPF('Level II')


# In[471]:


WRPF('Level III')


# In[472]:


WRPF('Level IV')


# In[473]:


#we need to check the distribution of the levels in below 60 and 60-80 bucket
dataset[dataset['wage_rate_of_pay_from']<60000]["pw_wage_level"].value_counts().plot(kind='bar')


# In[474]:


dataset[dataset['wage_rate_of_pay_from'].between(60000,80000)]["pw_wage_level"].value_counts().plot(kind='bar')
#hence buketing down , wage_rate_of_pay_from <=80000 -> level I ,80000-100000 --> level II, 100000-120000 -->level III and above 120000 level IV**


# In[475]:


#Checking for pw_unit_of_pay
print(dataset.pw_unit_of_pay.describe())
dataset.pw_unit_of_pay.value_counts().plot(kind='bar')
#Majority of the row contain year hence well hour,month etc.. rows since they do dont contribute much


# In[476]:


#Checking for worksite_city
print(dataset.worksite_city.describe())
#Majority of contribution are skewed hence we'll not consider this feature


# In[477]:


#Checking for worksite_county
print(dataset.worksite_county.describe())
#Majority of contribution are skewed hence we'll not consider this feature


# In[478]:


#Checking for pw_wage_level
print(dataset.pw_wage_level.describe())
dataset.pw_wage_level.value_counts().plot(kind='bar')


# In[479]:


#Checking for pw_source
print(dataset.pw_source.describe())
dataset.pw_source.value_counts().plot(kind='bar')


# In[480]:


# checking for pw_source_other
print(dataset.pw_source_other.describe())
#There are many unique values hence it need to be grouped


# In[481]:


#Correlation 
cor = dataset.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
sns.heatmap(cor)


# ### Handling missing values, ouliers and selecting features.
# ---------

# In[482]:


# pw_wage_level depends on job_title and wage_rate_of_pay,so cannot impute mode
# need to check the distribution of the levels in below 60 and 60-80 bucket
# As the value of unit of pay doesn't depends on any other column imputing with mode.

dataset['wage_unit_of_pay'].fillna(dataset['wage_unit_of_pay'].mode(),inplace=True)
dataset['pw_unit_of_pay'].fillna(dataset['pw_unit_of_pay'].mode(),inplace=True)

#pw_source is independent field and values not coverd in pw_source are covered in pw_source_other , so no relation
#There are 46 missing in 'pw_source' and 6372 in 'pw_source_other'(0.01%),hence imputiing with mode
dataset['pw_source'].fillna(dataset['pw_source'].mode(),inplace=True)
dataset['pw_source_other'].fillna(dataset['pw_source_other'].mode(),inplace=True)

# employment_start_date & employment_end_date has some null values which can be dropped
dataset.dropna(subset=['employment_start_date'], inplace=True)
dataset.dropna(subset=['employment_end_date'], inplace=True)
dataset.dropna(subset=['employer_state'], inplace=True)
dataset.dropna(subset=['job_title'], inplace=True)
dataset.dropna(subset=['full_time_position'], inplace=True)
dataset.dropna(subset=['wage_rate_of_pay_to'], inplace=True)
dataset.dropna(subset=['wage_unit_of_pay'], inplace=True)
dataset.dropna(subset=['pw_source'], inplace=True)
dataset.dropna(subset=['pw_source_other'], inplace=True)
dataset.dropna(subset=['prevailing_wage'], inplace=True)
dataset.dropna(subset=['worksite_county'], inplace=True)


# In[483]:


# checking levels of categorical variables
dataset = dataset[dataset['soc_code'].notnull()]
#Replacing names which serve under code section
dataset['soc_code'].replace('SOFTWARE DEVELOPERS, APPLICATIONS','15-1132',inplace = True)
dataset['soc_code'].replace('COMPUTER SYSTEMS ANALYST','15-1121',inplace = True)
dataset['soc_code'].replace('ELECTRICAL ENGINEERS','17-2071',inplace = True)
dataset['soc_code'].replace('MECHANICAL ENGINEERS','17-2141',inplace = True)
dataset['soc_code'].replace('ENGINEERS, ALL OTHER','17-2199',inplace = True)
dataset['soc_code'].replace('COMPUTER OCCUPATIONS, ALL OTHER','15-1199',inplace = True)
dataset['soc_code'].replace('ACCONTANTS AND AUDITORS','13-2011',inplace = True)
dataset['soc_code'].replace('DATABASE ADMINISTRATORS','15-1141',inplace = True)
#Removing missing data
dataset = dataset.drop(dataset[dataset.soc_code == ''].index)


# In[484]:


#Full time position
dataset = dataset.drop(dataset[dataset.full_time_position == ''].index)

##dropping some soc_codes as they are not needed
dataset = dataset.drop(dataset[dataset.soc_code == '2019'].index)
dataset = dataset.drop(dataset[dataset.soc_code == '1999'].index)
dataset = dataset.drop(dataset[dataset.soc_code == '1981'].index)
dataset = dataset.drop(dataset[dataset.soc_code == '1971'].index)
dataset = dataset.drop(dataset[dataset.soc_code == '1961'].index)
dataset = dataset.drop(dataset[dataset.soc_code == '1951'].index)
dataset = dataset.drop(dataset[dataset.soc_code == '1941'].index)
dataset = dataset.drop(dataset[dataset.soc_code == '1939'].index)
dataset = dataset.drop(dataset[dataset.soc_code == '1933'].index)
dataset = dataset.drop(dataset[dataset.soc_code == '1932'].index)
dataset = dataset.drop(dataset[dataset.soc_code == '1931'].index)


#Checking and imputing missing values
dataset = dataset.drop(dataset[dataset.employer_state == 'FM'].index)
dataset = dataset.drop(dataset[dataset.employer_state == 'AS'].index)
dataset = dataset.drop(dataset[dataset.job_title == '124592'].index)
dataset = dataset.drop(dataset[dataset.job_title == '62379'].index)
dataset['worksite_county'].dropna(axis = 0,inplace = True)


# In[485]:


dataset = dataset[['case_status', 'employer_state', 'soc_code', 'job_title', 
                   'full_time_position', 'wage_rate_of_pay_from',
                   'wage_rate_of_pay_to', 'wage_unit_of_pay', 'pw_wage_level',
                   'pw_source', 'pw_source_other','prevailing_wage']]


# In[486]:


#pw source other
print(dataset['pw_source'].value_counts())
dataset['pw_source'].replace('CBA','Other',inplace = True)
dataset['pw_source'].replace('DBA','Other',inplace = True)
dataset['pw_source'].replace('SCA','Other',inplace = True)

#Worksite county
dataset.loc[((dataset["wage_rate_of_pay_from"] < 80000) & (dataset["pw_wage_level"].isnull())),"pw_wage_level"] = "Level I"
dataset.loc[((dataset["wage_rate_of_pay_from"] >= 80000) & (dataset["wage_rate_of_pay_from"] < 100000) & (dataset["pw_wage_level"].isnull())),"pw_wage_level"] = "Level II"
dataset.loc[((dataset["wage_rate_of_pay_from"] >= 100000) & (dataset["wage_rate_of_pay_from"] < 120000) & (dataset["pw_wage_level"].isnull())),"pw_wage_level"] = "Level III"
dataset.loc[((dataset["wage_rate_of_pay_from"] >= 120000) & (dataset["pw_wage_level"].isnull())),"pw_wage_level"] = "Level IV"

### pw_source_other
dataset['pw_source_other'].replace('OFLC (ONLINE DATA CENTER)','OFLC ONLINE DATA CENTER',inplace = True)
dataset['pw_source_other'].replace('OFLC DATA CENTER','OFLC ONLINE DATA CENTER',inplace = True)


# ## Handling Ouliers.
# -----

# In[487]:


def Outlier_handling(dataset,column,count,replace_with):
    List_count=collections.Counter(dataset[column])
    List_Collect=list()
    List_Collect=[key for key,value in List_count.items() if value < count]        
    dataset[column].replace(List_Collect,replace_with,inplace=True)


# In[488]:


Outlier_handling(dataset,'employer_state',9147,'Other')
print(dataset['employer_state'].value_counts())


# In[489]:


Outlier_handling(dataset,'pw_source_other',10085,'Other')
print(dataset['pw_source_other'].value_counts())


# In[490]:


Outlier_handling(dataset,'soc_code',2000,'Other')
print(dataset['soc_code'].value_counts())


# In[491]:


dataset = dataset.drop(dataset[dataset.soc_code == 'Nov-21'].index)
dataset = dataset.drop(dataset[dataset.soc_code == 'Nov-31'].index)
dataset['wage_unit_of_pay'].replace('Bi-Weekly','Week',inplace = True)
print(dataset['wage_unit_of_pay'].value_counts())


# In[493]:


#pw source other
print(dataset['pw_source'].value_counts())
dataset['pw_source'].replace('CBA','Other',inplace = True)
dataset['pw_source'].replace('DBA','Other',inplace = True)
dataset['pw_source'].replace('SCA','Other',inplace = True)

#Worksite county
dataset.loc[((dataset["wage_rate_of_pay_from"] < 80000) & (dataset["pw_wage_level"].isnull())),"pw_wage_level"] = "Level I"
dataset.loc[((dataset["wage_rate_of_pay_from"] >= 80000) & (dataset["wage_rate_of_pay_from"] < 100000) & (dataset["pw_wage_level"].isnull())),"pw_wage_level"] = "Level II"
dataset.loc[((dataset["wage_rate_of_pay_from"] >= 100000) & (dataset["wage_rate_of_pay_from"] < 120000) & (dataset["pw_wage_level"].isnull())),"pw_wage_level"] = "Level III"
dataset.loc[((dataset["wage_rate_of_pay_from"] >= 120000) & (dataset["pw_wage_level"].isnull())),"pw_wage_level"] = "Level IV"

### pw_source_other
dataset['pw_source_other'].replace('OFLC (ONLINE DATA CENTER)','OFLC ONLINE DATA CENTER',inplace = True)
dataset['pw_source_other'].replace('OFLC DATA CENTER','OFLC ONLINE DATA CENTER',inplace = True)

#taking backup

df = dataset.copy()

dataset = dataset[dataset.wage_rate_of_pay_from!=0] #removing 0 value
#dataset.wage_rate_of_pay_to[dataset.wage_rate_of_pay_to==0] = 0.4 * dataset.wage_rate_of_pay_from[dataset.wage_rate_of_pay_to==0] 
dataset = dataset[dataset.prevailing_wage > 5000]
dataset = dataset[dataset.wage_unit_of_pay =='Year']
dataset = dataset[dataset.prevailing_wage<400000] #Removing outliers


# **After the several iteration over selecting features, checking co-relation and going back and forth while cleaning the data, we came about selecting few features which can good for our model**

# In[494]:


dataset = dataset[['case_status', 'employer_state', 'soc_code',
       'full_time_position', 'wage_rate_of_pay_from',
       'pw_wage_level','pw_source_other', 'prevailing_wage']]


cor = dataset.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
sns.heatmap(cor,annot=True)


# ### Model Building.
# ------

# In[495]:


#Getting dummies from categorical variables
dataset = pd.get_dummies(dataset,columns=['case_status','employer_state',
                                'soc_code','full_time_position',
                                'pw_wage_level','pw_source_other'],drop_first=True)

dataset = dataset.reset_index(drop=True)    


# In[496]:


X = dataset.loc[:, dataset.columns != 'prevailing_wage']
y = dataset['prevailing_wage']
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)


# In[497]:


#Linear regression
lm = LinearRegression(normalize=True)
lm.fit(X_train,y_train)
print('R-Sqaure Score:',lm.score(X_test,y_test))


# In[498]:


#Predicting values
y_train_pred = lm.predict(X_train)
y_test_pred = lm.predict(X_test)

#Checking scores
print("Train MAE:",np.round(mean_absolute_error(y_train, y_train_pred)))
print('Train RMSE: {}'.format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
print("Train MAPE:",np.round(np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100),"%")
print()
print("Test MAE:", np.round(mean_absolute_error(y_test, y_test_pred)))
print('Test RMSE: {}'.format(np.sqrt(mean_squared_error(y_test, y_test_pred))))
print("Test MAPE:",np.round(np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100),"%")
print()
print('Average prevailing wage: ', y_train.median())


# ### Regularised linear regression
# -----

# In[499]:


# train the model
lin_model = Lasso(alpha=0.005, random_state=45) # remember to set the random_state / seed
lin_model.fit(X_train, y_train)
print('R-Sqaure Score:',lin_model.score(X_test,y_test))


# In[500]:


pred = lin_model.predict(X_train)

print("Train MAE:",np.round(mean_absolute_error(y_train, pred)))
print('Train RMSE: {}'.format(np.sqrt(mean_squared_error(y_train, pred))))
print("Train MAPE:",np.round(np.mean(np.abs((y_train - pred) / y_train)) * 100),"%")

pred = lin_model.predict(X_test)
print()
print("Test MAE:",np.round(mean_absolute_error(y_test, pred)))
print('Test RMSE: {}'.format(np.sqrt(mean_squared_error(y_test, pred))))
print("Test MAPE:",np.round(np.mean(np.abs((y_test - pred) / y_test)) * 100),"%")
print()
print('Average prevailing wage: ', y_train.median())

