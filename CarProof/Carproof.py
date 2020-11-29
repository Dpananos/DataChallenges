
# coding: utf-8

# # Carproof Data Assignment
# 
# ## Author:Demetri Pananos
# 
# ___
# 
# 
# ## Objective:
# 
# Regression of car prices on various features.
# 
# In this notebook, I perform some prelim analysis on the data.  I will also prepare training and test sets in order to do some regression.
# 
# 

# In[41]:


#import libraries

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats.mstats import winsorize
from scipy.optimize import brentq
from IPython.display import display

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict



import fancyimpute

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.rcParams['lines.color'] = 'k'
# mpl.rcParams['axes.labelcolor'] = 'white'
# mpl.rcParams['axes.edgecolor'] = 'white'
# mpl.rcParams['xtick.color'] = 'white'
# mpl.rcParams['ytick.color'] = 'white'

get_ipython().magic('matplotlib inline')


# # Exploratory Analysis
# ___
# 
# ### Preliminary Examination
# 
# Below, I create a table listing various summary stats about the data.
# 
# We see a few things:
# 
# * AccidentDate and AccidentDetail are extremely sparse.  I am sure this is because many cars do not have a police reported accident.  If we assume that HasPoliceReportedAccident is the compelte history of accidents for the vehicle, we can make inferences about the other two columns
# 
# * Many correlated columns.  For instance, VehicleMarketClassDisplay and VehicelMarketClassId.
# 
# * VinValue is unique to the vehicle, so we can drop that.
# 
# * Vehicle Trim has high cardinality, but is required for the regression.  Need to think about that.
# 
# * TotalClaims is sparse.  May have to drop it due to lack of information.

# In[56]:


df = pd.read_excel('/Users/demetri/Desktop/data.xlsx')


# In[43]:


print('Shape of Data: ', df.shape)

def get_agg(df):
    
    dtypes  = df.dtypes.to_frame('Data Types')
    is_nulls = df.isnull().any().to_frame('Is Null')
    count_null = df.isnull().sum().to_frame('Count Null')
    distinct = df.nunique().to_frame('Distinct Count')

    stats = df.describe().T
    aggd = dtypes.            merge(is_nulls, left_index=True, right_index=True).            merge(count_null,left_index=True, right_index=True).            merge(distinct,left_index=True, right_index=True).            merge(stats, how = 'left',left_index=True,right_index=True).            fillna('--')
            
    return aggd

display(get_agg(df))


get_agg(df).to_clipboard()



# ____
# ### Visualizing the Endogenous Variable 

# In[44]:


y = df.VehicleSalePrice
ecdf = ECDF(y)



fig,ax = plt.subplots(ncols = 2, figsize = (8,5))
plt.subplots_adjust(wspace = 0.5)

ax[0].hist(y, bins = np.arange(0,y.max(),10000), edgecolor = 'white', color = 'steelblue')
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('Selling Price\n(in Hundreds of Thousands)')

ax[1].plot(ecdf.x, ecdf.y, color = 'white')
ax[1].set_ylabel('Cumulative Distribution')
ax[1].set_xlabel('Selling Price\n(in Hundreds of Thousands)')

ax[1].axhline(0.98, linestyle = 'dashed', color = 'red',label = r'$98^{th}$ Percentile')
ax[1].legend()
for a in ax:

    a.ticklabel_format(style = 'sci', axis = 'x', scilimits=(0,0))
    a.ticklabel_format(style = 'sci', axis = 'y', scilimits=(0,0))
    sns.despine(ax = a)

plt.tight_layout()
plt.savefig('/Users/demetri/Desktop/dist.pdf', transparent = True)


# In[ ]:





# In[45]:


fig,ax = plt.subplots(ncols = 2, nrows = 1, figsize = (8,5))
plt.subplots_adjust(wspace =0.5)

ax = ax.ravel()

order= df.VehicleMake.value_counts().sort_values(ascending = False).index

sns.barplot(data = df,
            y = 'VehicleMake',
            x = 'VehicleSalePrice',
            orient = 'h', 
            color = 'steelblue', 
            ax = ax[0],
            order = order
           )

sns.countplot(data = df, 
              y = 'VehicleMake',
              orient = 'h', 
              color = 'steelblue', 
              ax = ax[1],
               order = order)

plt.tight_layout()
plt.savefig('/Users/demetri/Desktop/price_count.pdf', transparent = True)

# sns.barplot(data = df,
#             y = 'VehicleYear',
#             x = 'VehicleSalePrice',
#             orient = 'h', 
#             color = 'steelblue', 
#             ax = ax[2],
#            )


# sns.countplot(data = df, 
#               y = 'VehicleYear',
#               orient = 'h', 
#               color = 'steelblue', 
#               ax = ax[3]
#              )


# In[46]:


fig,ax = plt.subplots(ncols = 2, figsize = (18,10))

order = df.groupby('VehicleMarketClassDisplay').VehicleSalePrice.mean().sort_values(ascending = False).index

plt.subplots_adjust(wspace = 0.5)
sns.barplot(data = df,
            y = 'VehicleMarketClassDisplay',
            x = 'VehicleSalePrice',
            orient = 'h', 
            color = 'steelblue', 
            ax = ax[0],
           order = order)

sns.countplot(data = df, 
              y = 'VehicleMarketClassDisplay',
              orient = 'h', 
              color = 'steelblue', 
              ax = ax[1],
             order = order)


# In[47]:


ax = df.loc[df.HasPoliceReportedAccident==1,['VinValue','VehicleYear','VehicleMake','VehicleModel','HasPoliceReportedAccident','VehicleSalePrice','ContractDate']]
nax = df.loc[df.HasPoliceReportedAccident==0,['VinValue','VehicleYear','VehicleMake','VehicleModel','HasPoliceReportedAccident','VehicleSalePrice','ContractDate']]

F = ax.merge(nax,on =['VehicleYear','VehicleMake','VehicleModel'])

axx = F.drop_duplicates(subset = ['VinValue_x','ContractDate_x']).VehicleSalePrice_x
naxx = F.drop_duplicates(subset = ['VinValue_y', 'ContractDate_y']).VehicleSalePrice_y

print(naxx.mean() - axx.mean())
plt.hist(axx,normed = True, alpha = 0.5)
plt.hist(naxx, normed = True, alpha = 0.5)


# ___
# 
# # Feature Engineering

# In[48]:


#First, we can convert the dates into days since most recent observation

df['AccidentRecency'] = (df.ContractDate- df.AccidentDate).dt.days
df['Age'] = (df.ContractDate - pd.to_datetime(df.VehicleYear,format = '%Y')).dt.days/365
# df['VehicleYear'] = df.VehicleYear - df.VehicleYear.min()

def TOM(x):
    
    if x<10:
        return 1
    elif x<20:
        return 2
    else:
        return 3

# df['TOM'] = df.ContractDate.dt.day.apply(TOM)

df['SaleCount'] = df.groupby('VinValue').cumcount()
df.loc[df.HasPoliceReportedAccident==0,'AccidentRecency'] = -1
df.loc[df.HasPoliceReportedAccident==0,'AccidentDetail'] = 'None'


# In[ ]:





# In[62]:


sf =df.drop(['AccidentDate','ContractDate','VinId','VehicleMarketClassId','TotalClaims','Company'],axis = 1)
sfix = sf.isnull()
get_agg(sf)


# In[63]:


sf_numerical = sf.select_dtypes(include = [np.number])
sf_cat = sf.select_dtypes(exclude = [np.number])

sf_cat = sf_cat.apply(lambda x: pd.Categorical(x))


sfn = pd.concat((sf_numerical,sf_cat), axis = 1).loc[:,sf.columns]

sfn.head()


# In[64]:


F = sfn.groupby(['VehicleMake','VehicleModel','VehicleTrim']).VehicleSalePrice.mean().to_frame('Trim_Ord')

Trim_ord = (F/F.sum()).sort_values('Trim_Ord').reset_index()

sfn=sfn.merge(Trim_ord, how = 'left').drop('VehicleTrim', axis = 1)


F =sfn.groupby(['VehicleMake','VehicleModel']).VehicleSalePrice.mean().to_frame('Model_Ord')
Model = (F/F.sum()).sort_values('Model_Ord').reset_index()
sfn=sfn.merge(Model, how = 'left').drop('VehicleModel', axis = 1)


F =sfn.groupby('VehicleMarketClassDisplay').VehicleSalePrice.mean().to_frame('Market_Ord')
Market = (F/F.sum()).sort_values('Market_Ord').reset_index()
sfn=sfn.merge(Market, how = 'left').drop('VehicleMarketClassDisplay', axis = 1)


for col in ['City','company_id','dealer_type','Province','FSA','AccidentDetail']:
    
    F =sfn.groupby(col).VehicleSalePrice.mean().to_frame(f'{col}_ord')
    Market = (F/F.sum()).sort_values(f'{col}_ord').reset_index()
    sfn=sfn.merge(Market, how = 'left').drop(col, axis = 1)

    
sfnn = pd.get_dummies(sfn,columns = ['VehicleMake','HasPoliceReportedAccident'],drop_first=[False,True])

# sfnn = pd.get_dummies(sfn,columns = ['HasPoliceReportedAccident'],drop_first=True)


# In[65]:


# sfnn.dropna().to_csv('correlated.csv',index = False)

sfnn.dropna().to_csv('correlated_2.csv',index = False)


# In[66]:


# from sklearn.feature_selection import mutual_info_regression,f_regression

# MI = mutual_info_regression(sfnn.dropna().drop('VehicleSalePrice',axis = 1),sfnn.dropna().VehicleSalePrice)
# F = f_regression(sfnn.dropna().drop('VehicleSalePrice',axis = 1).values,sfnn.dropna().VehicleSalePrice.values)


# In[67]:


# MI/=np.max(MI)
# F/=np.nanmax(F)
# d = pd.DataFrame(list(zip(sfnn.drop('VehicleSalePrice',axis = 1).columns,MI,F)), columns = ['Feat','MI','F'])
# d.sort_values('MI', ascending = False)


# In[68]:


# F,p = f_regression(sfnn.dropna().drop('VehicleSalePrice',axis = 1).values,sfnn.dropna().VehicleSalePrice.values)


# In[69]:


(df.groupby('VinValue').count().VinId>=2).sum()


# In[ ]:





# In[ ]:





# In[ ]:




