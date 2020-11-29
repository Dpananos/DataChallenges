
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import pickle
from scipy.stats import boxcox,probplot, norm
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['lines.color'] = 'k'
# mpl.rcParams['axes.labelcolor'] = 'white'
# mpl.rcParams['axes.edgecolor'] = 'white'
# mpl.rcParams['xtick.color'] = 'white'
# mpl.rcParams['ytick.color'] = 'white'

get_ipython().magic('matplotlib inline')


# In[18]:


df = pd.read_csv('/Users/demetri/Documents/Python/correlated_2.csv')



# In[19]:



Y,L=boxcox(df.VehicleSalePrice.values)
Xdf=df.drop('VehicleSalePrice', axis = 1)
X = Xdf.values



# In[20]:


X_train,X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.6, random_state = 56765)


# In[27]:





# In[15]:


#Training for XGB

#Takes forever to run.
boosted = xgb.XGBRegressor()

params = {
    
    'n_estimators': [1000,1500,2000],
    'learning_rate': [0.1,0.2,0.3],
    'max_depth': [3,5,10]
        }

gs = GridSearchCV(boosted, param_grid=params, n_jobs = -1, verbose = 4, scoring = 'neg_mean_squared_error')

gs.fit(X_train,Y_train)


print("R squared:",r2_score(Y_test,gs.best_estimator_.predict(X_test)))



# In[17]:


X_train.shape


# In[5]:


gs = pickle.load(open('boxcox.sav','rb'))


# In[ ]:


y_pred= gs.best_estimator_.predict(X_test)
err = Y_test - y_pred


# In[ ]:


print('R Squared:', r2_score(Y_test,y_pred))
print('MSE:', mean_squared_error(Y_test,y_pred))


# In[ ]:


lm = LinearRegression()
lm.fit(X_train,Y_train)

print('R Squared:', r2_score(Y_test,lm.predict(X_test)))
print('MSE:', mean_squared_error(Y_test,lm.predict(X_test)))


# In[ ]:


step = 0.5
bins = np.arange(-5,5+step,step)
plt.hist(err, color = 'steelblue', edgecolor = 'white', bins = bins, normed = True,align = 'mid')

param = norm.fit(err)
dist = norm(loc = param[0],scale = param[1])
x = np.linspace(-5,5,1001)

plt.plot(x,dist.pdf(x), color = 'white',label = 'Fitted Gaussian')

ax = plt.gca()
sns.despine(ax = ax)
plt.xlabel('Residual')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()



plt.savefig('/Users/demetri/Desktop/Error_Dist.pdf', transparent = True)




# In[ ]:


plt.scatter(Y_test,err, alpha = .5, c='steelblue',edgecolor ='grey', linewidth = 0.05)

plt.xlabel('Observed')
plt.ylabel('Residual')


ax = plt.gca()
ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


plt.tight_layout()

plt.savefig('/Users/demetri/Desktop/RvO.pdf', transparent = True)


# In[ ]:


scores = gs.best_estimator_.booster().get_fscore()

S =[]

for i ,f in enumerate(scores.keys()):
    
    
    S.append([Xdf.columns[i],scores[f]])


# In[ ]:


ax= pd.DataFrame(S, columns = ['Feature','F']).sort_values('F', ascending = True).iloc[-10:,:].set_index('Feature').plot(kind = 'barh', legend = False)

ax.set_xlabel('Number of Splits')
ax.set_ylabel('Feature')

plt.tight_layout()

sns.despine(ax = ax)

plt.savefig('/Users/demetri/Desktop/Feature_impo.pdf', transparent = True)


# In[ ]:


plt.hist(Y,edgecolor = 'white', bins = np.arange(10,60,2))
plt.ylabel('Frequency')
plt.xlabel('Transformed Sales Price')

ax = plt.gca()
sns.despine(ax = ax)

plt.savefig('/Users/demetri/Desktop/Transformed.pdf', transparent = True)


# In[17]:


y_pred = gs.best_estimator_.predict(X_test)

rel_err = (Y_test-y_pred)/Y_test




# In[18]:


plt.scatter(Y_test,rel_err)
plt.axhline(-0.1, color = 'r')
plt.axhline(0.1, color = 'r')

(abs(rel_err)<0.1).sum()/rel_err.size


# In[ ]:




