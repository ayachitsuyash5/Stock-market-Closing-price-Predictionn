#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Assignment 2
# 

# # 
# 
# Collect the data of 1 to 2 yrs. of any equity stock price from nseindia website.
# Follow the instructions below to build time series model.
# 1. Each student’s stock price data should be unique.
# 2. Apply relevant EDA techniques ( for eg. Uni variate, Bi variate, outlier treatment etc..)
# 3. Build time series models (from Exponential smoothing techniques to ARIMA models) on close price. And predict the next 15 days stock price using your model.
# 4. Interpret your results.

# Stock of comapny selected - Hindustan Unilever Ltd.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sn
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model  import ARIMA


# In[2]:


rcParams['figure.figsize']=15,6


# In[3]:


import  scipy.signal.signaltools

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered


# In[4]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[5]:


df= pd.read_csv('05-04-2020-TO-04-04-2022HINDUNILVREQN.csv',parse_dates=True,)


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.size


# In[9]:


df.describe()


# # Data Cleaning

# In[10]:


df.isnull().sum()


# In[11]:


df.info()


# In[ ]:





# In[12]:


df.duplicated().sum()


# # EXPLORATORY DATA ANALYSIS (HINDUSTAN UNILEVER STOCK PRICE)

# In[13]:


df.hist(figsize=(15,12),bins = 15)
plt.title("Features Distribution")
plt.show()


# In[14]:


#Univariate Analysis of close price HUL Stock 
plt.scatter(df.index,df['Close Price'])
plt.show()                               


# In[15]:


df.hist(column='Close Price', grid=False, edgecolor='black')


# In[16]:


sns.boxplot(x=df["Close Price"])


# In[17]:


# Finding the relations between the variables.
plt.figure(figsize=(20,10))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# # Build time series models (from Exponential smoothing techniques to ARIMA models) on close price. And predict the next 15 days stock price using your model.

# # Exponential Smoothing

# In[7]:


date = pd.date_range(start = '05/04/2020', end = '04/04/2022', freq='D')
date


# In[8]:


df['DAYS'] = pd.DataFrame(date)
df


# In[9]:


df.set_index(keys = 'DAYS', drop = True, inplace = True)
df.head()


# In[10]:


# Remove extra columns
col_remove = ['DATE','Series','Prev Close', 'Open Price', 'High Price', 'Low Price',
       'Last Price', 'Average Price', 'Total Traded Quantity',
       'Turnover', 'No. of Trades', 'Deliverable Qty',
       '% Dly Qt to Traded Qty']
df = df.drop(col_remove, axis = 1)


# In[11]:


df


# In[13]:


df.plot(grid=True)


# In[14]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[15]:


from pylab import rcParams


# In[16]:


rcParams['figure.figsize'] = 14,7
decomposition = seasonal_decompose(df,model = 'additive')
decomposition.plot()


# In[17]:


rcParams['figure.figsize'] = 14,7
decomposition = seasonal_decompose(df,model = 'multiplicative')
decomposition.plot()


# In[79]:


train = df[df.index<'2022']
test = df[df.index<'2022']


# In[80]:


train.shape


# In[81]:


test.shape


# In[82]:


import statsmodels.tools.eval_measures as      em
from   sklearn.metrics                 import  mean_squared_error
from   statsmodels.tsa.api             import ExponentialSmoothing, SimpleExpSmoothing, Holt
from   IPython.display                 import display
from   pylab                           import rcParams


# In[83]:


from sklearn.model_selection import TimeSeriesSplit


# In[84]:


model_ses = SimpleExpSmoothing(train,initialization_method='estimated')


# In[85]:


model_ses_autofit = model_ses.fit(optimized=True)


# In[86]:


model_ses_autofit.params


# In[87]:


ses_predict = model_ses_autofit.forecast(steps=len(test))


# In[88]:


ses_predict


# In[89]:


plt.style.use('seaborn')
plt.plot(train, label='Train')
plt.plot(test, label='Test')

plt.plot(ses_predict, label='Alpha =0.99 Simple Exponential Smoothing predictions on Test Set')

plt.legend(loc='best')
plt.grid()
plt.title('Alpha = 0.99 Predictions');


# In[90]:


def MAPE(y_true, y_pred):
    return np.mean((np.abs(y_true-y_pred))/(y_true))*100


# In[91]:


print('SES MSE:',mean_squared_error(test.values,ses_predict.values,squared=False))
#different way to calculate RMSE
print('SES RMSE (calculated using statsmodels):',em.mse(test.values,ses_predict.values)[0])
print('SES MAPE :',MAPE(test.values,ses_predict.values))


# In[92]:


resultsDf = pd.DataFrame({'Test RMSE': [em.rmse(test.values,ses_predict.values)[0]]},index=['Alpha=0.99,SES'])
resultsDf


# In[93]:


# Initializing the Double Exponential Smoothing Model
model_DES = Holt(train,initialization_method='estimated')
# Fitting the model
model_DES = model_DES.fit()

print('')
print('==Holt model Exponential Smoothing Estimated Parameters ==')
print('')
print(model_DES.params)


# In[94]:


# Forecasting using this model for the duration of the test set
DES_predict =  model_DES.forecast(len(test))
DES_predict


# In[95]:


## Plotting the Training data, Test data and the forecasted values

plt.plot(train, label='Train')
plt.plot(test, label='Test')

plt.plot(ses_predict, label='Alpha=0.99:Simple Exponential Smoothing predictions on Test Set')
plt.plot(DES_predict, label='Alpha=0.099,Beta=0.0001:Double Exponential Smoothing predictions on Test Set')

plt.legend(loc='best')
plt.grid()
plt.title('Simple and Double Exponential Smoothing Predictions');


# In[96]:


print('DES RMSE:',mean_squared_error(test.values,DES_predict.values,squared=False))
print('DES MAPE :',MAPE(test.values,DES_predict.values))


# In[97]:


resultsDf_temp = pd.DataFrame({'Test RMSE': [mean_squared_error(test.values,DES_predict.values,squared=False)]}
                           ,index=['Alpha=1,Beta=0.0001:DES'])

resultsDf = pd.concat([resultsDf, resultsDf_temp])
resultsDf


# In[98]:


# Initializing the Double Exponential Smoothing Model
model_TES = ExponentialSmoothing(train,trend='multiplicative',seasonal='multiplicative',initialization_method='estimated')
# Fitting the model
model_TES = model_TES.fit()

print('')
print('==Holt Winters model Exponential Smoothing Estimated Parameters ==')
print('')
print(model_TES.params)


# In[99]:


# Forecasting using this model for the duration of the test set
TES_predict =  model_TES.forecast(len(test))
TES_predict


# In[100]:


## Plotting the Training data, Test data and the forecasted values

plt.plot(train, label='Train')
plt.plot(test, label='Test')

plt.plot(ses_predict, label='Alpha=1:Simple Exponential Smoothing predictions on Test Set')
plt.plot(DES_predict, label='Alpha=0.99,Beta=0.001:Double Exponential Smoothing predictions on Test Set')
plt.plot(TES_predict, label='Alpha=0.25,Beta=0.0,Gamma=0.74:Triple Exponential Smoothing predictions on Test Set')

plt.legend(loc='best')
plt.grid()
plt.title('Simple,Double and Triple Exponential Smoothing Predictions');


# In[101]:


print('TES RMSE:',mean_squared_error(test.values,TES_predict.values,squared=False))
print('TES MAPE :',MAPE(test.values,TES_predict.values))


# In[102]:


resultsDf_temp = pd.DataFrame({'Test RMSE': [mean_squared_error(test.values,TES_predict.values,squared=False)]}
                           ,index=['Alpha=0.25,Beta=0.0,Gamma=0.74:TES'])

resultsDf = pd.concat([resultsDf, resultsDf_temp])
resultsDf


# #  ARMA Model

# In[103]:


## Auto Correlation Function plot
plot_acf(df, lags=20, ax=plt.gca());


# In[104]:


## AR Model 
ar1modell= ARIMA(df, order=[2,0,0]).fit()
ar1modell.summary()


# In[105]:


ar1modell.summary()


# In[131]:


ARIMA(df, order=[2,0,0]).fit().summary()


# In[110]:


ARMA(df, order=[3,0]).fit()


# In[111]:


## Auto Correlation Function plot
plot_pacf(df, lags=10, ax=plt.gca());


# # ARIMA Model

# In[114]:


from statsmodels.tsa.arima.model  import ARIMA


# In[115]:


## Moving Average(1)
ARIMA(df, order=[0,0,2]).fit().summary()


# In[116]:


from statsmodels.tsa.stattools import adfuller


# In[117]:


df.shape


# In[118]:


statioonary_test = adfuller(df, autolag='AIC')
statioonary_test


# In[119]:


ARIMA(df,order=[1,1,2]).fit().summary() 


# In[120]:


get_ipython().system('pip install pmdarima')


# In[121]:


from pmdarima import auto_arima


# In[123]:


auto_arima(df,start_p=0, d=0,start_q=0,max_p=5,max_d=2,max_q=5).summary()


# In[124]:


import itertools
p = q = range(0, 4)
d= range(0,2)
pdq = list(itertools.product(p, d, q))
model_pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Model...')
print('Model: {}{}'.format(pdq[1], model_pdq[1]))
print('Model: {}{}'.format(pdq[1], model_pdq[2]))
print('Model: {}{}'.format(pdq[2], model_pdq[3]))
print('Model: {}{}'.format(pdq[2], model_pdq[4]))


# In[125]:


arima_AIC= pd.DataFrame(columns=['param', 'AIC'])
arima_AIC


# In[126]:


import warnings
warnings.filterwarnings("ignore")

for param in pdq:
    arima_mod= ARIMA(df, order=param).fit()
    print('ARIMA{} - AIC: {}'.format(param,arima_mod.aic))
    arima_AIC= arima_AIC.append({'param':param, 'AIC':arima_mod.aic}, ignore_index=True)


# In[127]:


arima_AIC.sort_values(by='AIC', ascending=True)


# In[128]:


auto_arima(df, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)


# In[ ]:




