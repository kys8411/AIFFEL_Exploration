#!/usr/bin/env python
# coding: utf-8

# In[2]:


# (1) 데이터 가져오기

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

data = pd.read_csv('~/aiffel/bike-sharing-demand/train.csv')
print(data.shape)
data.head()
data.info()


# In[3]:


data['datetime']


# In[4]:


# (2) datetime 컬럼을 datetime 자료형으로 변환하고 연, 월, 일, 시, 분, 초까지 6가지 컬럼 생성하기
import datetime


data['year'] = data['datetime'].map(lambda x : int(str(x)[:4]))
data['month'] = data['datetime'].map(lambda x : int(str(x)[5:7]))
data['day'] = data['datetime'].map(lambda x : int(str(x)[9:10]))
data['hour'] = data['datetime'].map(lambda x : int(str(x)[12:13]))
data['min'] = data['datetime'].map(lambda x : int(str(x)[15:16]))
data['sec'] = data['datetime'].map(lambda x : int(str(x)[18:19])
                                  )

data.info()
data.head()


# In[5]:


# (3) year, month, day, hour, minute, second 데이터 개수 시각화하기
# sns.countplot 활용해서 시각화하기
# subplot을 활용해서 한 번에 6개의 그래프 함께 시각화하기

import matplotlib.pyplot as plt
import seaborn as sns

year_countplot = sns.countplot(x = 'year', data = data, order = data['year'].value_counts().index)
month_countplot = sns.countplot(x = 'month', data = data, order = data['month'].value_counts().index)
day_countplot = sns.countplot(x = 'day', data = data, order = data['day'].value_counts().index)
hour_countplot = sns.countplot(x = 'hour', data = data, order = data['hour'].value_counts().index)
min_countplot = sns.countplot(x = 'min', data = data, order = data['min'].value_counts().index)
sec_countplot = sns.countplot(x = 'sec', data = data, order = data['sec'].value_counts().index)

plt.plot()




# In[11]:


# (4) X, y 컬럼 선택 및 train/test 데이터 분리

data.info()

x = data[['season','holiday','workingday', 'weather', 'temp', 'atemp', 'windspeed', 'year','month','day','hour']]
y = data[['count']]

print(x.head())
print(y.head())


# In[12]:


# train / test 데이터분리 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape, x_train.shape)
print(x_test.shape, y_test.shape)


# In[13]:


# (5) LinearRegression 모델 학습

from sklearn.linear_model import LinearRegression

model = LinearRegression()


# In[14]:


# (6) 학습된 모델로 X_test에 대한 예측값 출력 및 손실함수값 계산
# 학습된 모델에 X_test를 입력해서 예측값 출력하기

model.fit(x_train, y_train)

predictions = model.predict(x_test)
predictions


# In[15]:


# 모델이 예측한 값과 정답 target 간의 손실함수 값 계산하기

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
mse


# In[19]:


# mse 값과 함께 rmse 값도 계산하기

RMSE = mean_squared_error(y_test, predictions)**0.5  # 기본값이 Squared(mse) 되어 있음.
RMSE


# In[23]:


# (7) x축은 temp 또는 humidity로, y축은 count로 예측 결과 시각화하기

plt.scatter(data[['temp']], y[['count']])
plt.scatter(data[['humidity']], y[['count']])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# 넘파이 array


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




