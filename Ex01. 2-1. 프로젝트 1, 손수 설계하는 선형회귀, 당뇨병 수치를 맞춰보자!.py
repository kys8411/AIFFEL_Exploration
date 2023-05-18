#!/usr/bin/env python
# coding: utf-8

# In[27]:


# (1) 데이터 가져오기

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

df_x=diabetes.data
df_y=diabetes.target


print(df_x.shape)
print(df_y.shape)


# In[28]:


# (2) 모델에 입력할 데이터 X 준비하기

df_x
print(type(df_x))

# <class 'numpy.ndarray'>
# 이미 numpy.ndarray로 되어있지만, 변환 저장하는 것 연습

df_x = np.array(df_x, dtype=np.float64)
df_x.dtype


# In[29]:


# (3) 모델에 예측할 데이터 y 준비하기

df_y
print(type(df_y))

# <class 'numpy.ndarray'>
# 이미 numpy.ndarray로 되어있지만, 변환 저장하는 것 연습

df_y = np.array(df_y, dtype=np.float64)
df_y.dtype


# In[33]:


# (4) train 데이터와 test 데이터로 분리하기

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state= 42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[51]:


# (5) 모델 준비하기

# 입력 데이터 개수에 맞는 가중치 W와 b를 준비해주세요.
# 모델 함수를 구현해주세요.

# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# print(model)

## 모델 함수 구현 ##

w = np.random.rand(10) # x가 10개의 벡터를 가지고 있으므로 W도 랜덤숫자 10개 
b = np.random.rand()   # y는 1개 랜덤 숫자 값으로 준비

print(w)
print(b)

# w에는 10개의 랜덤한 값으로 W1 부터 W10까지 넣음 
# b에는 하나의 상수를 넣음

def model(df_x, w, b):
    predictions = 0
    for i in range(10):
        predictions += df_x[:, i] * w[i]
    predictions += b
    return predictions
print("모델 함수 만들었슝!!")


# In[57]:


# (6) 손실함수 loss 정의하기
# 손실함수를 MSE 함수로 정의해주세요.

def MSE(a, b):
    mse = ((a - b)** 2).mean() # 두 값의 차이의 제곱의 평균 
    return mse
print("손실함수 만들었슝!!")


def loss(df_x, w, b, df_y):
    predictions = model(df_x, w, b)
    L = MSE(predictions, df_y)
    return L
print("에측값에서 실제값을 MSE로 계산해서 LOSS값 구함!!")


# In[53]:


# (7) 기울기를 구하는 gradient 함수 구현하기
# 기울기를 계산하는 gradient 함수를 구현해주세요.

def gradient(df_x, w, b, df_y):
    # n은 데이터 포인트의 개수
    N = len(df_y)

    # y_pred 준비 
    y_pred = model(df_x, w, b)
    
    # 공식에 맞게 gradient 계산
    dw = 1/N * 2 * df_x.T.dot(y_pred - df_y)
    
    # b의 gradient 계산
    db = 2 * (y_pred - df_y).mean()
    return dw, db
print("그라디언트도 만들었슝!!")


# gradient 계산

dw, db = gradient(df_x, w, b, df_y)
print("dW:", dw)
print("db:", db)


# In[87]:


# (8) 하이퍼 파라미터인 학습률 설정하기

LEARNING_RATE = 0.25


# In[88]:


# (9) 모델 학습하기
# 목표: MSE 손실함수값 3000 이하를 달성......ㅎㄷㄷㄷ
# 처음 3만 → 2800까지 줄어듬, 데이터 전처리를 통해서 더 줄일 수 있겠지만, 시간관계상...ㅠ


losses = []

for i in range(1, 1001):
    dw, db = gradient(x_train, w, b, y_train)
    w -= LEARNING_RATE * dw
    b -= LEARNING_RATE * db
    L = loss(x_train, w, b, y_train)
    losses.append(L)
    if i % 10 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))


# In[89]:


import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()

w, b


# In[90]:


# (10) test 데이터에 대한 성능 확인하기

prediction = model(x_test, w, b)
mse = loss(x_test, w, b, y_test)
mse


# In[91]:


# (11) 정답 데이터와 예측한 데이터 시각화하기

plt.scatter(x_test[:, 0], y_test)
plt.scatter(x_test[:, 0], prediction)
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


# 넘파이 array


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




