# import tensorflow as tf
from tensorflow.keras.models import Sequential # 시퀀셜 모델과
from tensorflow.keras.layers import Dense # 덴스 레이어를 쓸수있다. 
import numpy as np


#1. 데이터 로드
x =  np.array([1,2,3]) # shape=(3, 1)
y =  np.array([1,2,3])

#2. 모델구성 MLP(다층 퍼셉트론)
model = Sequential()
model.add(Dense(5, input_dim=1)) # input layer
model.add(Dense(3)) 
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1)) # output layer

'''단층신경망에 비해 훈련횟수(epochs)를 훨씬 줄여도 loss값을 구할 수 있다.'''

#3. 컴파일
model.compile(loss='mse', optimizer='adam') 

#4. 훈련
model.fit(x, y, epochs=200, batch_size=1) 

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4]) # 새로운 x값을 predcit한 결과 
print('4의 예측값 : ', result)

# loss 값 : 0.38xx
# 6의 예측값 : 5.55xx

