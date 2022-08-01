# import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np

x =  np.array([1,2,3,5,4]) # shape=(5, 1)
y =  np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1)) 

#3. 컴파일
model.compile(loss='mse', optimizer='adam') 

#4. 훈련
model.fit(x, y, epochs=2800, batch_size=1) 

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)
result = model.predict([5]) 
print('5의 예측값 : ', result)