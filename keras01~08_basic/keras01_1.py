# import tensorflow as tf
from tensorflow.keras.models import Sequential # 시퀀셜 모델과
# 시퀀셜 모델 layer를 순찾거으로 추가
from tensorflow.keras.layers import Dense # 덴스 레이어를 쓸 수 있다.
import numpy as np

#1. 데이터 정제해서 값 도출
x = np.array([1,2,3]) # 입력 데이터, shape=(3,1)
y = np.array([1,2,3]) # 라벨

#2. 모델구성
model = Sequential() # 순차 모델
model.add(Dense(1, input_dim=1)) # 입력 = 벡터 1개, 출력 1개

#3. 컴파일
# 컴파일이란 모델을 학습시키기 위한 학습과정을 설정하는 단계이다.
# mse(mean squared error): 회귀 용도의 모델을 훈련시킬 때 사용되는 손실함수
model.compile(loss='mse', optimizer='adam') # 평균 제곱 에러 mse 이 값은 작을수록 좋다. optimizer='adam'은 mse값(loss) 감축시키는 역할. 85점 이상이면 쓸만하다.


#4. 훈련
model.fit(x, y, epochs=4200, batch_size=1 ) # epohs은 전체 데이터 훈련 횟수, batch_size는 한 번에 훈련 시키는 데이터량, batch가 작을수록 값이 정교해진다.

#5. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([4]) # 새로운 x값을 예측(predict)한 결과. predict()는 테스트 이미지의 분류 결과를 예측한다. 반환값이 예측 확률이다.
print('4의 예측값 : ', result)