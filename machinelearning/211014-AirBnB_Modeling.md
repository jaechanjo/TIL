## Done


### Preprocessing

- get_dummies : 범주형 데이터 전처리

- StandaradScaler: 수치형 데이터 표준화 (종속변수는 뺀다!)

### Evaluation

- 범주형 데이터를 학습에 많이 쓰면, 그 결과도 범주화되어 나오는 경향이 있다.(성능 개선 고려 사항)

- 오차율과 잔차의 차이
	1. 오차율은, 예측이 꼬리를 내며 극단적으로 잘못 경우 확인, + 과대평가인지, - 과소평가인지를 확인한다.
`err = (pred – y_test) / y_test`
`sns.histplot(err)`
`plt.grid()`

	2. 잔차는, 실제 0을 기준으로 얼마만큼의 차이를 내는 +,-로 카운트, 개수가 분포해있는지 확인한다.
`err = (pred – y_test)`
`sns.histplot(err)`
`plt.grid()`

## To Do

	- Chapter 3. Korea Happiness Index Data EDA
