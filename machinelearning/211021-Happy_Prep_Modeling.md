## Done

### Preprocessing

> StandardScaler

`from sklearn.preprocessing import StandardScaler`
`scaler = StandardScaler()`
`scaler.fit(X)` | `scaler.fit(X_train)`

	1. X를 fit 해주는 것은 X_test를 모두 얻을 수 있는 상황일 때
	2. X_train만 fit 해주는 것은 X_test를 완전히 배제하고 모델을 학습시키므로, 즉 X_test를 얻을 수 없는 상황일 때

> Analyze Missing Values

- 단지 개수가 적고, Nan이라는 이유로 제거할 생각만 하지 말고, 이유를 분석해라.
	1. 왜? Nan값이 나왔을까? 지금은 전문가에 의해 처리된 데이터인데…
	2. 아! 다 더해보니, 0이 Nan으로 처리되었구나!

> Compare Efficiency of Models

- LinearRegression 과 XGboost를 Regression 분석할 경우
- 지표는 1.mean_absolute_error(y_test,pred) 2.sqrt(mean_squared_error(y_test, pred)
- 오차의 값을 통해, 성능 비율이 비교가 가능하다!

> Visualize Coefficiency of Regression

`plt.bar(X_train.columns, model_lr.coef_)`

- 표준화된 값에 대한 coefficiency이므로 단순한 상관관계, 지표의 영향력이라고 해석할 수 없다.
	1. 정규화 과정이 평균과 편차라는 것을 고려할 때, 편차가 크므로 모형의 변동을 설명하는 정도가 높아서, 이 수치가 함께 높아진 것이다. 즉, 국가별 economy의 편차 정도라는 것!
	2. 단, 정규화가 되지 않은 경우에는 단순 상관관계로 해석할 수 있을 것이다.

`plt.bar(X_train.columns, model_xgb.feature_importances_)`

- 위와 같은 순서로 나온다.
- 단, 그 정도가 다른데, 그것은 모델의 알고리즘 차이로 인해 이것이 Linear과 달리 비선형적인 부스팅하기 때문이다.

## To Do

	- New Game Data EDA
