## Done


### Regression

	- 크게 범주형 종속인 로지스틱과 연속형 종속인 선형 회귀
	- 분석모형을 융합하는 방법에는 Boosting, Bagging, RandomForest(Ensemble학습) 등이 있는데, 매개변수 최적화 기법으로 Gradient Descent(Batch, Stochastic), 경사하강법, 이외에도 모멘텀, AdaGradient(학습률을 조정), Adam(학습률 조정 + 모멘텀)등을 사용하고 있다. 결국 Residual을 어떤식으로 개선시키는지에 대한 다양한 방식이 있다고 이해하면 될 것이다.

> > Strategy whether two or more datas are overlapped with Nan or not

`df[‘A_col’].isna() & df[‘B_col’].isna()).sum()`
`df[‘A_col’].isna().sum()`

> > When is the time to clean data?

	- 실제로 수치형과 범주형을 시각화로서 상관성 분석을 해보고나서,
	- 이상치의 문제점을 확인한 뒤에야, 클리닝의 필요성을 인지하고 시작하는 것이 바람직하며, 시간 소모가 아니다.

## To Do

	- AirBnB Data Cleaning & Modeling
