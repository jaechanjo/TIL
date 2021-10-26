## Done

### PreProcessing

> Pruning Categorical Variables
	- 범주의 종류가 너무 많은 경우, 소수 빈도에 대해 Others로 묶어주기.

`pb = df[‘Publisher’].value_counts()`
`plt.plot(range(len(pb)), pb)`
`df[‘Publisher’].apply(lambda s: s if s not in pb[20:] else‘others’)`

### Evaluation

> RMSE & MAE

`pred_xgb = model_xgb.predict(X_test)`
`pred_lr = model_lr.predict(X_test)`

`print(‘XGB MAE’:, mean_absolute_error(y_test, pred_xgb))`
`print(‘XGB RMSE’:, sqrt(mean_squared_error(y_test, pred_xgb)))`

`print(‘LR MAE’:, mean_absolute_error(y_test, pred_lr))`
`print(‘LR RMSE’: sqrt(mean_squared_error(y_test,pred_lr)))`

> Limitation of Regression

	- 판매량이 많은 경우 대부분은 Box plot에서 이상치들이었다. 즉, 회귀 분석에 활용되지 못하고, 오차가 크게 벌어지게 된 것이다.

	- Linear Regression 같은 경우는 이상치들을 맞추려 가중치를 변경하며 더하다 보니, 낮은 값들이 되려 음수로 예측되는 경우가 발발했다.

	- Xgb.feature_importances_ 는 양음을 구분하지 않고, 종속변수에 대한 독립변수의 주요성 정도만을 보여준다.

## To Do

	- Chapter 5 EDA
