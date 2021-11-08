## Done

### Modeling

> Classification Evaluation

- F1-Score != Accuracy, Precision(높을 수록 헛다리 짚는 공수가 줄어든다.-긍정 샘플 수가 적으면 왜곡), Recall(높을 수록 검출력이 좋다.-모두 긍정이라해서 맞혔어, 왜곡)

> Regression Evaluation

	- MAE, RMSE – 모델끼리 비교 외, 종속변수의 Scale에 따라 어느 정도 감안해서 평가할 수 있다.(예를 들어, 코스피 모델 예측에서 1000이다. 무의미하지만, 전세계 인구에서 그랬다면, 훌륭하다.)

	- sklearn Input Structure: [[col1], [col2], ..., [colN]] – 2d
`X.shape(-1,1)`: X = [col1,col2, ... , colN]

	- MLP(ANN), SVM 계산량이 있는 모델, 복잡도의 한계가 무궁무진한 모델의 경우, lambda는 물론, max_iter도 조절한다.

	- nd.array는 할당을 하면 사본이 아닌 view를 반환한다.
`X_new = X.copy()`

	- SVM

	1. kernel: 이진 변수가 많으면 linear, 연속 변수가 많으면 rbf
	2. C: 클수록 하드마진, 작을수록 소프마진, 10**n범위에서 튜닝
	3. gamma: kernel시 클수록 곡률이 복잡, 즉 C, gamma는 함께 조정한다. 

	- ANN

	1. 은닉층 구조 조절이 관건
	2. 시계열 예측, 이미지 분류, 객체 탐지 등을 제외하고는 깊은 층의 신경망이 과적합으로 인한 성능 이슈가 자주 발생한다.

	- Ensemble

	1. Random Forest: Bagging 방식 여러 트리 학습 결합 모델
	2. XGBoost & LightGBM: Boosting 방식 여러 트리 순차적으로 학습 결합 모델
	3. 트리의 개수가 많을수록 좋겠지만, 계산 시간 고려 + 어느 수 이상이면 수렴/ 나무 최대 깊이는 4이하일 때 과적합을 피할 수 있더라/ 학습률, 크면 클수록 과적합, 작으면 과소적합 (0.1 or 0.05)

	- Grid Search-ParameterGrid(조합을 다 일일이 풀어준다.)

`from sklearn.model_selection import ParameterGrid`

`grid = {“n_neighbors” : [3,5,7], “metrics” : [“Manhattan”, “Euclidean”] }`

`list(ParameterGrid(grid))`: 조합을 dic으로 풀어서 list 저장

	- Macro vs Micro Average

	1. Macro : 보통 평균
	2. Micro : 샘플 개수의 평균 – 클래스 불균형으로 인한 Macro의 왜곡을 보완해준다.

	- df.shape() – 행: 샘플수, 열, 특징 수/ 특징 수가 적을 때 되려 SVM, MLP와 같은 복잡한 모형을 쓰는 것은 되려 독이 된다.경향상 샘플 수가 많은데, 특징이 적을 때는, 여러 트리들을 앙상블하는 모델, RF, XGBoost, LightGBM이 성능이 좋게 나왔을 것이다.

	1. 그렇다면, 샘플수가 많은지 특징 수가 많은지 어떻게 판단할 수 있느냐? 대략적으로 샘플 수는 10000개, 그리고 특징 수는 30을 했을 때 샘플 수보다 많은지 적은지. 근데 주의할 점은, 모델에 실질적으로 학습하는 샘플수를 기준으로 해야한다! 예를들어 337이라할 때, 8:2로 학습시킨다면, 실제 비교시 참고해야할 샘플수 260개정도로 본다.

 

> Variable Type

`DataFrame.dtypes`
`DataFrame.infer_objects().dtypes` : ‘1’을 int로 추론 – 잘 안되기도 함.

	- 상태 공간의 크기 + 도메인 지식

	- apply도 순회하는 느낌 > for 구문 (속도가 빠르다)

 

> > 혼합형 변수가 적절하지 않은 모델

	1. KNN, Kmeans 분류 지표가 거리=유사도 – 스케일 영향을 많이 받는다. 그러므로 혼합은 좋지 않다.(단, Knn-코사인 유사도는 예외)
	2. 회귀모델: 변수의 스케일 차이 – 계수의 차이- 예측 안정성이 떨어짐. 하나의 변수에 더더욱 극적인 영향을 받는 것이므로.
	3. 나이브 베이즈: 하나의 확률 분포를 가정하므로, 안된다. -이진형: 베르누이 – 다항: 다항 – 연속형 : 가우시안.

> Complexity Parameter

	1. max_iter(ex:신경망, SVM) : ConvergenceWarning 경우는 학습을 더 시킨다면 수렴을 하므로 더 나아진다. 반면에, 여기서는 학습이 됐는데도 불구, 더 추가로 시행한다면 과적합 성능 하락할 가능성.

	2. SVM – kernel : poly(degree) 잘쓰지 않는다. > rbf(gamma) > linear (과적합)

	3. Logistic – C(계수 패널티): 클수록 정규화, 작을수록 과적합

	4. ANN- hidden_layer_sizes

	5. SVR-epsilon(최대허용오차): 클수록 소프트마진, 작을수록 하드(복잡도 커짐)

	6. DT: learning-rate: 클수록 복잡도 커짐, 작을수록 과소적합 단순화

## To Do

	- Machine Learning Prep & Improving
