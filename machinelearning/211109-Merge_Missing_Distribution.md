## Done

### Merge

> Axis by vector or matrix

 

`merged_df = pd.DataFrame()`
`for file in os.listdir(“test”):
`if ‘may’not in file and ‘.csv’ in file:`
`df = pd.read_csv(“test/”+ file)`
`merged_df = pd.concat([merged_df, df], axis =0, ignore_index=True)`
`merged_df.sort_values(‘date’, inplace=True)`

> Different Format

`def date_type_converter(value):`
`YYYY, MM, DD = value.split(‘ ‘)`
`return YYYY.replce(‘년’,’-‘) + “0” + str(int(MM)).replace(‘월’,’-‘) + “0” + str(int(DD)).replace(‘일’,’-‘)`

`df2[‘date’] = df2[‘date’].apply(date_type_converter)`

> Reference Data

`Rf_Series = ref_df.set_index(‘index_col’)[‘value_col’]`

`Rf_Series.to_dict()` : Rf_dict = {Rf_Series.index : Rf_Series.value ...} : set_index() because key is index

`S.replace(Rf_dict)`

`df.add_prefix(‘211109_’)` : 모든 columns 이름에 ‘211109_’가 붙는다. : merge 되는 두 df의 col을 구분해주려고!

> Distance_Matrix

`from scipy.spatial.distance import cdist`
`cdist(XA= A_df, XB = B_df, metric=’citybloc’)` : cityblock = manhattan

> > ndarray.argsort(axis=0 or 1)
	1. axis = 0:행별로 오름차순의 인덱스 반환
	2. axis = 1:열별로 오름차순의 인덱스 반환

	- haversine : 위경도 데이터로 거리 계산 모듈
	- df[‘new_col’] = list or ndarray -> df[][].values (df = df 꼴이면, 인덱스 충돌 발생)

`dist_mat = cdist(df_location, df3_location, metric = harversine)`
`close_subway_index = dist_mat.argsort()[:0]` :idxmax, idxmin보다 argsort[]가 좋은 이유가 순서대로 여러개도 뽑을 수 있으니까!
`df[‘가까운역’] = df3.iloc[close_subway_index][‘역명’].values`: loc 안써서 인덱싱하면 열 기본인데, loc쓰면 행 기본, 즉 행, 열로 구분해줘야 함
`df[‘가까운 역까지_거리’]=dist_mat[close_subway_index][:,0]`

> Duplicated Data

`S = df.groupby[‘ID’][‘val’].sum()`
	1. `df[‘sum’] = df[‘ID’].replace(S.to_dict()`:S_하나의 변수를 추가하는 경우!
	2. `pd.merge(df, S, left_on = ‘ID’, right_index = True)`

### Missing

> Delete(엄밀히 따지면, Train 기준으로 하되, 분할하지 않아도 상관없다.)

> > axis = 0
	1. cond: 결측이 제거되고 남은 데이터가 학습 성능 수렴에 충분한가?
	2. cond: 모델이 적용될 분야에서의 새로운 데이터에 앞으로도 결측이 없는가?

> > axis = 1
	- 차원 축소:(한 컬럼의 결측이 정말 많을 때에만 고려) 도메인 지식에 근거해서, 모델의 취지를 훼손하지 않는 차원인지 판단 – 새로운 데이터 상관 없지! 계속 그 특성은 빼면 되니까. 개수에도 상관없고!

	- 만약 정말 중요한 변수라면, 근데 결측치가 30%이상이라면, 그때는 판단해서 행단위 제거라던가, 대체를 하거나, 예측을 해서 가져가기로 한다.

`df.isnull() <-> ~df.isnull(), df.notnull()`
`df.dropna(axis=0, how =’any’)`

> SimpleImpute(Train과 Test를 완전히 격리시켜서, 모델의 객관성을 제고하기 위해, 분할하고 대체한다.)

`Train_X.corr().sum() / len(Train_X.columns) – 1)` : 하나의 변수가 그 나머지 변수와의 관계성 정도를 대략적으로 살펴볼 수 있다. -> 관계가 작으면 -> 대표값 대체 활용 가능

`from sklearn.impute import SimpleImputer`
`SI = SimpleImputer(strategy = ‘mean’)`
`SI.fit(Train_X)` #Train에 test가 섞이지 않도록
`Train_X = pd.DataFrame(SI.transform(Train_X), columns=Train_X.columns)` #Train에 test가 섞이지 않도록
`Test_X = pd.DataFrame(SI.transform(Test_X), columns=Train_X.columns)` #아, 새로운 데이터에도 결측이 생기면 이러한 SI과정으로 채워서 새로운 모델에 적용할 수 있지. 그러나 지우면, 매번 랜덤으로 발생하는 결측에는 대응할 수가 없지. 왜? 무작위 개체를 지워서 전체 구조를 다르게 만들지 말고, 채워서 전체 구조는 유지하자.

	- 범주형일 경우에는 most_frequent 최빈을 써야하니까 SI 모듈을 2개 만들어서 따로따로 처리해주면 되고, 이진형 범주 같은 경우는 평균이 곧 1의 비율이잖아, 그러니까 나중에 round해서 가까운 class로 편입시켜주면 된다.

> TimeSeries(시계열은 분할하지 않고 처리한다. 되려 랜덤 분할 시 시계열의 연속성이 깨진다.)

> > DataFrame.fillna()

	- 시계열은 연속적이므로 대부분의 경우, 이전 이후값과 맥락을 같이 한다.

`df.fillna(method= ‘ffill’).fillna(method=’bfill’)` : ffill로 채우고 안되는 지점은 bfill로 반대도 마찬가지이다.근데 생각을 해보면, 새로 들어오는 데이터에서는 늘 과거를 갖고 채울 수 있지. 그러니까. ffill을 먼저 쓰는게 자연스럽다. 또한 연속으로 결측이 발생하면, ffill을 쓰더라도 멀리 떨어진 값이 대체될 수 있지. 그러면 cumsum()으로 그런 양상인지 확인할 필요도 있겠다. 만약에 그렇다면, interpolation으로 선형적 관계로 채워줄 수도 있겠네, 기울기는 양끝단의 값을 이용해서 설정할 필요도 있고.

> Prediction

> > predict Missing col using No Missing col(결측 열과 비슷한 결측 패턴의 다른 열은 논외, 결측이 없는 열을 입력 결측 열에서 살아있는 값을 라벨로 지도학습하여 모델 구축하고, 이를 갖고 결측을 예측하는 방법)

	1. cond : 결측이 너무 많으면, 라벨 부족, 학습 일반화 불가
	2. cond : 결측X 열과 결측 열이 충분히 관계가 있어야해, 이때는 다중공선성이 절실한 순간이지. 나중에 하나를 제거할지라도 -> 결국 모델을 하나 만드는 과정이기에, 시간이 걸린다.

 

`Train_X.isnull().sum() / len(Train_X)` : 결측치 비율이 하나에 너무 쏠린 경우, 라벨이 부족해서 예측 일반화가 떨어진다.

`Train_X.corr().sum() / (len(Train_X.columns) – 1)`: 다른 변수간의 관계가 오히려 높아야 결측값 예측이 가능하다.

`from sklearn.impute import KNNImputer`
`KI = KNNImputer(n_neighbors = 5)`

`KI.fit(Train_X)`

`Train_X = pd.DataFrame(KI.transform(Train_X), columns= Train_X.columns)`: 근데, test를 만약에 포함해 만든 모델에 train을 변환시켜서 train을 다시금 추후 모델로 만든다면, 모델을 2번돌리네, 보다 직접적 -> X
`Test_X = pd.DataFrame(KI.transform(Test_X), columns=Test_X.columns)` : train으로 만든 모델로 test를 먹여 test를 채운다는 모델이니까 간접적이다. -> O

> Categorical Variable Process(상태공간-len(S.unique()) 유한한 변수)

> > get_dummies(범주형 상태공간의 크기 적당히 작은 경우)
	- drop_first=True는 차원축소, 과적합방지, 예측력 향상인데, 간혹 설명력을 중시하는 Tree에서는 남기는 경우가 있다.

	1. 말 그대로 범주가 많을 때, 새로운 변수가 수십개 늘어날 수도.
	2. 클래스 불균형-Sparse Data일 때, 차원만 늘리게될 수도. (희소해지게 된다면, 자카드 유사도를 쓰렴)

	- pd.get_dummies()는 학습데이터에서 더미화한 양상과 새로 들어온 데이터에서 더미화하는 양상이 달라질 수도 있기 때문에, 실제 사용할 수는 없다. (ex: 기독교만 들어오는 경우를 들어줬는데 
– 아! OHE 모델을 구축을 하잖아. 그래서 기독교와 대응되는 더미가 모델에 학습되어 있잖아. 근데 get_dummies는 모델 개념이 아닌 메써드지 그러므로, 즉흥적으로 가장 단순한 더미를 만들기 때문에, 데이터 모양이 통일되지 않지.)


`from feature_engine.categorical_encoders import OneHotCategoricalEncoder as OHE`

`dummy_model = OHE(variables = Train_X.columns.tolist(), drop_last=True)`

`dummy_model.fit(Train_X)`

`d_Train_X = dummy_model.transform(Train_X)`
`d_Test_X = dummy_model.transform(Test_X)`


> > Convert Numerical(범주형 상태공간의 크기가 너무 클 때 – 근데 이진형(-1/1)로 쓰면 됨!)

	- 종속변수가 연속형이라면, 데이터 전처리 단계에서, X의 범주를 Groupby.mean() 이런 걸 해서 특정 숫자로 바꿔 넣어준다라는 것인데, 만약에 평균해서 A, B가 200으로 같아질 수도 있지 -> 정보의 손실 단점/ 반면, 차원을 줄일 수 있고, 다양한 범주가 있더라도 상관없으며, (위험) Y의 정보를 통한 X의 치환이므로 모델 성능이 당연히 의미없이 좋아진다.

`for col in [‘cate1’, ‘cate2’]:` #범주 변수만 순회
`temp_dict = Train_df.groupby(col)[‘Class’].mean().to_dict()`
`Train_df[col] = Train_df[col].replace(temp_dict)`
`Test_X[col] = Test_X[col].astype(str).replace(temp_dict)`

### Distribution

> Outlier

> > How to define Outlier

	1. IQR Rule – Outlier < Q1-IQR*1.5 & Outlier > Q3+IQR*1.5: 단일변수에 대해서만 고려하기 때문에, (설명-종속) 두 변수 간의 고려 한계, 경우에 따라 너무 많은 이상치 양산할 수도 있다.

	2. Z-score: 시그마값을 이용해서 분포의 비율, 확률로 제거하는 것이다. 사실 이것도 위의 것과 같지.

	- `df.apply(IQR_rule)` : 데이터 프레임에서는 열단위로 apply 
	- Outlier의 비율은 1% 미만이 정상이라고 생각한다.

	3. DBSCAN(밀도 기반 군집화): 여기에 속하지 않는 것을 이상치라고 규정한다. -> 중심점과 경계점(중심점의 이웃)이 아닌 샘플을 이상치(labels_ = -1)라 간주한다. -> eps(반경), min-samples(최소 샘플 포함 수) 이 둘의 조절이 데이터마다 천차만별일 거 아냐.. 쉽지 않다.

	- eps(반경)의 참고를 위해 cdist -> dis_matrix를 이용함.

`cluster_model = DBSCAN(eps = 0.67, min_samples = 3).fit(Train_X)`
`print(sum(cluster_model.labels_ == -1))`

`Train_X = Train_X[cluster_model.labels_ != -1]`

	- quantile(0.1) 하위 10%, quantile(0.9) 상위 10%

> Independent or Nonlinear

> > 그래프로 보는게 이상적 but, 여러 특징들을 다 시도해보고, feature selection

	- K-fold Cross Validation -> Evaluation: cross_val_score
`import sklearn.model_selection import cross_val_score`
`score = cross_val_score(LR(), X, Y, cv = 5, scoring = mean_absolute_error)`

> Non-descript Dist(->Normal_distriubution)

	- skewness : 절대값 기준 1.5, 왜도 > 0 -> 오른쪽 꼬리, 왜도<0 -> 왼쪽 꼬리
`scipy.stats.skew`

`Train_X.skew()`

->Scale diff 줄인다.

	1. np.log10(X-MIN(X)+1):log는 음수에서 정의되지 않으므로, 최소값을 빼서 모두 양수로 변환한다. 그리고 1을 더해 0보다 큰 값으로 변환한다.

	2. Square Root

> Scale Diff

	1. Standard Scaling(이론은 무한대, 보통 -3-3)
	2. Min-Max Scaling(0-1)

 
	- Standard Scaler는 회귀모델에서 표준처럼 사용하는데, 완전히 스케일이 다같아지는건 아니지.. 그래서, Min-Max를 쓰는데, 특히 신경망, knn(거리를 유사도로 쓰므로, 구간을 모두 일치시켜주는 표준화 씀)에서는 표준처럼 여겨지고 있다.

`from sklearn.preprocessing import MinMaxScaler & StandardScaler`
	- inverse_transform: 입력값을 학습을 끝내고 다시 되돌리고 있다면, 이 메써드를 쓴다.

> Multicollinearity

> > LR(선형 회귀-로지스틱), MLP, SVM – 선형식이 모델에 포함되어 있는 경우, 변수간 선형성은 모델의 강건한 파라메터  추정을 어렵게 한다. 즉, 추정할 때마다 파라메타의 변동성으로 인해 결과가 흔들릴 수 있다. (ex: 선형회귀는 잔차의 제곱을 최소화 -> 그때 미분 극소값을 찾는데 행렬의 역행렬이 결과값에 들어가고, 선형성이 역행렬을 존재하지 않도록 할 수 있다. 즉, parameter X ->모델 구축 X)

> > Tree의 경우, 상관성 자체가 예측 성능에 영향을 끼치지는 않지만, 문제는 x1과 x2가 선형성이 높다면, 이 둘 중 임의의로 택하고, 설명이 크게 달라질 수도 있다.

> > 차원이 늘어난다는 문제라는 측면에서는, 축소하는 것이 Best!

	1. VIF(Variance Inflation Factor): 설명 변수 안에서 특정 변수를 라벨로 설정하고, 나머지 변수들로 얼마나 예측 설명할수 있는지(R squared) : 10이상이면, 삭제하는 경향

`from sklearn.linear_model import LinearRegression as LR`
`VIF_dict = dict()`
`for col in Train_X.columns:`
`LR().fit(Train_X.drop([col], axis=1),Train_X[col])`
`r2 = model.score(Train_X.drop([col], axis=1),Train_X[col])`
`VIF = 1/(1-r2)`
`VIF_dict[col] = VIF`

	2. PCA(Principle Component Analysis): n차원에 대해 n개의 주성분이 존재한다. 주성분이란, 고유벡터를 말하는데, 가령 2차원에서는 공분산 기하학적 의미에서 타원형의 장축 방향, 단축 방향 2개의 주성분이 존재한다. 이때, PCscore는 주성분을 차원의 축으로 하는 데이터로 투영, 회전시킨 것이다. 또, 분석 특징 상, 이 둘은 직교 벡터 관계이므로 상관성이 낮고(다중공선성 해소), 그리고 장축 고유 벡터는 모형 설명력이 가장 뛰어나므로, 이것만 채택하는 것으로 차원 축소도 누릴 수 있다.(단, n차원인 경우, n개 보다 작은 상위 m개의 주성분을 채택한다.)

`from sklearn.decomposition import PCA`

`PCA_model = PCA(n_components = 3).fit(Train_X)`
`Train_Z = PCA_model.transform(Train_X)`
`Test_Z = PCA_model.transform(Test_X)`
`PCA.explained_variance_ratio_`

> Class Imbalance

	- 클래스가 불균형한 경우에는, 재현율을 살필 필요가 있다. 재현율은 실제 참인 것 중에서 맞힌 확률인데, 보통 참, 긍정은 우리가 관심을 갖는 대상, 보통 소수의 대상일 경우가 빈번하므로, 클래스 불균형일 수록, 재현율을 살펴야 된다.

 
 

- 해결방법: 소수 클래스의 결정 공간을 넓히는 개념 – 재샘플링(오버 샘플링, 언더 샘플링)을 할 때, 결정 경계에 가까운 소수 샘플링은 늘리고, 결정 경계에서 가까운 다수 클래스를 제거하면, 결정 공간이 늘어나는 효과가 생긴다.
(단, 평가 데이터에 대해서는 절대로 재샘플링을 하면 안돼, 다만, 학습 모델에 대해 소수 데이터를 더 잘 맞추게하려고, 떠먹여준 것뿐! 현실과는 다르다!)

	1. 오버 샘플링 (1:1까지는 아니다. 현실 데이터도 도메인 특성상 치우친 그 이유가 있을 것이므로, 되려 잘 맞추려면 3:1, 4:1 정도에 멈춘다.)
> > SMOTE(Synthetic Minority Over-Sampling Technique)

 

`from imblearn.over_sampling import SMOTE`
`fit_sample(X, y)` 보통은 fit / transform 이건 train과 test 모두 적용해줘야 할 때!/  오버샘플링은 오로지 학습만 하는 거지! 그러니까 구별해줄 필요가 없지. `fit_sample(Train_X, Train_y)`

`o_Train_X, o_Train_Y = SMOTE(k_neighbors= 3, sampling_strategy={1: int(Train_Y.value_counts().iloc[0]/2), -1: Train_Y.value_counts().iloc[0]}).fit_sample(Train_X,Train_Y)`

	- 재현율과 정확도의 적당한 비율로 취하는 전략으로 비율을 정한다. 모두 다 도메인 지식에 근거해야할 것!

2. 언더 샘플링
> > NearMiss
 
	- 소수클래스와 가깝다는 건 최전방 녀석들, 즉 결정경계 근처의 적들을 타파!
`from imblearn.under_sampling import NearMiss`
`version = 2` : 모든 소수 샘플까지의 평균거리를 쓴다. (n_neighbors의 개수를 정하지 않아도 된다.)

`NearMiss(version = 2, sampling_strategy={1 : u_Train_Y.value_counts().iloc[-1], -1: u_Train_Y.value_counts().iloc[-1] * 5}).fit_sample(Train_X, Train_Y)`

> 비용 민감 모델(cost_senstive_model)
 
	- 긍정을 부정으로 잘못 맞추는 것이 더 치명적이다 – 위음성 비용을 위양성 비율 보다 크게 해야 한다라는 말과 같다.
 

	- predict.proba(X) -> 
 

`model = LR(max_iter = 100000).fit(Train_X, Train_Y)`
`probs = model.predict_proba(Test_X)` : 확률 모델에 대한 cut-off-value 조절.
`probs = pd.DataFrame(probs, columns = model.classes_)`
`pred_Y = 2 * (probs.iloc[:,-1] >= cut_off_value) – 1`
`recall = recall_score(Test_Y, pred_Y)`
`accuracy = accuracy_score(Test_Y, pred_Y)`

 
 
 
	- 사전에 각각의 가중치를 넣어줄텐데, 1에 오차페널티를 높이는 것은 1-긍정-더잘맞춰라 이므로!, 보통 클래스 불균형 비율에 따라서 값을 넣어준다.

`model = SVC(class_weight = {1:8, -1:1}).fit(Train_X, Train_Y)`
`pred_Y = model.predict(Test_X)`
`print(recall_score(Test_Y, pred_Y))`
`print(accuracy_score(Test_Y, pred_Y))`

## To Do

	- Irony of Dimension 
