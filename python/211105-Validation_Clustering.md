## Done

### Linear

> 독립성, 정규성, 등분산성 충족시
`scipy.stats.kstest(x,’norm)`
- ANOVA(일원분산분석)

	- 셋 이상의 그룹 간 차이가 존재하는지 확인_가설 검증 방법
	- 그렇다면, 독립 표본을 여러 번 시행하면 되지 않느냐?
	- 그렇지 않다. 설령, pvalue>0.05이어서, 영가설을 기각하지 못할 때에도, 여러 그룹을 시행하며 pvalue가 작아져서, 잘못 기각하는 오류가(1종 오류)가 발생하게 될 수 있다.

	- F=(집단 간 분산)/(집단 내 분산) : 즉 집단 간 중심의 차이가 클수록, 집단이 정밀하게 모여있을수록, 집단간 차이가 유의하다고 판단할 수 있다.
`scipy.stats.f_oneway(sample1, 2, 3...)`

	- P-value < 0.05 -> 영가설을 기각했을 때 -> 어떤 그룹간 차이가 유의해서, 대립가설을 채택했는지 ‘사후분석’: Tukey HSD(honestly significant difference) = 두그룹의 차(Max-Min)/SE(두 그룹의 표준편차) > 유의수준 : 두 그룹이 차이가 났다고 판단.
`statsmodels.stats.multicomp.pairwise_tureyhsd(Data, Group)`: 독립 표본 t-검정과 비슷하게 출력됨.

`Group = [‘A’] * len(A) + [‘B’] * len(B) + [‘C’]*len(C)`
`Data = A.tolist() + B.tolist() + C.tolist()`

`reject = False(차이X)/ reject=True(차이 O)`

> 정규성 또는 등분산성 중 하나라도 미충족시(독립 표본 t-검정의 경우 equal_var = False로 하고 계산한 것과는 구분됨)

- Kruskal-Wallis H Test(비모수 방법)

> 상관관계(연속형 변수간)과 카이제곱검정(범주형 변수간)

> > 상관관계
	1. 피어슨 상관계수: 선형으로 증가하는지 파악, 머신러닝에서는 x를 알 때 y를 정확히 예측할 수 있는 것이 선형이라 해석한다.(-1~1)
`scipy.stats.pearsonr(x,y)`

	2. 스피어만 상관 계수: 단조 증감 관련성 파악 주력
	- x, y를 순위를 매겨 추세를 파악한다.
`scipy.stats.spearmanr(x,y)`

`피어슨: 0.7, 스피어만: 1이라면, 선형은 아니지만, 단조 증가다.`
`DataFrame.corr(method = pearsonr(default) or spearmanr)`

	- plt.xticks() label 건너 뛰기, 간추리기
`plt.xticks(df[‘일자’].iloc[::8])`:Series가 들어간 경우, index가 상대 위치, values가 label이 된다.

	- sns.pairplot() = pd.plotting.scatter_matrix()

	- itertools.combinations(target_cols, 2)
`target_cols = [‘금값’, ‘은값’, ‘달러 환율’]`

> > 카이제곱검정
	- H0: 두 변수가 독립, H1: 두 변수가 종속

	- Contigency Table(교차 테이블): 실제값과 기대값(두 변수가 독립일 때 값)
`pandas.crosstable(S1,S2)`
	- 상황에 따라 pivot_table을 쓸 수 있다. 상황에 맞게
	- 카이제곱 검정 통계량(독립인지 아닌지만 본다.): (실제값 – 기대값)^2/기대값 -> 클수록, 영가설 기각, pvalue 작아짐
`statistics, pvalue, dof, expected =scipy.stats.chi2_contigency(obs = pd.crosstable(S1,S2).values)`

> 군집화(비지도학습)

> > 계층적 군집화(Hierarchical Clustering)
	- 유사도_거리_수치변수들_범주형 변수를 더미화(if modeling, drop_first: 추론 가능, 상관성 제거, 계산량 감소, 차원 축소, 과적합 방지)

	1. euclidean 거리
	2. Manhattan 거리: 정수형 (리커트 척도 데이터_ex) 설문조사 결과)
	3. Cosine 유사도: 크기가 아닌 방향만 볼 때(상품 추천 시스템)(여러 변수 간 스케일 차이가 있어 방향성 차이만 보려고 할 때, K-means(euclidean)이므로 계층적 군집화 사용)
--이진형 데이터(0-1)
	4. 매칭 유사도: 전체 특징 중 일치하는 비율
	5. 자카드 유사도: 1을 가지는 특징 중 일치하는 비율(희소한 이진형 데이터에 적합) – ex: 페라리 소유: 1 1로 일치하는 것은 의미가 있어, 희소하니까! 근데, 0으로 일치하는 것은 무수한 변수들이므로 일치하더라도 별 의미가 없네

- 군집간거리
	- 1/2/3.최단/최장/평균 연결법: 군집간 모든 값들끼리 거리를 구해 계산량이 많고, 이상치에 영향을 받음
	- 4.중심 연결법: 이상치 둔감, 계산량 적음 – 선호
	- 5.와드 연결법: 한쪽만 유독 큰 군집이라면, 왜곡된 거리가 나올 가능성이 크다. 즉, 군집내 거리도 고려하여 크기가 비슷한 군집에 대해 거리를 계산하는 방법(default), 이상치에 둔감하지만 계산량이 많다.

- 평가
	- 1. 장점: 덴드로그램으로 군집화 과정 확인, 거리/ 유사도 행렬로 군집화 가능, 다양한 거리 척도 활용 가능(K-means는 euclidean만 가능), 시행마다 같은 결과 출력
	- 2. 단점: 계산량이 많고, 개수 설정에 제약이 존재한다.(거리가 같은 경우, 특정 군집 개수에 대한 출력이 공백인 경우도 존재하게 된다.)

`sklearn.cluster.AgglomerativeClustering`
`n_clusters`
`affinity:{‘Euclidean’, ‘manhattan’, ‘cosine’, ‘jaccard’,‘precomputed’_거리, 유사도 행렬을 입력}`
`linkage:{‘ward’,’complete’,’average’,’single’}`
	- ‘ward’일 때는, ‘Euclidean’을 사용해야만 한다.

- practice_1
`from sklearn.cluster import AgglomerativeClustering`
`clusters = AgglomerativeClustering(n_clusters = 3, affinity = ‘euclidean’, linkage=’ward’).fit(df)`

`df[‘result’] = clusters.labels_`
	- 이진변수(0-1)의 평균은 1의 비율!

- pratice_2
`clustering = AC(n_clusters = 10, affinity = ‘jaccard’, linkage = ‘average’)` : jaccard 이진형(0-1) 데이터에 대해서만 쓸 수 있다. 

`matrix_df_cluster_info.groupby([‘소속군집’])[matrix_df.columns].means().idxmax(axis=1)` :인덱스 별 열 방향으로 최대값을 갖는 column을 반환

> > K-means Clustering

	- 샘플의 영향을 주지 않고, 임의로 중심점을 k개 뿌려서, 군집화 및 중심점 업데이트 반복

	- 장단점
	1. 장점: 적은 계산량, 군집 개수 설정에 제약이 없고 쉽다.
	2. 단점: 초기 중심점에 따라, 군집 결과가 달라질 수 있음.(임의성 존재), 분포가 복잡하거나 특이하거나, 군집별 밀도 차이가 극명하다면 좋은 성능을 내기 어렵다. 유클리디안 거리만 사용해야 한다.(중심점), 중심점 업데이트 과정의 수렴의 문제가 있을 수 있다.(-> Max_iter = 1000, 반복횟수 제한)

`from sklearn.cluster import KMeans`
`clusters = KMeans(n_clusters =3, max_iter=10000).fit(df)`

`pd.DataFrame(clusters.cluster_centers_,
columns = df.columns, index = range(3))`

> 연관규칙: 거래, 로그 데이터 추천 시스템 분석 기법

	- 지지도, 신뢰도(조건부 확률 개념)

	- Apriori 원리: 최소 지지도(min support)/최소 신뢰도(min Confidence)_사용자가 설정 보다 크면, ‘빈발’하다고 한다. 이 때, 어떤 아이템 집합이 빈발이면, 이 아이템의 부분 집합도 빈발하다. X가 Y의 부분 집합이면, X는 n(Y) + a 이므로 그렇습니다. 반대로 아이템 집합이 빈발하지 않다면, 부모의 집합도 빈발하지 않다. 

	- 후보 규칙 생성(Apriori 원리 이용)
	1. 최대 빈발 아이템 찾기 : 최대 집합 개수 중, 최소 지지도 이상이면서, 이것의 모든 부모집단이 빈발하지 않는 집합. 
	2. 최대 빈발 아이템을 갖고, 자식 집단을 또 부모와 자식으로 경우의 수 테이블을 만들어 다양한 후보 규칙들을 파악한다.

	- 신뢰도의 Apriori 원리 : 동일한 아이템 집합일 때, X1 이 X2의 부분 집합일 때, C(X1 -> Y1) <= C(X2 -> Y2) : 부모집단의 부분집합 순으로 신뢰도를 도표로 그렸을 때, 한 노드에서 최소 신뢰도 이하라면, 그 부분집합의 신뢰도도 그 이하이므로 탐색할 필요가 없다.

	- mlxtend : 빈발 아이템 집합 탐색, 연관 규칙 탐색

`mlxtend.frequent_patterns.apriori(one_hot_df, min_support):
	- one-hot encoding: mlxtend.preprocessing.TransactionEncoder
	- 대부분의 결과가 ndarray형태로 출력되는데, 인덱스가 없으므로 다루기가 어렵다. 그래서 항상 df으로 변환해주는 것까지!

`mlxtend.frequent_patterns.association_rules(frequent_dataset, metrics = ‘confidence’, min_threshold = min_Confidence)` :신뢰도의 Apriori 원리를 이용

	- 평가:
result[[‘antecedents’, ‘consequents’, ‘support’, ‘confidence’]] : 이때, 부모일 때 자식이 전체 중 support 만큼 등장하였고, 그리고 부모일 때 자식일 가능성이 약 20% 정도 되는 구나, 그 정도 연관 규칙성이 있구나!

> > 시퀀스 데이터 빈발 **순서** 패턴

- Sequence : 순서 데이터, 로그 데이터(ex: 고객 구매 기록, 고객 여정, 웹 서핑 기록)

	- 출현 횟수를 계산 방식: 윈도우 크기(L) _hyper - 사이 간격을 두고 순서대로 출현한 것도 같이 계산할 건지? 

	- 최대 지지도 선택: 순서를 맞출 때만 선택! (규칙 X)
	- 최소 신뢰도 제거: 부분집합이면 신뢰도가 작으니까 제거하는 것은 문제가 없다.(규칙 O) : 3개 이상의 순서가 나타나는 경우 부모 2-1꼴 만들어서 가장 큰 집합의 부모 노드에 대한 최소 신뢰도를 본다. 근데, 최소 신뢰도 보다 작다면, 당연히 그보다 더 작을 부분 부모 노드 1-2꼴은 볼 필요가 없겠다.

> > 시계열 데이터 

	- 시계열은 연속형 인덱스를 갖고 있는 시퀀스 데이터다. 즉, 엄밀히 의미만 따지면, 시계열 데이터는 시퀀스 데이터에 속한다.

	- 연속형 인덱스는 패턴 찾기가 어려우므로, 범주화, 이산화한다. SAX(Symbolic Aggregate Approximation) (1) 윈도우 분할, (2) 윈도우별 대표값 계산, (3) 알파벳 시퀀스로 변환 (ex: 보통 -> 큼 -> 보통 -> 작음 -> 보통)

> > 활용

	1. 협업적 추천 방식에 활용(시청기록이 유사한 사용자의 차집합 정보를 추천해준다.)

	2. 특징 요소별로 군집화, 비지도학습 가능

> > A/B 테스트 : 임의로 나눈 두 집단에 대해 서로 다른 콘텐츠를 제시한 뒤 통계적 가설 검증을 통해 어느 집단이 효과적인지 판단.

	- 통계분석시 검정을 통해 통계량만을 보고 판단하는 것은 자칫 기계적으로 판단한 우려가 있다. 따라서 항상 시각화도 함께 하여, 실제 데이터의 양상이 어떤지 살펴보는 것이 중요하다.

	- 차집합 인덱싱

`df.loc[~df[‘고객ID’].isin(churn_ID), ‘고객ID’]`

## To Do

	- Machinelearning
