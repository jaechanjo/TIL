## Done

### Dimensional Curse

	- Sklearn.feature_selection.SelectKBest

	1. score_func: chi2, mutual_info_classif, f_regression

	2. k:선택한 특징 개수

	3. get_support(): True or False 선택 여부에 따른 Boolean list – fancy indexing

	4. pvalues_:scoring_func으로 측정한 특징별 p-value (0에 가까우면 x에 따라 y의 클래스 간 집단 평균의 차이가 유의미하다. 즉 라벨 설명력이 높다. 독립이 아니라)

`from sklearn.feature_selection import *`
`selector = SelectKBest(score_func = f_classif, k = 30)
`selector.fit(X_train, y_train)`
`selected_features = `X_train.columns[selector.get_support()]`

	- 상태공간의 개수로 연속, 범주형 구분

`continous_cols = [col for col in X_train.columns if (X_train[col].unique()) > 3]`
`binary_cols = [col for col in X_train.columns if (X_train[col].unique()) <= 3]`

	- 라벨 설명력 분석 – 연속형(f-classf)/ 범주형(chi2)

`continous_cols_pvals = f_classif(X_train[continous_cols], y_Train)[1]`
`binary_cols_pvals = chi2(X_train[binary_cols], y_Train)[1]` :(statistics, p-value)이므로 1번째 인덱스 값만 – p-value만


### Iris Clustering

> K-means clustering

> > km.inertia_ : distortion, 각 군집의 중심점으로부터 샘플들의 거리의 제곱의 합(분산) -> 즉, 작을 수록 중심에 오밀조밀하게 모여 있는 예쁜 군집화가 된다. 

	- plotly

`import plotly.express as px`
`fig = px.line(x=k_range, y=distortion, labels={“x”:”k”, “y”:”distortions”})`
`fig.update_layout(width = 800, height = 500)
`fig.show()`

> How to tuning k(hyper_parameter)

> > KElbowVisualizer

`from yellowbrick.cluster import KElbowVisualizer`
`km = KMeans()`
`visualizer = KElbowVisualizer(km, k=(1,11))`
`visualizer.fit(X_train)`
`visualizer.poof()`

> > kneed(detect k by knees or elbows)

`from kneed import KneeLocator`
`kneedle = KneeLocator(x=k_range, y=distortions, curve=”convex”, direction=”decreasing”)`
`kneedle.elbow` : .knee
`kneedle.elbow_y`: .knee_y

`kneedle.plot_elbow()` : plot_knee()

> > Silhouette Method

	- Within cluster, distance sum of square_군집 내 거리, 분산 + ‘군집 간의 거리’(n_cluster >= 2) -> 군집 간 거리가 멀수록, 군집 내의 거리는 작을 수록 실루엣 계수가 커진다. 성능이 좋다.

	- (-1~1), 0일 때는 군집 간 변별력이 없다. -1은 군집화 결과가 좋지 않다. (사실 이 둘의 차이점이 말로만 들어서는 애매해)

`from sklearn.metrics import silhouette_score`

`silhouette_scores=[]`
`k_range = range(2,11)`
`for i in k_range:`
`print(i)`
`km=KMeans(n_clusters=i)`
`km.fit(X_train)`
`label=km.predict(X_train)` : X_test는 최종 성능 평가할 때 하는 거고, 지금은 train 안에서 label 뽑고 몇 개의 중심점, hyperparameter를 정하면 좋을지 정하는 것, 즉 validation data라고 보는 것이 혼란을 방지할 수 있겠다.
`sc_value = sihouette_score(np.array(X_train), label, metric= “euclidean”, sample_size=None, random_state=None)`
`silhouette_score.append(sc_value)`

> > SilhouetteVisualizer

`from yellowbrick.cluster import SilhouetteVisualizer
`k_range = range(2,10+1)`
`for i in k_range:`
`print(i)`
`km=KMeans(n_clusters=i)`
`visualizer = SilhouetteVisualizer(km)`
`visualizer.fit(X_train)`
`visualizer.poof()`

	- 절대적 실루엣 계수의 값이 큰지 + 면적이 데이터 양인데, 면적을 균등하게 분배하고 있나

> Evaluation

	- 시각화로 비교해보니 얼추 비슷한데, acc 값은 0.24??!?!

	- 그 이유는, 군집의 분류 명이 매칭되지 않기 때문이다.

`import scipy`

`def find_matching_cluster(cluster_case, actual_labels, cluster_labels):`
`matched_cluster={}`
`actual_case = list(set(actual_labels))`
`for i in cluster_case:`
`idx = cluster_labels == i`
`new_label = scipy.stats.mode(actual_labels[idx])[0][0]`
`matched_cluster[i] = new_label`
`return matched_cluster

	- Re_assign

`train_new_labels = [train_matched_cluster[label] for label in cluster_lables]`

## To Do

	- Rewind & Modeling
