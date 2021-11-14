## Done

### K-means VS Agglomerative

> K-means

> > 최적의 환경

	1. 원형 혹은 구 형태의 분포
	2. 동일한 데이터 분포
	3. 동일한 밀집도
	4. 군집의 중심점에 주로 밀집된 분포
	5. Noise와 Outlier가 적은 분포

> > 민감성

	1. Noise와 Outlier에 민감 -> 중심점도 결국 평균의 개념과 유사하기 때문이다.
	2. 중심의 초기값, 초기 위치에 따라 결과 영향
	3. k값, 중심점의 개수, 군집의 개수 사전 설정에 어려움.

> Agglomerative Clustering

> > 차별점
	1. 군집의 데이터 분포가 동일하지 않더라도, 극복할 수 있다. -> ward linkage 군집간 거리는 군집 간 분포, 크기가 다르더라도, 군집 내 거리를 고려하여 군집 간의 거리를 측정해주는 방법으로, KMeans를 보완할 수 있다.
	2. Noise와 Outlier에 덜 민감하다. -> Ward linkage가 군집 내 거리를 고려하므로, 이상치에 덜 민감하다.
	3. 유사도를 측정하는 거리의 선택폭이 넓다. -> Kmeans(euclidean)뿐이나, 계층 군집화는 리커드 척도에서 Manhattan 거리 혹은 스케일의 차이는 무시하고 방향의 유사도를 볼 때 Cosine 유사도를 쓸 수 있다. 그밖에도 이진형 종속 변수일 때, 매칭 유사도 혹은 자카드 유사도도 고려해볼 수 있다.
	4. 군집화 과정을 볼 수 있다. 즉, 설명력이 좋다. 반면, KMeans는 사후 해석일 뿐, 또 초기값과 k값에 영향을 많이 받는다.

> > 의미

	1. 사실 실무에서는 많이 쓰이지 않는다. 전체 군집을 확인하는 것에 쓰이므로 대용량 데이터에 대해 계산량이 많아 비효율적이기 때문이다. 
	2. 그러나 간편한 측면이 있다. k개수를 정하지 않아도 되며, random point, 랜덤 초기값이 아니므로 항상 동일한 결과가 나오기 때문에 안정성이 있다. 
	3. 의의는 뒤에 배울 HDBSCAN(계층 군집화 + 밀도 기반)의 기초지식으로서 가치가 있다.
`import matplotlib.pyplot as plt`
`from scipy.cluster.hierarchy import dendrogram, linkage`
`def create_linkage(model)` : 거리행렬로 바꿔주는 함수를 정의
`aggl_dend = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(train_X)` : 전체 군집을 모두 확인하겠다.
`linkage_matrix = creat_linkage(aggl_dend)`
`dendrogram(linkage_matrix, truncate_mode = ‘level’, p =3)` :  p값은 시행착오를 통해 조정해야하는 parameter이다.
`plt.show()`

## To Do

	- DBSCAN
