## Done

### DBSCAN

> Non-spherical

`from sklearn.cluster import DBSCAN`
`dbscan = DBSCAN(eps = 0.2, min_samples=6)`
`dbscan.fit(moon_data)`
`dbscan_label = dbscan.labels_`
`set(dbscan_label) = {-1, 0, 1}`: -1 is Outlier

	1. Core point: 클래스를 구분할 수 있는 중심 점 : 주변 eps으로 돌렸을 때 min_samples의 개수를 충족하는 점
	2. Border point: 클래스를 구분하지는 못하되, 주변 중심점에 편입되는 점 : eps로 돌렸을 때 min_samples의 개수를 충족하지 못하는 점.
	3. Outlier point : eps로 돌렸을 때, 아무것도 들어오지 않은 점

> > label comparison

`compare_kmeans_clusters = dict(zip(kmeans.labels_, moon_labels))`
` compare_kmeans_clusters = {0: 0, 1: 1}`

> > Silhouette score

	- 군집내 분산 분에 군집 간 거리이므로, 구형 데이터가 아닌 경우에는 제대로 작동하지 않는다. 

> > ARI – Adjusted rand index (실제 label과 예측 label의 유사도 비교)

`from sklean.metrics import adjusted_rand_score`

`dbscan_ari = adjusted_rand_score(moon_labels, dbscan_new_labels)`
`print(round(dbscan_ari, 4))`

> > find_match_cluster btw model_label with actual_labels

	- 그전에, 빈도에 따라 new_label을 매칭해줄 때, 개수가 많은 것부터로 변환해서 해주기!, 클래스 개수 차이가 나게 되면, 적은 개수의 데이터에서 최빈 값을 잘못 뽑을 수 있으니까. 큰 개수에서부터 하나하나 정확하게 뽑도록 하자. dbscan의 경우에는 이상치가 들어가므로, 이상치는 None으로 newlabel로 했다면, 다시 변환할 때 -1 그대로 감안해서 바꿔주는 거

`def find_matching_cluster(cluster_case, actual_labels, cluster_labels):`

## To Do

	- HDBSCAN
