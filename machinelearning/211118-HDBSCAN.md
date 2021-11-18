## Done

### HDBSCAN

> Different sizes, densities, noise, arbitrary shapes인 데이터에 적합하다.

> Hierarchical 계층적 구조를 반영한 Clustering이 가능하다.

	- numpy stack

`hdb_data = np.vstack([moon, blobs1, blobs2, blobs3])`
`pd.DataFrame(hdb_data, columns=[‘x’,’y’])`

	- plotly

`import plotly.express as px`
`fig = px.scatter(hdb_data_df, x=”x”, y=”y”)`
`fig.update_layout(width = 600, height=500, title=’hdb_data_distribution’)`
`fig.show()`

	- HDBSCAN

`import hdbscan`
`hdbscan_model = hdbscan.HDBSCAN(min_cluster_size = 5, min_samples=None, cluster_selection_epsilon=0.1)`
`hdbscan_model.fit(hdb_data)`
`hdbscan_label = hdbscan_model.fit_predict(hdb_scan)`

`hdb_scan_df[‘label’] = hdbscan_label`
`hdb_scan_df[‘label’] = hdb_scan_df[‘label’].astype(str)`
`fig = px.scatter(hdb_scan_df, x=”x”, y=”y”, color=”hdbscan_label”)`
`fig.update_layout(width = 600, height=500, title=’hdb_data_distribution’)`
`fig.show()`

	- min_cluster_size = 5, min_samples=None, cluster_selection_epsilon=0.1 각각 커지면 커질수록 하나의 군집을 크게 잡게 되고, 세세한 분류 과적합을 방지하고 규제화된 양상으로 나아간다.

## To Do

	- HDBSCAN
