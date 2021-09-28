## Done

> Geo-graphy of Visualization

> > import

`!pip install geopandas`
`!pip install pyshp`
`!pip install shapely`
`!pip install plotly-geo`

> > Choropleth of Plotly

	: FIPS data format : country + state code

``fig = ff.create_choropleth(
    		fips=fips, values=data,
    		show_state_data=False,
    		colorscale=colorscale,
    		binning_endpoints=list(np.linspace(0.0, 1.0, len(colorscale) - 2)), 
    		show_hover=True, centroid_marker={'opacity': 0},
   		asp=2.9, title = 'USA by Voting for DEM President
)``

> PCA (Principal Component Analysis)

> > Unsupervised Learning & Dimensional Reduction

``pca = PCA()
pca.fit(X_train)
plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
plt.grid()``

> > Scree plot: pca.explained-variance is EigenValue that explain how much is it reflected with fluctuation of model : Select how many variables do we


> Modeling

> > LightGBM Regression : Logistic light version, fast & nice perfomance

``from lightgbm import LGBMRegression

model_reg = LGBMRegressor()
model_reg.fit(X_train, y_train)``

> Evaluation

> > mean-absolute-error or mean-squared-error

``from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report
from math import sqrt``

``pred = model_reg.predict(X_test)
print(mean_absolute_error(y_test, pred))
print(sqrt(mean_squared_error(y_test, pred))
print(classification_report(y > 0.5, pred > 0.5))``


## To Do

	- Chap 2. Regression Analysis
