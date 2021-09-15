## Done

	- LOL-EDA : League of Legends

Data Resource in Kaggle Link is [Resource](https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min)

### EDA or Basic Statistic

#### Visualize columns with Pearson's Correlation

	sns.heatmap(df.corr()[['blueWins']], annot = True)
	* annot = annotation(represent numerical data)
	- select columns have high correlation with 'blueWins

#### Show Plot about Selected Columns

	sns.histplot(x='blueGoldDiff', data=df, hue='blueWins', palette='RdBu', kde = True)

	sns.histplot(x='blueKills', data=df, hue'blueWins', palette='RdBu', kde=True, bin=8)
	* bin is how much assign in a unit.

### Preprocessing Data

#### Stardardization using StandardScaler

	- drop a column of multicollinearity pair

	- Standardize numerical data

	scaler = StandardScaler()
	scaler.fit(X_num)
	X_scaled = scaler.transform(X_num)
	X_scaled = pd.DataFrame(X_scaled, index=X_num.index, columns = X_num.columns)

	X = pd.concat([X_scaled, X_cat], axis = 1)

### Modeling, Training

	from sklearn.linear_model import LogisticRegression

	from xgboost import XGBClassifier

### Evaluation

	pred = model_xgb.predict(X_test)
	print(classification_report(y_test, pred))
	* Check precision value through xgb model
	

	model_coef = pd.DataFrame(data=model_lr.coef_[0], index=X.columns, columns=['Model Coefficient'])
	* Convert built in correlation array with dataframe

	model_coef.sort_values(by='Model Coefficient', ascending = False, inplace = True)
	* Can sort values in dataframe format

	plt.bar(model_coef.index, model_coef['Model Coefficient']0
	plt.xticks(rotation=90)
	pli.grid()
	plt.show()


## To Do

	- Chap 4. European Soccer Database EDA
