## Done

> Category Variable

	- Countplot : x = category variable/ y= count

`sns.countplot(x='Class', data=df, order=['H','M','L']`

#### sloution of overlap x-index with each other

`plt.xticks(rotation=90)`
`plt.show()`

	- Covert numerical data

`df['Class_value'] = df['Class'].map(dict('H'=1, 'M'=0, 'L'=-1))`

	- Visualization

`gb = df.groupby('Topic').mean()['Class_value'].sort_values()`

`plt.barh(gb.index, gb)`

	- Preprocessing Data : convert category with on-hot(1-0) vector

``X = pd.get_dummies(df.drop([dependent or Multicollinearity var], axis = 1),
		     columns=[independent categorical var],
		     drop_first = True)
  y = df['Class']``

#### Multicollinearity

	- statistical phenomenon in which predictor variables in logistic regression model are highly correlated.
	- Consequently, it makes data incorrect inferences about relationships between explanatory and respose variables.
	- So, Take strategy of popping data 'drop_first' for this solution.

> Training/Evaluation with Logistic, XGBoost

	- Training of Logistic

`from sklearn.linear_model import LogisticRegression`

``model_lr = LogisticRegression(max_iter=10000)
  model_lr.fit(X_train, y_train)``

***when occuring `Iteration Limit Error`, We could manipulate `max_iter num`***

	- Evaluation of Logistic

`from sklearn.metrics import classification_report`

``pred = model_lr.predict(X_test)
  print(classification_report(y_test, pred))``

	- Training of XGBoost

`from xgboost import XGBClassifier`

``model_xgb = XGBClassifier()
  model_xgb.fit(X_train, y_train)``

	- Evaluation of XGBoost

``pred = model_xgb.predict(X_test)
  print(classification_report(y_test, pred))``

	- Deepen Analysis

``fig = plt.figure(figsize=(15, 8))
  plt.bar(X.columns, model_lr.coef_[0, :])
  plt.xticks(rotation=90)
  plt.show()``

***`model_lr.coef_[0]` is matched with `Class[0]` and then columns is features of the rest set by X***

``fig = plt.figure(figsize=(15, 8))
  plt.bar(X.columns, model_xgb.feature_importances_)
  plt.xticks(rotation=90)
  plt.show()``


## To Do

	- Victory Fomula of LOL by EDA


