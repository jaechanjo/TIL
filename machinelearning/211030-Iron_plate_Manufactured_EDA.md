## Done

### Tips of EDA 

- Make boolean Series list

	1. Logical Operator &
`(df[‘Pastry’] == 1) & (df[‘Z_Scratch’] == 0)`

	2. pandas.Series.astype
`df[‘Pastry’].astype(bool)`

	3. map, lambda, astype()
`condition_bf = [df[‘Pastry’], df[‘Z_scratch’]]`
`list(map(lambda s: s.astype(bool), condition_bf))`

- Categorize lots of Variables

	- np.select, choices
`choices = [‘Pastry’, ‘Z_scratch’]`
`df[‘class’] = np.select(conditions, choices)`

- Pair Plot – Visualizaion with Numerical Variables
`color_code = {‘Pastry’:’Red’, ‘Z_Scratch’ : ‘Blue’}`

`color_list = [color_code.get(i) for i in df[‘class’]]`

`pd.plotting.scatter_matrix(df.loc[:, df.columns != [[‘Pastry’, ‘Z_Scratch’, ‘Class’ ]]], c=color_list, figsize=[30,30], alpha=0.3, s=50, diagonal=’hist’)`

- hist or dist or countplot with Categoric Datas

`g = sns.factorplot(x=’class’, data=df, kind=’count’, palette=’YlGnBu’, size=6)`
`g.ax.xaxis.set_label_text(‘Type of Defect’)`

	- annotation

`for p in g.ax.patches:
`g.ax.annotate((p.get_height()), (p.get_x() + 0.25, p.get_height() + 10))`

- Correlation with Variables in Heatmap

`corr = df.corr()`

`mask = np.zeros_like(corr, dtype(bool))`
`mask[np.triu_indices_from(mask)] = True`

`f, ax = plt.subplots(figsize=(11,9))`
`cmap=sns.diverging_palette(1,200, as_cmap=True)`

`sns.heatmap(corr, mask=mask,cmap=cmap, vmax=1,vmin=-1,center=0, linewidth=2)`

- Train_test_split

	- It’s sure that X_test is also standardized, because of train scale

`from scipy.stats import zscore`
`X_test = X_test.apply(zscore)`

## To Do

	- Iron_Plate_Manufactured_Modeling
