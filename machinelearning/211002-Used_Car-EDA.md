## Done

### EDA or Basic Static Analysis

	- Check the columns and the values, In context of data theme, we distinguish what data is important or how much is it involved in Nan. Furthermore, It can be determined whether categorical or numerical type.
`df.head()`

	- Check the number of null values
`df.info()`
`df.isna().sum()`

	- Check the uniform status of data
`df.describe()`
`len(df.value_counts())`

### Categorical Data Analysis

	- According to category of data, we try to plot countplot.
`sns.countplot(y=’manufacturer’, data=df.fillna(‘n/a’), order=df.fillna(‘n/a’)[‘manufacturer’].value_counts().index)`

	- `value.counts()` print type of list, so if the size is too big that it can’t show all, we try this code.
`for index, num in zip(df[‘model’].value_counts().index, df[‘model’].value_counts()):
print(index, num)

	- when data has so many categories of X, we can convert X with Y by following code.
`sns.countplot(y=’type’, data=df.fillna(‘n/a’), order=df.fillna(‘n/a’)[‘type’].value_counts().index)`

### Numerical Data Analysis

	- if histplot is not useful, then choose the rugplot.
`sns.rugplot(x=’price’, data=df, height=1)`
`sns.histplot(x=’age’, data=df, bins=18, kde=True)`

## To Do

	- Data Cleaning
