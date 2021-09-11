## Done

### Research Data

- Well-formed Data - Tabular

- it's Essential to cite it

### Data Resource

- name: Student's Academic Performance Dataset

- link: [Student's Academic Performance Dataset](https://www.kaggle.com/aljarah/xAPI-Edu-Data)

- citation:

	- Amrieh, E. A., Hamtini, T., & Aljarah, I. (2016). Mining Educational Data to Predict Student’s academic Performance using Ensemble Methods. International Journal of Database Theory and Application, 9(8), 119-136.

	- Amrieh, E. A., Hamtini, T., & Aljarah, I. (2015, November). Preprocessing and analyzing educational data set using X-API for improving student's performance. In Applied Electrical Engineering and Computing Technologies (AEECT), 2015 IEEE Jordan Conference on (pp. 1-5). IEEE.

### EDA or Basic Static Analysis

- Columns

`df.head(), df.info(), df.describe()`

	- df.head(): Understand each meanings

![head](TIL/img/210911/head.png)

	- df.info(): non-null & int64, object

![info](TIL/img/210911/info.png)

	- df.describe(): well-uniformed distribution

![describe](TIL/img/210911/describe.PNG)
			
	- \*``rename : `df_rnm = df.rename(columns={'NationalIty' : 'nationality'})``
			

- histogram

	sns.histplot(x='raisedhands', data=df, hue='Class', hue_order=['H', 'M', 'L'], kde=True)

![raisedhands](TIL/img/210911/raisedhands.png)

`sns.histplot(x='VisITedResources', data=df, hue='Class', hue_order=['H', 'M', 'L'], kde=True)`

![visitedsource](TIL/img/210911/visitedsource.png)

`sns.jointplot(x='VisITedResources', y='raisedhands', data=df, hue='Class', hue_order=['H','M','L'])`

![jointplot](TIL/img/210911/jointplot.png)

`sns.pairplot(data=df, hue='Class', hue_order=['H','M','L'])`

![pairplot](TIL/img/210911/pairplot.png)

## To Do

- The rest of EDA
