## Done

> 2020 America Election & 2017 Census Data

>	* [2020 America Election](https://www.kaggle.com/unanimad/us-election-2020)
>	* [2017 America Census](https://www.kaggle.com/muonneutrino/us-census-demographic-data)

> > find new insight concatenating both data

>	* Because of a lot of states, it's nice to consider convertion of state-code
>	[State Code](https://www.infoplease.com/us/postal-information/state-abbreviations-and-state-postal-codes)

### EDA

> Value Masking Strategy

> > In 'party' column, we extract only about two values : ['DEM', 'REP']
	
	df_pres.loc[df_pres['party'].apply(lambda s : str(s) in ['DEM', 'REP'])]

> Column Renaming

	table_pres.rename({'DEM': 'Pres_DEM', 'REP': 'Pres_REP'}, axis = 1, inplace = True)

> Exception of Multicollinearity

> > For example, there is always argumentary on sex data processing about mulicollinearity. because men and women have relationship of reverse.

> Operation with all of values in columns

	df_census['Men'] /= df_census['TotalPop']

> > New copy ver

	df_norm = df.copy()
	df_norm['Pres_DEM'] /= df['Pres_DEM'] + df["Pres_REP']

> Correlation Visualization with Heatmap in Seaborn

	plt.figure(figsize=(5,12))
	sns.heatmap(df.corr()[['Pres_DEM', 'Pres_REP']], annot = True]])


## To Do

	- Visualization with 'Plotly' tool

	- Data Preprocessing for modeling train

	- LightGBM model & XGBoost 

	- Evaluation
