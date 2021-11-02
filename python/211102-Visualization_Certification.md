## Done

### Data Visualization

- Preferences

`plt.rcParams[‘font.family’] = ‘Malgun Gothic’` : Korea doesn’t applied on Python
`ptt.rcParams[‘font.size’] = 20`
`plt.rcParams[‘figure.figsize’] = (40,15)`

- DataFrame plot

`df.plot(kind=’line’, x=’날짜’, y=[‘상품1’, ‘상품2’, ‘상품3’])`: automatic legend
=
`plt.plot(df[‘날짜’], df[‘상품1’])`
`plt.plot(df[‘날짜’], df[‘상품2’])`
`plt.plot(df[‘날짜’], df[‘상품2’])`

- xticks, yticks

`plt.xticks([1,2,3,4,5],[a,b,c,d,e])`
->
`xtick_range = np.cumsum([0 31 28 31 30 31 30 31 31 30 31 30])`
`plt.xticks(xticks_range, df[‘날짜’].loc[xticks_range])`

- groupby.plot

`df[‘월’] = df[‘날짜’].str.split(‘-‘, expand=True).iloc[:,1]`

`df.groupby(by=’월’)[[‘상품1’,’상품2’,’상품3’]].sum().plot(kind=’line’)
 
`plt.xticks(range(12), [str(i+1), ‘월’ for i in range(12)])`

	- Muliti Index - df.groupby(as_index = False) : convenient

`df.groupby([‘분기’, ‘대리점’], as_index = False)[‘수량’].sum()`

- pivot_df.add_suffix(‘접미사’) : columns name + suffix

- if x is str category, it can be ordered randomly

`df.loc[‘분기’] = list(range(len(pivot_df)))`

`pivot_df.plot(kind=’scatter’, x=’분기’, y=’대리점_1_출하량’)`

- Mulitiple Bar Plot – Mulit Index – groupby, set_index ->(unstack()/stack())<- pivot_table

`df.groupby([‘제품군’, ‘년도(year)’], as_index=True)[‘수량’].sum().unstack().plot(kind = ‘bar’)`

- pyplot.pie()

`plt.pie(x=grouped_df[‘수량’], labels = grouped_df[‘제품군’], labeldistance = 0.3, autopct=’%1.1f%%’)`: labeldistance is how label is far from center of circle

- pyplot.boxplot()

`df.groupby([‘쇼핑몰 유형’])[‘판매 금액’].apply(list)`: For distribution, we accept all of datas

`df.boxplot(columns = [‘실험1’, ‘실험2’, ‘실험3’]`

- Series.to_dict() -> Series.replace(dict)

`plt.pcolor(pivot_df, edgecolors = ‘black’, cmap=plt.cm.Reds)`
`plt.colorbar()

- How to show Top 10 

`threshold1 = grouped_df_by_customer[‘sum’].quantile(0.9)`
`threshold2 = grouped_df_by_customer[‘count’].quantile(0.9)`

`cond1 = grouped_df_by_customer[‘sum’] >= threshold1`
`cond2 = grouped_df_by_customer[‘count’] >= threshold2`

`grouped_df_by_customer.loc[cond1 & cond2].sort_values(by=[‘sum’, ‘count’], ascending= False).head(10)`

- Let’s see the Barplot Trend clearly

`plt.ylim([min(group_df[‘count’]) * 0.9, max(group_df[‘count’]) * 1.1])`

- Text Mining – Important Keyword Top 10
`wordlist = df[‘기사제목’].str.split(‘ ‘, expand = False).sum()` : sum() is connet all of lists

`wordlist = pd.Series(word_list)`
`wordlist.value_counts()[:10]` : Series.value_counts()


### EDA (Exploratory Data Analysis)

- from scipy.stats import * : import all of tools

- Make Test Data by randomly 

`income = np.random.normal(20000, 50000, 100)`
`income = np.append(income, 10**9)`

- `trim_mean(income, 0.2)` : mean[20% ~ 80%]

- mode
`scipy.stats.mode`
`Series.value_counts().index[0]`: Be careful for overlap mode values

`x = np.random.choice([‘A’, ‘B’, ‘C’], 1000)`
`mode(x)` : [0]: mode value, [1]: counts

- var, std

`np.var(x, ddof =1)`: presume using sample data -> /n-1`
`np.std(x, ddof = 1)`

`variation(x)’: Considering scale variation(But! All of data must be positive value)

- Scaling

	1. Standard Scaling[-inf~+inf]
`from sklearn.preprocessing import StandardScaler`
`scaler = StandardScaler()`
`Z = scaler.fit_transform(X)`
`pd.DataFrame(Z, index = X.index, columns = X.columns)`

	2. Min-max Scaling[0~1]
`from sklearn.preprocessing import MinMaxScaler`
`scaler = MinMaxScaler()` : 인스턴스화-객체 선언
`Z = scaler.fit_transform(X)` : fit_transform -> ndarray
`pd.DataFrame(Z, index = X.index, columns = X.columns)`
 
- Range
`np.max(x) – np.min(x)`

- IQR
`np.quantile(x,0.75) – np.quantile (x, 0.25)`

- percentile vs quantile      
`np.percentile(x, q)` q(1~100)
`np.quantile(x,q)`    q(0~1)

- For distribution, countplot() 
`pd.Series(x1).value_counts(sort=False).plot(kind =’bar’)`: for not descending mode

- Certification Hypothesis

> 1. 정규성 검정(분포가 정규 분포를 따르는지)
> 	- KS test
`scipy.stats.kstest(x, ‘norm’)` -> result=(statistics, pvalue) -> pvalue < 0.05 : 작으면 정규성 크면 미충족, 비모수 검정 고려

> 2. 등분산성 검정(2개 이상의 산포 비교)
`scipy.stats.levene(s1,s2…)` -> pvalue < 0.05 : 두 분산은 같지 않다.

> > 정규성 충족시[t-통계량 = (평균의 차)/분산]
> > 	- 단일 표본 t-검정(평균과 기준값 비교)
`scipy.stats.ttest_1samp(x,popmean)`: popmean 검정하고자하는 기준값 (H1, 대립가설에 따라서) -> result = (statistics, pvalue) -> pvalue < 0.05: x는 popmean과 같지 않다./ t-stat >0 -> x_bar > popmean, t-stat < 0 -> x_bar < popmean

> > d(차이)가 정규성 충족
> > 	- 쌍체표본 t-검정(한 표본에 대해 실험 전, 후_시간 차이로  값 차이의 유의성을 비교하는 것)
`scipy.stats.ttest_rel(a,b)`:a,b 실험 전 후 결과/ pvalue < 0.05 -> a와 b 차이가 유의하다./ t-stat>0 -> 양의 변화, t-stat<0 -> 음의 변화 (양의 변화다_데이터를 봐봐 살이 빠지는 변화인지 아닌지)

> > 정규성, 독립성, 등분산성(분산이 평균에 영향) 충족시
> > 	- 독립 표본 t-검정(단일 표본은 1samp에 대해 기준값과 비교했다면, 독립 표본은 2samp의 평균 비교)
`scipy.stats.ttest_ind(a,b,equal_var =True or False)` -> result=(statistics, pvalue) : statistics가 양수 -> a_bar>b_bar, 음수 -> a_bar< b_bar/ pvalue < 0.05, a_bar != b_bar

> > 정규성 미충족
> > 	- 윌콕슨 부호-순위 검정(평균과 중위수 비교)
`scipy.stats.wilcoxon(x)` (popmean = x의 중위수)_ 정규분포가 아니니 평균보다 중위수, 또한 X_bar – mean이 0이 되므로-> t-stat >0 -> x_bar > x_med, t-stat <0 -> x_bar < x_med/ pvalue < 0.05 -> x_bar != x_med

> > 	- Mann-Whitneyu 검정
`scipy.stats.mannwhitneyu(a,b)` -> t-stat >0 -> a_bar>b_bar, t-stat<0 -> a_bar < b_bar, pvalue < 0.05 -> a_bar != b_bar

## To Do

	- Certification & Pattern
