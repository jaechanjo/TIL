## Done


### Data Visualization

> world happiness score trend in order of the last 5 years

`fig = plt.figure(figsize(10,50))`

``rank2020 = rank_table[‘2020’].dropna()
for c in rank2020.index:
t = rank_table.loc[c].dropna()
plt.plot(t.index, t, ‘.-‘)``

``plt.xlim([‘2015, ‘2020’])
plt.ylim([0, rank_table.max().max() + 1])
plt.yticks(rank2020, rank2020.index)`` # 앞 ticks를 어떻게 배치할 지를 df을 제공하면 알아서 index 순에 맞게 배치를 해주는 것
`ax = plt.gca()` # get current axis
`ax.yaxis.set_label_position(‘right’)` # label postion change
`ax.yaxis.tick_right()` # tick position is also changed

``plt.tight_layout()
plt.show()``

> cumulative bar chart about “Top 20 Happiness Scores”

``fig = plt.figure(figsize = (6, 8))``
``data = df_all[df_all[‘year’] == ‘2020’]
data = data.loc[data.index[:20]]``

`d = data[data.columns[4:]].cumsum(axis=1)` # cumulative barplot idea from using cumsum() y values
``d = d[d.columns[::-1]]
d[‘country’] = data[‘country’]``


`sns.set_color_codes(‘muted’)` # soft color
`colors = [‘r’, ‘g’, ‘b’, ‘c’, ‘m’, ‘y’, ‘purple’][::-1]`
`for idx, c in enumerate(d.columns[:-1]):` # extract index
`sns.barplot(x=c, y=’country’, data=d, label=c, color=colors[idx])`

`plt.legend(loc=’lower right’)`
`plt.title(‘Top 20 Happiness Scores in Details’)`
`plt. xlabel (‘Happiness Score’)`
`sns.despine(left = True, bottom=True)` # delete axis

## To Do

	- Data Prep & Modeling
