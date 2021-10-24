## Done

### EDA

> Distribution
	- 값의 범위가 넓어 히스토그램 분석이 어려운 경우, 아웃라이어를 제거하면서 확인.

`gs = df[‘Global_Sales’].quantile(0.99)`
`df = df[df[‘Global_Sales’] < gs]`
`sns.histplot(x=’Global_Scales’, data=df)`

> > When classifying it by ategorical variable
`sns.histplot(x=’Global_Sales’, hue=’Genre’, kde=True, data=df)`
	- category가 많아서 단순히 hue 색상 구분만으로 어렵다면, boxplot을 이용한다.

> > What is apply func?
	- apply 함수는 대상 컬럼 각각 원소를 elementwise 하게 지정 함수로 변환해주는 함수입니다.
`sns.histplot(data = df[‘User_Score’].apply(float), bins =16)`

> > When can we stop to remove outlier?
`uc = df[‘User_Count’].quantile(0.97)`
	- 99, 98, 97 순으로 상위 범위를 넓혀가다가, 그 폭이 답답하게 막힌 순간, 그때부터는 빈도가 꽤 생기는 지점이라고 판단할 수 있다. 즉, 그 정도에서 제거해준다.

> Correlation

> > What is different with kind, ‘hist’&’hex’in seaborn?
	- 먼저, 이 둘은 빈도의 분포 정도를 색상으로 파악할 수 있는 도구입니다.
	1. hist는 해당구간에 존재만 한다면, 명확히 표현해줍니다.
	2. hex는 상대적 빈도수에 따라 과감히 지워질 만큼 구간의 존재유무를 파악하기 어렵습니다.

> > Multicolinearity
	- Critic과 User count와 score은 엄밀히 들여다보면 차이가 있습니다. 먼저 Critic은 업무입니다. 즉, 판매가 발생하지 않더라도,비평을 하게 됩니다. 그러나 user는 반드시 판매가 선행됩니다. 후기 평가를 남기는 방식으로 Global Sales과 연결고리가 있습니다. 즉, User와 Sales 연결성은 모델 학습 의미가 없으므로, 제거하도록 합니다.

> > Visualization with different columns
	1. 먼저 두 값의 범위가 다르므로 10배해서 맞춰줘야 하겠다.
	2. 별도의 DataFrame을 구성하여 이 둘의 열을 동시에 한 그림에 비교 시각화를 하자.

`critic = df[[‘Critic’]].copy()`
`critic.rename({Critic_Score’:’Score’}, axis=1, inplace=True)`
`critic[‘Score_by’] = ‘Critics’`

`user = df[[‘User’]].copy() * 10`
`user.rename({User_Score’:’Score’}, axis=1, inplace=True)`
`user[‘Score_by’] = ‘Users’

`tot = pd.concat([critic, user])`

`sns. boxplot(x=’Score_by’, y=’Score’, data=tot)`

## To Do

	- Preprocessing & Modeling
