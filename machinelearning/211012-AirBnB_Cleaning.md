## Done


### Data Cleaning

- 각 컬럼을 분석하여 미기입/ 오기입 데이터 확인
	1. 수치형: 통계를 이용해서, info(), boxplot(), rugplot()
	2. 범주형: unique(), value_counts()를 이용한다.

- 클래스가 너무 많을 경우
`neigh = df[‘neighbourhood’].value_counts()`
`plt.plot(range(len(neigh)), neigh)`
`df[‘neighbourhood’] = df[‘neighbourhood’].apply(lambda s: s if str(s) not in neigh[50:] else ‘others’)`

- 이상치 제거하기(데이터 특성과 분포를 보고)
: 꽉 차있다면, uniform한 분포의 데이터
`sns.rugplot(x=’ price’, data=df, height =1)`
`p1 = df[‘price’].quantile(0.95)`
`p2 = df[‘price’].quantile(0.005)`

`df = df[(df[‘price’] < p1) & (df[‘price’] > p2)]`
`df[‘price’].hist()
: 잘 바뀌었는지 확인

- 다른 변수에 대해 같은 클리닝을 진행할 때, 한 번 더 백분위수 확인!(위의 클리닝에 계속 영향을 받으므로)
`mn1 = df[‘minimum_nights’].quantile(0.98)`

: 괜찮다면, 계속 클리닝 진행, 아니라면 새롭게 분포를 보고 고려!
`df = df[df[‘minimum_nights’] < mn1]`
`df[‘minimum_nights’].hist()`

- 0값은 입력하지 않은 값이며, 백분위수로 처리하기에는 너무 많아도 많은 경우
: (대응) 새로운 열을 만들어, 0인지 아닌지 구분은 해놓자!
: 그리고 fillna()를 하자!
`df[‘is_avail_zero’] = df[‘availability_365’].apply(lambda x: ‘Zero’if x==0 else ‘Nonzero’)`

-미기입 데이터(Nan) 처리하기
: (대응) fillna()하기 전, 새로운 열을 추가해 구분해두자!
`df[‘review_exists’] = df[‘reviews_per_month’].isna().apply(lambda x: ‘No’if x is True else ‘Yes’)`

`df.fillna(0, inplace =True)`

## To Do

	- Preprocessing & Modeling
