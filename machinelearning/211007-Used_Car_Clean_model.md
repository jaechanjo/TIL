## Done

### Data Cleaning

> Categorical Data

	1. Remove Row involve in missing value

	2. Replace missing value with ‘others’

	3. Collect so few category to convert ‘others’

	4. (plus) Through training by classifier, fill missing data with inferred value

> > 범주형 데이터를 one-hot 벡터로 get dummies해서 바꿔주게되면, 차원의저주에 걸려 성능이 급격히 저하될 수 있다.
따라서, 상위 10, 이렇게 딱 정하고만 갈 수는 없으니까, 직접 그리면서 필요한 만큼을 뽑아줘야 한다.
어떤식으로 뽑아주냐, `value_counts()`를 보고 종류별로 빈도수가 많은 상위를 뽑아주되, 그 범위의 기준은 상위 몇개부터
급격히 완만해냐에 따른 지점에서 끊어준다.
군집분포라든가, 주성분분석에서 쓰이는 Elbow나, Scree plot과 같은 idea!

`col = target column`
`counts = df[col].fillna(‘others’).value_counts()`
`plt.grid()`
`plt.plot(range(len(counts), counts)`

`n_categorical = num you decide to cut for control point`
`others = counts.index[n_categorical:]`
`df[col] = df[col].apply(lambda s : s if s not in others else ‘others’)` 

> Numerical Data

값의 범위가 너무 넓을 경우, histplot 등이 잘 작동하지 않으므로, rugplot을 사용한다.

- quantile()을 이용해서 Outlier를 제거하고 시각화해서 확인한다.
`p1 = df[‘price’].quantile(0.99)`
`p2 = df[‘price’].quantile(0.1)`
`df = df[(p1 > df[‘price’]) & (df[‘value’] > p2)]`


- describe()을 이용해서 이상치 제거가 잘 이루어졌는지 평가한다.
	1. 평균이 중간값에 얼마나 가까워졌는지 -> 가우시안 분포
	2. 3분위수랑 Max가 얼마나 가까워졌는지 같은 자리수면 베스트! 단, 그렇지 않더라도 용인 가능!

- heatmap() 음의 상관성이 심각해, annotation이 잘 안보인다면, YlOrRd 색도 예쁘다.
`sns.heatmap(df.corr(), annot = True, cmapp= ‘YlOrRd’) 

- 선형 회귀를 모델링으로 사용하지 않는 경우, drop_first를 할 필요가 없다. 그렇다면, drop_first를 왜 하는가? 차원 축소의 의미와, 다중공선성을 완화하려는 시도이다.

- Scatter로 시각적 확인이 어려운 경우가 있다. 값의 빈도가 아주 적은 경우, 그때는 histplot을 활용하자.

### How to an comprehend error by histogram

- 가장 최빈값이 0보다 작다면 예측을 과소로 평가했다라는 의미, 반대로 크다면 과대 평가했다는 의미
- 왜도와 첨도에 따라 분포에 대한 유동적인 해석이 가능하다.

`err = (pred – y_test) / y_test * 100`
`sns.histplot(err)`
`plt.xlim(-100, 100)`
`plt.grid()`

- 만약, err = (pred – y_test)로만 다룬다면, 즉 residual을 다루는 것인데 이때는 0을 중심으로 좌우 대칭된 분포는 모델의 필연적인 존재 방식이며, 첨도가 높다라는 것은 예측이 꽤 잘되었다라고 볼 수 있겠다.

## To Do

	- chaper 2. AirBnB Data Analysis
