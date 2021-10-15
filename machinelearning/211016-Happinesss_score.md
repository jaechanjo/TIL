## Done

### Data made by export

	- 전문가에 의해 쓰였다라는 것은 전문 리서치 회사(ex: 한국리서치, 월드갤럽)에서 제작한 데이터, 통계적 검증을 마친 데이터이기에 정보라고 할 만하다.

	- 모집단을 추정하기 위해 샘플링, 표본 집단의 분포를 신뢰수준에 따라 검증을 거친 데이터라는 것이다.

### Happiness Score

	- 캔드릴 사다리(ladder): 0-10점 점수로 행복지수를 표현했다. 즉, 표현 기법이다.

	- Distopia: 최악의 평가 지표/ Residual: 나머지 고려 기준으로 설명되지 않는 데이터들

#### How to utilize related DF
	- put in a total dictionary
`df = dict()`
`df['2015'] = pd.read_csv('2015.csv')`
`df['2016'] = pd.read_csv('2016.csv')`
`df['2017'] = pd.read_csv('2017.csv')`
`df['2018'] = pd.read_csv('2018.csv')`
`df['2019'] = pd.read_csv('2019.csv')`
`df['2020'] = pd.read_csv('2020.csv')`

	1. columns의 이름을 단번에 바꿀 수 있다.
`cols = ['country', 'score', 'economy', 'family', 'health', 'freedom', 'generosity', 'trust', 'residual']`
`for key in df:`
`df[key].columns = cols`

	2. concat할 때도 역시나 simple
`df_all = pd.concat(df, axis=0)`

	3. 데이터프레임들 간단히 살펴보기
`for key in df:`
`print(key, df[key].columns)`

#### Additional Skills

	- 여러열의 값들을 사칙연산하는 방법
`df[‘2018’][‘Score’] – df[‘2018’][[‘GDP’, ‘Generosity’]].sum(axis=1)`

	- 열의 순서를 바꿀 때는 이렇게 간편히!(기존 GDP가 앞일 때)
`df[‘2016’] = df[‘2016’][[‘Generosity’, ‘GDP’]]`

	- Pivot테이블 활용하기
`df_all.pivot(index=’country’, columns=[‘year’], values=’rank’)`

## To Do

	- EDA of Happiness Score Data
