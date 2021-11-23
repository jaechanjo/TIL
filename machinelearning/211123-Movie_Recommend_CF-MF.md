## Done

### Recommend System

> Rating

> > Explicit Feedback: 영화의 점수, 리뷰 점수, 좋아요 표시와 같은 사용자가 직접적인 의도를 담아 표시한 반응

> > Implicit Feedback: 조회 여부, 시청 여부, 구매 여부, 찜 표시(직접 구매까지 이어지기 전 단계, 간접적인 반응)

> Rating Matrix Recommend

> > Colaborative Filtering(CF)

	1. User_based: 보통 유저 기반 협업적 필터링 방식을 주로 사용하는데, 그 이유는 새로운 컨텐츠에 대해 수반되어야 할 필요 정보량이 많지 않기 때문에, 극단적으로 부합하는 단 하나의 사용자만 존재하더라도 방법을 사용할 수 있다.

	2. Item_based: 그러나, 아이템 기반이라는 것은 아이템 자체에 대한 평가가 구축되어야 하므로, 단 하나를 대표로 사용하는 것은 무용지물과 다름 없다.

	- KNN 알고리즘, 하나의 새로운 데이터를 가장 유사도가 높은 최근접 이웃으로 예측하는 것이다. 유사도는 기존에 알고 있는 정보로 예측한다. 즉 User-based라면 user의 다른 시청기록을 가지고 유사도를 측정하여, k명의 가장 유사한 유저들을 선정한다. k명의 유저가 영화 A를 평가한 점수에, 유사도 가중을 주어, 평균을 낸다. 그렇게 새로운 데이터를 예측한다.

	- 유사도는 보통 Cosine Similiarity를 사용한다.

`from surprise import Dataset, Reader`
`from surprise.model_selection import train_test_split`

`reader = Reader(rating_scale=(1,5))`
`data = Dataset.load_from_df(rating_df[[‘user_id’, ‘movie_id’, ‘rating’]], reader)`
`trainset, testset = train_test_split(data, test_size=0.25)`

`from surprise import KNNBasic`
`from surprise import accuracy`

`algo = KNNBasic(k=40, min_k=1, sim_options={“user_based”: True, “name”:”cosine”})`
`algo.fit(trainset)`
`predictions = algo.test(testset)`

`accuracy.rmse(predictions)`

`print(predictions[:20])`


> > Matrix Factorization(MF)

	- CF는 유사한 값들의 가중평균을 이용해 예측했다라면, MF는 행렬 분해의 수학적 방법으로 예측한다.

	- 특히 하나의 Rating Matrix를 P와 Q로 행렬 분해를 하여, 각각 모르는 항들을 일반적으로 지도학습을 통하여, 찾고 이를 다시 원래 모양으로 만든다. 그리고 이 과정을 Factorization이라고 한다.(이때 R:mXn, P:mXd, Q:dXn이라고 하면, 이때, d는 무궁무진하게 늘려도 되지, 데이터의 양에 따라 조절되는 유동적인 값이며, 값이 숨겨져 있다고 하여, latent factor라고 한다. 그러면 m개의 user_id에 대해 벡터의 성질, 또 마찬가지로 n개의 movie_id에 대해 벡터의 성질로 표현 가능하다.)

	- P와 Q를 선형회귀 모델의 weight로 보고 cost fuction이 경사하강법이라는 최적화 기법에 의거해, 최소가 되도록 하는 P와 Q를 찾아 구한다. 이를 통해, Factorization을 진행하고, 값을 예측하게된다. 이 과정이 Matrix Factorization, MF-based 모델링이다. 또한 중간 지도학습의 선형회귀모델이 사용되므로, Model-based CF라고 하기도 한다. 

	- SVD의 경우에는 특이값 분해로 UDV 3개의 행렬로 분해하는 방법이다. 이때 D는 대각항에 고유값을 갖고, V는 고유벡터로 이루어져있어, 차원축소의 방법인 PCA를 할 때 활용되어지는 기법이기도 하다. 아무튼 이걸로 행렬분해를 진행하고 Factorization을 한다. 따라서 엄밀히 따지면 3개로 분할되는 것인데, 중요한 Parameter는 latent factor -> n_factor=n

	- n_factor의 k값이 늘어날수록, 더 많은 데이터를 학습하게 되고, 과적합에 빠지게 되고, 시간 소요도 늘게 된다. 즉 적당한 k값을 찾아준다. 

`from surprise import SVD`
`from surprise import accuracy`

`algo = SVD(n_factor=50)`
`algo.fit(trainset)`
`predictions = alog.test(testset)`

`accuracy.rmse(predictions)`

`print(predictions[:20])`

## To Do

	- Recommend Algorithm
