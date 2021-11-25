## Done

### Recommend Evaluation

> CF, MF의 한계 : 시간적 고려가 안된다.

	1. 시간이 고려된 선호도가 아닌, 타겟 선호도에 대해 Estimated 되거나 Factorized 된 점수이라는 점이 한계다.

	2. 한 유저나 아이템이 시간에 따라 변화를 감지하지는 못한다.

	3. Test 데이터 역시 무작위 일부분 추출이므로, 시간이 고려되지 않은 “랜덤한 빈공간 찾기”식으로 평가됨. 사실 빈공간을 예측할 때 어떤 우선순위가 있는 것이 아니다. 다만 무작위적인 것이지, 그러므로 순서에 따른 근거의 변화, 근거 역시 예측값이므로, 이로 인한 변동성이 발생할 수 있다.

> > 시간을 고려해보자!(미래의 알 수 없는, 그러나 관심있는 데이터를 이용해서, 과거의 학습 모델이 얼마나 잘 맞추는지가 진정한 추천 시스템이므로)

`rating_df[‘time’].quantile(q=0.8, interpolation=’nearest’)`
`train_df = rating_df[rating_df[‘time’] < 990909][[‘user_id’, ‘movie_id’, ‘rating’]]`
`test_df = rating_df[rating_df[‘time’] >= 990909][[‘user_id’, ‘movie_id’, ‘rating’]]`

`user_watch_dict_list_test  = test_df.groupby(‘user_id’)[[‘user_id’, ‘movie_id’]].apply(lambda x: x[‘movie_id’].tolist()`

	- rating_matrix

`data = Dataset.load_from_df(df=train_df, reader=reader)`
`train_data = data.build_full_trainset()`
`algo = SVD(n_factors=50`
`algo.fit(train_data)`

	- Extract contents you don’t watch
`test_datat = rain_data.build_anti_testset()`
`predictions = algo.test(test_data[:20])`

> Evaluation 

> > MAP@K (모델이 좋아할거야라고 예측한 제안들 중에 정말로 사용자가 좋아한다고 말한 확률: Precision -> 각유저에서 모든 추천 대상 유저로, 전체 시스템에 확장시킨 지표: Mean Average Precision, K는 TOP K개 모델이 몇 개를 제안할 것인가)

	- k의 개수가 커지면 커질수록, 추천의 개수가 늘어나므로 맞추는 비율이 떨어진다. MAP값이 떨어진다. 즉, 적절한 K값을 설정해주는 것이 유저의 만족도가 높아질 것!

> > CTR, CVR : 추천이 실제 시청 혹은 구매로 이어지는 경우를 나타낸 지표, 최대화하는 방향으로 추천시스템을 구축한다.

> Limitation

> > 데이터의 개수(유저, 아이템의 수)가 많아지면, Rating Matrix가 sparse해집니다. 빈 값도 많아질 경향, 즉 실제로 맞출 수 있는 비율도 줄어들어, Precision이 급격히 하락할 것입니다. 따라서, Extract Candidata, 그러나 Maximize Diversity로 하고, Narrow Down, Consider Recent Actions(시계열 기반 축소), 다시금 유의미한 Ranking 즉 성능 좋은 추천을 향해 간다.

> Word2Vec:콘텐츠를 벡터로 표현. 가장 기본적인 추천 알고리즘인 연관 규칙(ex: 가장 유사한 영화)을 사용하는 데 필요한 재료를 만드는 것이다.

> > 장점: CF 만큼 데이터가 커도 조금 더 빠르고 정확하게 벡터로 표현할 수 있다는 점. 개념적으로 큰 차이가 없다.

> > 한 문장, 하나의 feature에 대해, 관심 단어의 주변 단어를 가지고 word를 item으로 치환하면 item2vec으로 활용할 수 있다. 그러나, 이것은 어디까지나 학습 데이터 안에서만 가능한 것이고, 만약 새로운 단어, 유저, 아이템이라면 cold-start 문제가 필연적으로 발생하게 된다. 즉, Meta-Prod2Vec과 같이 여러 feature의 연관성을 기반으로 함께 학습시키고, 최대한 그물망을 넓게 쳐서 하나라도 더 걸리게 함으로써, cold-start 문제를 극복해본다.

> Meta-Prod2Vec

> > Word2Vec: [[‘movie_id:3186’, ‘movie_id:1721’,...]] vs Meta-Prd2Vec: [[ ‘movie_id:3186’, ‘year:1990’, ‘genre:Drama’, ‘movie_id:1721’, ‘year:1990’, ‘genre:Drama’,...]] : 여러 특징 들을 총 동원해서 새로운 유저, 아이템에 대한 cold-start 문제를 해결하는 것이다.

> > 임베딩(embedding) 기반이란, 끼워넣는다 즉, 연관 유사 기준에 부합하는 유저 혹은 아이템으로 끼워넣어 추천하겠다!

	- 같은 개념이므로, 코드도 같은 것으로 쓴다.
`from gensim.models import Word2Vec`

`model = Word2Vec(movie2vec_dataset,
size =100, #lookup할 크기, 학습된 네트워크의 hidden layer(projection layer)에 단어의 벡터가 표현된 것이 lookup
window=6, # 주변에 몇 개의 단어들을 함께 살펴볼 것인가
sg=1, #skip-gram OR cbow 중에 무엇을 선택하실 것인가요(두가지방법이라는데, 무엇?) 
hs=0, #hierarchical softmax OR negative sampling 의 두가지 최적화 방법 중에 무엇을 사용하실 건가요?
negative=20, #negative sampling 파라미터
min_count=1, #유사도, 연관 판가름 기준 척도랄까? 최소 word가 몇 번 이상 등장해야, 기인지 아닌지 볼 건인가
iter=20)`

`model.wv.most_similar(“movie_id:1”, topn=5)` #movie_id 1번과 가장 유사한 5가지 콘텐츠 그러면 그것대로 추천해주면 되는거겠지??

> > item2item이란, generate candidate의 한 방법으로, 특정 영화에 대해 메타데이터를 활용하여 연관 영화를 추출한 리스트로, 이 내역이 곧 candidate이 되는 것이다.

## To Do

	- CBF Recommend System
