## Done

### Generate Candidate

> Ranking & Predictor of Recommend System

> > Based on Train Dataset, Candidates according to UserID

	1. 가장 최근 시청한 영화 3편의 유사 영화 5개씩 목록, 즉 user 별 15개의 영화 목록 리스트

	- Word2Vec을 사용하여, feature 별로 가장 유사한, feature가 가장 빈번하게 겹치는 유사 추천 목록을 user 별로 추출했다.

	2. 평점 카운트가 10개 이상인 영화의 평점 평균 상위 10개의 영화 목록 리스트

	3. 장르 별, 평점 평균 상위 Top 10의 영화 목록 리스트

	4. 연도 별, 평점 평균 상위 Top 10의 영화 목록 리스트

	- 최종 병합하여, Train 데이터 기반 4가지 추천 후보군 생성

> > Based on Ranking Model(Factorization Machine), Candidates according to UserID

	1. Item(Movie) 별 (탐색을 통해) 선택한 feature와 User 별 선택한 feature를 libsvm format으로 변환하여, 2차원 벡터로 모델에 fit하게 가공해준다.

	2. 모델에 학습 시켜서, Predict 결과값들을 확인하면, {User : {MovieID : Probability, ...}, User2 : {...}, ...} 형태로 출력된다.

	- 랭킹 모델을 이용한 추천 후보군 K개 생성 완료.

### Evaluation

> 시간에 따른 최신 test data로 평가를 하는 이유

	- 새로 들어온, 최근 특정 유저의 시청 기록이, 정말 과거 학습 데이터 모델 기반 추천 목록과 부합할 때라야, 그 추천 모델의 가치를 평가할 수 있는 것이다.

> > MAP(Mean Average Precision)

	- Precision : 모델이 긍정이라고 예측했을 때, 시청자들이 실제로 추천 영화를 봤다고, 선호했다고 예측했을 때(Y) 실제로 맞은 확률

	- 각각 유저의 Precision을, 전체 대상 유저들로 확대하여, 평균한 값: MAP : 가령, 0.1이면 10개 중 1개 정도는 추천이 맞는다고 해석할 수 있다.

> 추천 후보군, CF 모델의 한계를 보완하여 CBF 아이디어 기반 후보군 추출은 Lage Scale의 Extreme Sparse Vector에 대해 feature를 축소시켜, libsvm 포맷으로 랭킹 모델링에 학습시켜 추천할 수 있다.

## To Do

	- E-commerse Clustering
