## Done

### Generate Candidate & Modeling

> 상황: Feature(UserId or ItemId)의 개수가 많아지면 많아질 수록, Matrix에서 1(쓸 수 있는 정보)이 Sparse해진다.(Extreme Sparse Matrix) 

	1. Word2Vec 모델로 Item2Item 후보군, 특히 Meta-Prod2Vec으로 여러 단어들과 연관된 후보들 생성한다.

	2. 탐색적 다양한 추천 후보군 생성

	- 속성 선택 – User/ Item Based – 분포를 보며, 쓸만하지 않은 특징은 제거한다. ex: zipcode) 

	- 특정 시간 기준으로 train과 test 분리(분리 전과 분포 동질성 확인)

	- 범주형 데이터 수치 인덱스 라벨링 – 변환(Feature Mapping – Profiling featuremap mapping)

	- y, 종속, 출력, 기준 변수 선택(ex: 시청 유무, 평가 선호 여부) –(One-hot Encoding, Extreme Sparse Problem)-> libsvm Format Dataset 변환(여러 feature들을 하나의 벡터로 표현하기 위해(pair-wise), 누적 인덱스 값을 사용한다.)

`for idx, row in train_df.iterrows():`
  `vec = []`
  `label = row['y']`
  `vec.append(str(label))`
  `row = row.drop(labels=['rating'])`
  `row = row.drop(labels=['y'])`
  `for key, value in row.items():`
   `col_idx = col_accum_index_dict[key] + value – 1`
   `vec.append(str(col_idx) + ":" + str(1))`
  `print("%s\n" % " ".join(vec))`
  `break`

	- Ranking & Predictor Modeling: Linear Reg + MF Algo : Factorization Machine(Adapt to Recommend System)

	1. linear regression, svm처럼 동작하는 general predictor
	2. 변수 간 모든 pair-wise interaction을 계산하는 알고리즘
	3. General Predictor의 장점 + MF 알고리즘이 가지는 의미 단위 해석(latent factor)의 장점-주요 성분 위주 선택과 집중
	4. Sparse한 데이터셋을 가지고 있을 때 적합( 추천 시스템 학습 모델에 적합)

`fm = FactorizationMachine(k=4, #latent factor의 수`
   `lr=0.005, # weight구하는 오차 최소화 학습률`
   `l2_reg=True, #릿지 규제`
   `l2_lambda=0.0002, # 클수록 규제화`
   `epoch=30,#학습 시행 횟수`
   `early_stop_window=3, # 3번까지 지켜보겠다.`
   `train_data='./train.txt',`
   `valid_data='./test.txt')`
`fm.train()`

## To Do

	- Model Evaluation
