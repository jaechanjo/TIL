## Done

### Modeling

- CPU & Thread

	- HT 기능, Hyper Threading 기능이 탑재된 CPU는 하나의 코어가 일을 하는 동안, 남는 잉여 자원을 활용하는 기술, 즉 하나의 코어가 2가지 방향으로 움직일 수 있다.

	- For example, 2 Core 4 Thread, 4 Core 8 Thread

`import os’
`n_cpu=os.cpu_count()`
`n_thread=n_cpux2`

- Regularization

	1. C_parameter

	- C는 오차 합의 계수다. 즉, **작으면 Regularization 강화**, 크면 Overfitting.

	- 정규화 항의 종류:L1_lasso, L2_ridge, elastic_net, 회귀계수 합의 계수 람다, alpha값이 **클수록** 0으로 수렴하니까, **차원 축소 Regularization 강화**

	- ‘tol’=threshold() 작으면, 약하다, 회귀계수 값이 완전히 값이 일정하게 수렴할 때까지 기다린다./ 크면, 강하다, 허용 범위가 작다. 어느정도 변화는 있지만 얼추 나왔다 싶으면 바로 끊어버린다.

	2. gamma-SVM kernel

	- decision boundary의 곡률과 관계있다. 클수록 복잡한 경계.

- Grid Search

	- C나 gamma와 같은 hyper parameter는 여러 개를 시도해보면서, validation data에 대한 오차가 가장 낮은 값을 찾는다. 이를 grid table 형식으로 search해보겠다.

- Evaluation Index

	1. Precision
	- 모델이 0을 참이라고 예측했을 때, 맞춘 확률

	2. Recall
	- 실제 정답이 0이라고 할 때, 모델이 맞춘 확률

	3. F1-Score
	- 0이 참일 때, Precision과 Recall의 조화평균, 이것의 기하학적인 의미 사다리꼴 비스듬한 변에 쏜 평균, 즉 이상치에 쉽게 휘둘리는 평균에서 보완된 평균. 즉, 클래스가 불균형할 때, 휘둘리는 일반 평균 말고, 조화평균을 쓴, F1-Score을 사용한다. 그리고 사용한 게 F1-Score

- Random Forest: Improving decision tree problems

	1. Bagging
	- Bootstrapping: sampling with replacement(하나의 원소에 대해 뽑을 때마다 다시 돌려 놓는다. 매 뽑는 순간순간 마다 복원<->다 뽑은 뒤 복원: sampling without replacement)으로 구성이 조금씩 다른 여러 데이터 셋을 생성해낸다.
	- Aggregating: 합하거나, 평균을 내어 분산을 줄인다.(표본 평균의 분산이 /n되는 것을 떠올려라)
2Drop-out: bootstrapping해도 얼마나 다르겠니, 그래서 더욱 다양하게 하기 위해, 각각 decision tree의 일부 뉴런들을 무작위 제거하여 Correlation of Trees를 감소하게 한다.

- MLP(Multi Layer Perceptron) ->(문학적) ANN(Artificial Neural Network) 

	- Activation function – Output fuction: 마지막 결과를 얻어내는 단계에서 쓰이는 함수이므로, 별개로 취급한다. 상황에 따라 이 부분만 변화를 줄 수도 있겠지.

	- **(Node num of a layer, Layer num) Guide** 

	- Grid_Search ([Max(factor=1) – Min(factor=10)], layer=1) : layer는 우선은 1개로 시작하자. 대부분 1개에서 좋은 해결법이 나오는데, 여러 경우를 고려해하므로, 복잡한 해결법이 필요한 경우더라도 1부터 시작해!

	- Solver : lbfgs는 데이터 양이 많지 않을 때 좋은 성능을 보인다고 알려져있다.

	- 층의 개수가 늘어나면, 차수가 높아진다. 이렇게 생각해, 2차 함수랑 4차 함수를 떠올려봐, 2차함수는 하나의 극대, 극소가 있으므로 헤매지는 않지만, 4차함수는 두개의 봉우리가 있다. 그래서 local minimum에 빠질 가능성이 언제나 도사리고 있어, 그러니까 우연찮게 가중치의 초기값을 잘못을 배치하게 된다면(물론 이를 보완한 Boltzman Machine을 이용해 초기값을 보정해줄 수 있다.), 큰 오류 사고가 일어난다. 즉, 비슷한 성능을 낸다면, 단연 단순한 1개의 층에서 가능한 적은 노드의 수를 채택하는 이유가 되는 것이다. (물론, 복잡한 문제를 풀 때, 또 그에 상응하는 많은 데이터가 확보되어 있을 때, 은닉층의 수를 늘려가는 것은 옳다.-이런 상황 속에서도 global minimum을 찾기 위한 다양한 기법들이 개발되고 있으므로 딥러닝이 새로운 대안으로서 각광받고 있는 것이다.)

- Model Evaluation

	- Logistic Regression 같은 경우 모델의 특성상 복잡도를 늘리는데 한계(penalty)가 있었지. 즉, 아무래도 성능이 낮게 나오는 경향이 생긴 것으로 평가된다.

	- Decision Tree 복잡도를 늘리는 것은 가능했으나, 과적합에 따른 test 성능 평가에서 약간의 저하가 발생될 수 있었을 것이다.

	- Random Forest 복잡도를 늘리기는 하되, Bagging, Drop-out과 같은 기법들을 사용하면서, decision tree의 변동성-복잡성을 성능을 개선시키는 방향으로 해소하였다. 즉 가장 좋은 성능을 보였다. **그러나, 앙상블 모델이므로 컴퓨팅 파워 요구량도 클 뿐더러, 숲이므로 해석이 어렵다.**

	- Support Vector Machine LR < SVM < RF, 비선형 가우시안 모델로 LR보다 복잡성은 확보가 되었다.

	- ANN_MLP activation, alpha, **hidden_layer_sizes**, solver등 여러 시행착오를 통해 hidden_layer_sizes를 선택한 만큼 뛰어난 성능이 나왔다.

- Interaction term vs Multicolinearity

	1. Interaction term(교호작용): X1, X2가 Y에 영향을 끼치지는 않지만, X1과 X2가 결합되면서 Y에 중요한 영향을 끼칠 수 있다. – (X1X2, X1^2, X2^2, sin(X1), sin(X2)...) 다양한 조합을 만들어 학습시켜 본다.)

	- 근데, 이걸 어떻게 알아? 뭐가 어떻게 만들었을 때 그것들을 또 무슨 조합으로 합쳤을 때 효과가 있을지 어떻게 알아?? 그래서, 사실 거의 안쓰고, 도메인 지식이 뚜렷할 때만 효과가 있다고 가정을 하고 쓴다.

	2. Multicolinearity(다중공선성): X1과 X2가 Y에 영향을 끼치는 변수이며, X1과 X2 사이에서도 선형관계가 관찰되는 경우, 즉 차원만 늘리고 복잡한 모형에, 속도 저하, 과적합, 성능 저하를 시키면서도, 각자의 영향이 시너지가 아닌 상쇄시키는 악효과만 일으키게 된다.

- Boosting(취약 집중 공략)

	- 오분류 데이터를 다음 라운드에 가중치를 크게 부여해, 꼭 뽑히게 만든다. 그래서 다음은 잘 맞추고, 최종 다수결 투표에서 잘 맞추도록 하는 기법.

	- 딥러닝을 제외하고, 대부분 상황에서, 대부분의 기법보다 거의 1등의 성능을 내고 있는 모델이다.

	1. XGBoost: Grad Boost + Regularization term
	2. lightGBM : XGBoost 성능 비슷 + 가벼워 빠르다 -> 선호

## To Do

	- Next Patter Practice
