## Done

### Linear

``X_1, y_1 = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1, n_classes=2, class_sep=3) #class_sep: separate 잘 구분되도록하는 지표
# 각 관측치가 선형 관계에 있도록 만든다.
X_1[:,1] = (X_1[:,0] + X_1[:,1])/2
# 각 항에 오차항을 만들어준다.
rng = np.random.RandomState(2)
X_1 += 2*rng.uniform(size=X_1.shape)``

``color_code = {0:'Orange', 1:'Skyblue'}
plt.figure(figsize=(7,7))

sns.scatterplot(x=X_1[:,0], y=X_1[:,1], c=[color_code.get(i) for i in y_1])
plt.show()``

### NonLinear

``plt.figure(figsize=(7,7))
n = 200
r_list =[1, 3, 5]
sigma = 0.2
param_lists = [(r, n, sigma) for r in r_list] 
coordinates = [ CircleXY(param[0], param[1], param[2])  for param in param_lists]
color_code = {1:'Orange', 3:'Skyblue', 5:'Green'}

for j in range(0, len(coordinates)):
    x,y,group = coordinates[j]
    sns.scatterplot(x=x, y=y, c=[color_code.get(i) for i in group]) # 각각 점들을 하나하나 모두 찍어보고 싶다면, 산점도!
plt.show()``

### Data Processing

``X1,X2,y_2_bf=[ np.append(np.append(coordinates[0][i],coordinates[1][i]),coordinates[2][i])  for i in range(0,3)]
X_2=np.c_[X1,X2] # 두 x를 튜플로 묶어준다.
mapping_dic = {1:0, 3:1, 5:2}
y_2_bf2=[mapping_dic.get(i) for i in y_2_bf]
y_2=np.array(y_2_bf2) # y를 범주형, 반지름을 크기 순으로 0, 1, 2로 치환해서 다시 넣어준다.``

## To Do

	- Nonlinear-Clustering
