## Done

### Matplotlib
> Matlab, mathmatical data visualization

	import matplotlib as mpl
	import matplotlib.pyplot as plt

> > font set

	plt.rc('font', family='NanumGothic)

> line plot

	x = np.linspace(1, 10, 50)
	y = x + 1
	plt.plot(x, y)

> > style set

	plt.plot(x, y, '(color)(marker)(linestyle)')

> > option

	plt.figure(figsize=(x_size, y_size)) #Canvas
	plt.title('text')
	plt.xlabel('text2')
	plt.ylabel('text3')
	
	plt.plot(x, y, 'g*-', label = '$x^2$') #Sketch
	plt.legend(loc = 'best')

	plt.xlim([2, 7])
	plt.ylim([3, 8])

	plt.xticks([3, 5, 7], rotation = 90)
	plt.yticks([4, 6, 8], rotation = 90)

	plt.show() #Picture

> barplot

	plt.figure(figsize=(a,b))
	plt.bar(df.index, df.values)
	plt.show()

	- plt.barh is horizontal version

> histplot

	plt.figure(figsize=(12,6))
	plt.hist(df['loan_amnt'], bins=40)
	plt.show()

	- bins: how much is it densely placed on a unit

> scatter plot

	: relationship with two variables

	plt.figure(figsize=(12,6))
	plt.scatter(df['grade'], df['int_rate'])
	plt.show()

> > sort index strategy

	before visualization, sort values in dataframe

### Seaborn

	: beatiful than matplot

	import seaborn as sns

> barplot

	plt.figure(figsize=(12,6))
	plt.barplot(df.index, df.values)
	plt.show()

> countplot

	plt.figure(figsize=(12,6))
	plt.countplot(df['grade'])
	plt.show()

> histplot(distplot)

	plt.figure(figsize=(12,6))
	plt.distplot(df['loan_amnt'])
	plt.show()

	- histplot also exists but, kde is different

> scatterplot

	plt.figure(figsize=(12,6))
	plt.scatterplot(df['int_rate'], df['annual_inc'], hue='grade')
	plt.show()

> boxplot

	plt.figure(figsize=(12,6))
	sns.boxplot(df2['grade'], df2['int_rate']) 
	plt.show()

> jointplot

	plt.figure(figsize=(12,6))
	sns.jointplot(df2['loan_amnt'], df2['int_rate'], kind = 'hex')
	plt.show()

> pairplot : every scatterplot with columns

	sns.pairplot(df[['loan_amnt', 'int_rate', 'installment']])

> heatmap

	plt.figure(figsize=(12,12))
	sns.heatmap(df.corr()['grade'])
	plt.show()

	- df.corr(): built in function show correlation

### Image Visualization

	img_nm = './img/lena.jpg'

	lena = plt.imread(img_nm)

	plt.imshow(lena)

	- ; is role of eliminating text aside

> Image is Tensor : array

	lena.shape = (512, 512, 4)
	
	- (x-size, y-size, RGB-Brightness)

> > separte RGB in image

	R = lena.copy()
	G = lena.copy()
	B = lena.copy()
	R[:, :, (1,2)] = 0 
	G[:, :, (0,2)] = 0 
	B[:, :, (0,1)] = 0

	- index.0 : Red, index.1 : Green, index.2 : Blue

> > monochorme photograph

	from PIL import Image

	img = Image.fromarray(lena[:,:,2])
	img.save('./img/lena_color_split4.jpg')

> > crop picture

	small_lena = plt.imshow(lena[100:400, 100:400, 2]
)

> > turn upside down, traspose axis of diagonal line


	mo_lena = lena[:,:,0]
	plt.imshow(mo_lena[::-1].T)

> > color inversion

	plt.imshow(255 - mo_lena)


## To Do

	- Crawling
