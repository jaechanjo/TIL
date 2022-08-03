## Done

### Data Handling

- The Importance of Data Preprocessing

	1. Enable to Analyze (efficiently)

	2. The 79% Time of Total Mean Analysis

	3. Improvement of Machien Learning Performance

- Googling is the Actual Solution

`python + pandas(module) + how to order columns(question)`

- Tuple Iteration is faster than list

	- Tuple SWAP
`a, b = 1, 2`
`a, b = b, a`

- Dictionary

	- key is fixed value like tuple, but value can be variable like list

- itertools.product(*L)

- zip() <-> itertools.product() : simultaneously vs ordered operated

`for v1, v2 in itertools.product(L1, L2)` = `for v1 in L1: for v2 in L2`

- Combination & Permutation

`for comb In itertools.combinations(L, 2):`
`for perm in itertools.permutations(L, 2):`

- numpy

	- `np.zeros(shape)`: raw vector
	- `np.arange(start, stop, step)`
	- `np.linspace(start, stop, num)`: equidistant intervals
	- `np.random.random(shape)` : random 0~1 matrix
	- `np.random.normal(mean, std, shape)` : normal distribution
	- `np.random.randint(start, end, shape)` : start~end random matrix

	```X[I, j] <-> L[i][j]```
	
	
	```X[[‘a’,’b’,’c’]]``` : list index not 2-d concept

- Pandas

	- Series = Index + ndarray(values)
	- loc vs iloc: loc include end(It is convenient based on name index) but, iloc end + 1 index like habit
	- df[‘col’] -> Series/ df[[‘col’]] -> df
	- pd.set_options(‘display.max_columns’, 100)
	- dtype: int < float < object

- os
	- os.getcwd() : get current password
	- os.chdir(path): change directory to path

- etc

	- `\n` : change line
	- `\t` : tap

- `\` is special key in python so, if you want to use normarly, you have to apply under rules
	1. \\
	2. \ -> /
	3. r”/”: raw string ***

- text data handling

	- nested iteration in comprehension paragraph

	```
	[[float(value) for value in line.split(‘,’)] for line data.split(‘\n’)]
	```

	- read()
	
	```
	os.chdir(path)
	f = open(‘test.txt’,’r’)
	data = f.read()
	f.close()
	data[:1000]
	```

	- readline() : include ‘\n’

	```
	os.chdir(path)
	f = open(‘test.txt’,’r’)
	header = f.readline()
	data = []
	line= f.readline()
	while line: data.append(list(map(float, line.split(‘,’) ) )
	line=f.readline()
	f.close()
	```


	- write() + with as

	```
	with open(‘test.csv’, ‘w’) as f:
	f.write(header)
	for line in data:
	f.write(‘,’.join(list(map(str, line))))
	f.write(‘\n’)
	```

- Load & Save Data

	- pd.read_csv(filepath, sep=’\t’, header=None, use_col=[‘ID’,’X1’,’X2’], index_col=’ID’, nrows=100)

	- df.to_csv(filepath, sep=’\n’, index=False)

	- pd.read_excel(‘test.xlsx’, sheet_name = ‘first’, skiprows = range(6))

	- df.to_excel: input lots of sheets into one excel

`with pd.ExcelWriter(xlsx file path) as writer:` : load or save


`df1.to_excel(writer, sheet_name=’sheet1’, index=False)`
`df2.to_excel(writer, sheet_name=’sheet2’, index=False)`

- Merge & Concat & Pivot & Groupby Datas

	- pd.merge(df1, df2, left_on=’employee’, right_index=True)

	- file names
		1. os.listdir(path) : All of file names list
		2. All of sheet names in excel in list type :
		```
		wb = xlrd.openworkbook(file, on_demand = True)
		wb.sheet_names()
		```

	- pd.concat()

	1. CSV-for paragraph

	```
	for file in os.listdir(‘folder name’):
	if ‘2015’in file:
	df = pd.read_csv(‘jjc/’+file, sep=’\t’)
	concat_df = pd.concat([merged_df, df], axis = 0, ignore_index = True)
	```


	2. CSV-list comprehension (memory burden)

	```
	concat_df = pd.concat([pd.read_csv(‘jjc/’+file, sep=’\t’) for file in os.listdir(‘forder name’) if ‘2015’in file], axis = 0, ignore_index =True)
	```


	1. Excel-for paragraph
	
	```
	import xlrd
	wb = xlrd.open_workbook(‘test.xlsx’, on_demand =True)
	sheetnames = wb.sheet_names()
	sheetnames
	```

	```
	merged_df = pd.DataFrame()
	for sn in sheetnames():
	df = pd.read_excel(‘test.xlsx’, sheet_name= sn, skiprows=range(6))
	df = df.iloc[:,1:]
	merged_df = pd.concat([merged_df, df], axis = 0, ignore_index =True)
	```

	2. Excel-list comprehension (memory burden)

	```
	concat_df = pd.concat([pd.read_excel(‘test.xlsx’,sheet_name = sn, skiprows= range(6)).iloc[:,1:] for sn in sheet_names()], axis = 0, ignore_index = True)
	```

- Pivot(Result) == Groupby(Process)

	- pd.pivot_table(df, index=’제품’, columns=’쇼핑몰 유형’, values=[’판매금액’, ‘수량’], aggfunc=’max’)

	- df.groupby([‘type’,’kind’], as_index = False)[‘num’,’sales’].agg([‘mean’,’max’,’my_func’])

- How to Order Data Structure


	- List_Tuple - sort() vs sorted()
	1. `L.sort(reverse =True)`: inplace = True

	2. `sorted(L, key= lambda x: abs(x-3), reverse = True)`


	- Series, DataFrame – sort_values()
	`S.sort_values(ascending=False, na_position = first, key = lambda x: len(x))`

	`df.sort_values(by=[‘name’,’type’], ascending =False, na_position = first)`


	- Series – value_counts() & unique() (== set())

	`S.value_counts(ascending =False, normalize = True)`:Instead of count num, print ratio num -> Class Imbalance Problem

	`S.unique()`: ndarray including Nan -> Determine whether Categorical or not by len(S.unique())


	- DataFrame.drop_duplicates(subset=[‘name’], keep=’last’)


- Indexer – df.loc[], df.iloc[]

	- **(replacement)** df[‘A’] = ‘view’ -> SettingWithCopyWarning(likelihood of original data modification) -> df.loc[‘A’] = ‘view’

- Masking

	- df[‘A’].isin([1,3,5])

- Series.str accessor

	- Series.str.strip() : remove blank
	- Series.str.contains(s) : Boolean data whether it includes str(s) or not
	- Series.str.split(sep, expand=False) : expand (True : add new column or False: print result of list)

	```
	df[‘Serial_num’].str.split(‘-‘, expand = True).head()
	concat_df = pd.concat([df, df[‘Serial_num’].str.split(‘-‘, expand = True)], axis =1)
	concat_df.rename({0:’공정’, 1:’제품’, 2:’식별자’}, axis=1, inplace= True)
	```

	- Series.astype(str)


## To Do

- Visualization 
- Matplotlib.pyplot
