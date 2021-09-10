## Done

	- Package(Module, Library, Class)

		1. os: for manage file or dirtory

		```path = os.getcwd()```
		: check present path

		```os.chdir('./basic_python/')```
		: change path

		```os.mkdir('test')```
		: make path

		```os.rmdir('test')```
		: remove path

		```os.listdir()``` #ls
		: list present contents

		2. glob	: sort the file in condition

		```glob('*.ipynb')```

		3. time, datetime : for time data

		```time.sleep(2)```

		```current_time = datetime.now()```

		: method(year,day,month,hour...)

	```current_time.strftime(%y/%m/%d %a %p)```

```dt.strptime(current_str_time,  '%y/%m/%d %a %p')```

		4. Read & Write text

		```f = open('test.txt', 'w')```
		```f.write(text)```
		```f.close()```

		```f = open('test.txt', 'r')```

	
	- Numpy (Numerical Python)

		: Vector arithmetic calculation

		: BroadCasting

		1. Linear Algebra
		: Scalar, Vector, Matrix, Tensor
		
		: Inner Product of Vector #A @ B

		2. np function - Universal

		```argmax(), argmin()```
		: print index of max or min value

		```arange(start, end, steps)```
		: range specific np.array

		```np.linspace(start, end, size)```
		: np.array adapt to specific size

		```reshape(A, B)```
		: change [A, B] dimesion array

		```transpose()``` #T
		: change row with column

		```unique(np.array)```
		: exclude overlapped data


		3. np.array calculation
		: So fast!, but only same dtype

		- special array
		: ones(), zeros(), empty(), eye()

		4. feature or input function
		
		```test_array.T``` #transpose

		5. Weighted Sum
		: Training Model of Machine Learning

		: w.T @ x

		6. Indexing & Slicing
		
		- fancy indexing
		```np.array[(bool_A) | (bool_B)]```

## To Do

	- crawler
