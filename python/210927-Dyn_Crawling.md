## Done

> Dynamic Webpage Crawling 

> > If page moves, but link is fixed, it it **dynamic page**
	
	1. Hidden url(Developer tools -> Network)

`url = link
info = { key1: value1, key2: value2}`

	2. check request method & provide info(referer, user-agent)

`requests.get or post(url, headers=info)`

	3. randomly approach web

`seed = np.random.randint(100) 
np.random.seed(seed)
a = np.random.randint(5)
time.sleep(a)`

	4. response by request method, 'header = info' & check state of connect

`response = requests.get(url, headers=info)`

	5. check response.text & choose how to import it whether html or json

`import json
data = json.loads(response.text)
data[0]['ranks']`

	6. process and extract wanted data


> > strip()

- extract only text from data mixed with blank or etc

> > reshape(-1) vs reshape(-1, 6)
-  Former means transposition of axis, but -1 of latter means automatic match

> > input list into empty dataframe

`df = pd.DataFrame(columns=['A', 'B']

df[A] = list_A
df[B] = list_B

df`
