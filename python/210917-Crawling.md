## Done

### Data Crawling

> Program of exploring web automatically

> > 1. url define

	url = input('input url address : ')

> > * how to take a roundabout way

	seed = np.random.randint(100)
	np.random.seed(seed)
	a = np.random.randint(5)
	time.sleep(a)

> > 2. request url

	import requests
	response = requests.get(url)

> > 3. text -> html

	from bs4 import BeautifulSoup
	html = BeautifulSoup(response.text, 'html.parser')

> > 4. html.select

	for con in html.select('dd.class')[:3]:
		print(con.text)

> #### HTML (Hyper Text Markdown Language)

> > > `<ul> <li> <tag class = 'class_name1 class_name2' id = '1234' href = 'link'>text</tag> </li> </ul>`

> > html.select(tag.class_name1 or 2) , html.select(tag#1234)

> > html. select(ul > li > span.class)


> #### Checking Successful Connection

> > > requests.codes.ok
> > > 1. 100 - explain their web site
> > > 2. 200 - Success of Connection
> > > 3. 300 - We move url to another
> > > 4. 400 - User's request fault
> > > 5. 500 - Server Problem

	if response.status_code == requests.code.ok:

		Crawling

	else:

		Fail Crawling

## To Do

	- Crawling of Dynamic Pages

