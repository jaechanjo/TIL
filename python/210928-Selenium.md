## Done

### Selenium

> > Framework for web application test
> > it can be advantage of Crawling
> > it makes us approach data if static or dynamic crawling is not acceptable
> > Using it, we can control with all web browser exists currently

	- import
`from selenium import webdriver`
`from selenium.webdriver.common.keys import Keys`

	- Open Web
`driver = webdriver.Chrome('./chromedriver.exe')`

> > Download 'chromedriver' according to chrome version

	- Control
`driver.find_element_by_xpath('copied xpath text').clear()`
`driver.find_element_by_xpath('copied xpath text').send_keys('text')`
`driver.find_element_by_xpath('copied xpath text').click()`

	- Practice : Keyward '해방촌' Selenium Crawling of Instagram

	driver = webdriver.Chrome('./chromedriver.exe')

	url = 'https://www.instagram.com/'
	driver.get(url)
	
	driver.implicitly_wait(10)

	# Log-in
	driver.find_element_by_xpath('//*[@id="email"]').send_keys(fb_id)
	driver.find_element_by_xpath('//*[@id="pass"]').send_keys(fb_pw + '\n')

	driver.find_element_by_xpath('//*[@id="loginForm"]/div/div[5]/button').click()

	driver.implicitly_wait(10)

	# exit pop-up window

	driver.find_element_by_xpath('/html/body/div[5]/div/div/div/div[3]/button[2]').click()

	# Search

	key_ward = '해방촌'

	driver.find_element_by_xpath('//*[@id="react-root"]/section/nav/div[2]/div/div/div[2]/input').send_keys(key_ward)

	driver.implicitly_wait(10)

	## click first item of automatic list

	driver.find_element_by_xpath('//*[@id="react-root"]/section/nav/div[2]/div/div/div[2]/div[3]/div/div[2]/div/div[1]/a/div').click()

	driver.implicitly_wait(10)

	## click first post of search result

	driver.find_element_by_xpath('//*[@id="react-root"]/section/main/article/div[1]/div/div/div[1]/div[1]/a/div[1]/div[2]').click()

	driver.implicitly_wait(10)

	# Crawling Reccursion written date, Like number, tag

	page = 50
	dates = []
	likes = []
	tags = []

	for i in range(50):
		if i == 0:
			html = BeautifulSoup(driver.page_source, 'html.parser')
			date = html.select('time')[0]['title'] #written date
			like = html.select('a.zV span')[0].text #like number
			tags = [tag.text[1:] for tag in html.select('a.xil3i')]

			print(date, like, tags)

			driver.find_element_by_xpath('next post button xpath').click()

			dates.append(date)
			likes.append(like)
			tags.append(tags)

		else:
			
			html = BeautifulSoup(driver.page_source, 'html.parser')
			date = html.select('time')[0]['title'] #written date
			like = html.select('a.zV span')[0].text #like number

			if like == []: #if content is video, there is no like num so, it replaces hits
				like == html.select('video hits xpath')[0].text
			
			tags = [tag.text[1:] for tag in html.select('a.xil3i')]

			print(date, like, tags)

			dates.append(date)
			likes.append(like)
			tags.append(tags)

## To Do

	- Toy Project : Entertainment Data Analysis
