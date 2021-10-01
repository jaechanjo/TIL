t_search = pd.DataFrame(index = tag_list,
                       columns = ['search_num'])



#페이스북 로그인 페이지로
driver = webdriver.Chrome('./chromedriver')

url = 'https://www.facebook.com/'
driver.get(url)
time.sleep(2)

# 로그인 완료

driver.find_element_by_xpath('//*[@id="email"]').send_keys(fb_id)
time.sleep(2)
driver.find_element_by_xpath('//*[@id="pass"]').send_keys(fb_pw + '\n')
time.sleep(2)

크롤링

l_search = []

for key, i in zip(tag_list[100:], [x for x in range(1,112+1)]):
    
    print(f"{i}번째 {key} 크롤링 중입니다.")
    
    driver.find_element_by_xpath('//*[@id="mount_0_0_Rd"]/div/div[1]/div/div[2]/div[2]/div/div/div/div/div/label').send_keys(Keys.CONTROL, 'a')
    time.sleep(2)
    driver.find_element_by_xpath('//*[@id="mount_0_0_Rd"]/div/div[1]/div/div[2]/div[2]/div/div/div/div/div/label').send_keys(Keys.DELETE)
    time.sleep(2)
    driver.find_element_by_xpath('//*[@id="mount_0_0_Rd"]/div/div[1]/div/div[2]/div[2]/div/div/div/div/div/label').send_keys(f'#{key}' + '\n')
    time.sleep(2)
    
    # 검색량 긁어오기
    
    html = BeautifulSoup(driver.page_source, 'html.parser')
    time.sleep(2)
    text = html.select('div.bi6gxh9e span.d2edcug0')[1].text[:2+1]
    l_search.append(text)
