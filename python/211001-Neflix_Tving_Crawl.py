tving = []
netflix = []

for i in range(1,37+1):
    
    if i in [17, 33]:
        time.sleep(2)
        driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
        time.sleep(3)
    
    print(f"{i}번째 크롤링 중입니다.")
    
    #들어가기
    driver.find_element_by_xpath(f'//*[@id="contents"]/div[2]/div/div[{i}]').click()

    time.sleep(2)
    
    # 티빙

    driver.find_element_by_xpath('//*[@id="triggerMenu"]/div/div[1]/div/div/button[4]').click() #누르고
    time.sleep(2)
    
    html = BeautifulSoup(driver.page_source, 'html.parser')
    for name in html.select('div.title'):
        tving.append(name.text.strip())
    
    time.sleep(2)
    driver.find_element_by_xpath('//*[@id="triggerMenu"]/div/div[1]/div/div/button[4]').click() #끄고

    # 넷플릭스
    
    time.sleep(2)
    driver.find_element_by_xpath('//*[@id="triggerMenu"]/div/div[1]/div/div/button[1]').click() #누르고
    time.sleep(2)
    
    html = BeautifulSoup(driver.page_source, 'html.parser')
    
    for name in html.select('div.title'):
        netflix.append(name.text.strip())
    
    time.sleep(2)
    driver.find_element_by_xpath('//*[@id="triggerMenu"]/div/div[1]/div/div/button[1]').click() #끄고
    time.sleep(2)
    
    driver.find_element_by_xpath(f'//*[@id="header"]/div/button[1]').click() #나오고
    time.sleep(2)
