from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
import random
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager

from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.common.by import By
import pandas as pd
# 1. Khai báo browser
driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()))

# 2. Mở URL của post
driver.get("https://www.foody.vn/ho-chi-minh/banh-9-sach-banh-sau-rieng-hoang-dieu-2/binh-luan")
sleep(random.randint(5,10))

review_count = driver.find_element("xpath","/html/body/div[2]/div[2]/div[2]/section/div/div/div/div/div[1]/div/div/div[1]/div/div[1]/div/ul/li[1]/a/span")
review_count = int(review_count.text)
click_count = int(review_count/10)
print ("click_count",click_count)
# 3. Lấy link hiện comment
print("show comment link")
for i in range(click_count):
    try:
        showcomment_link = driver.find_element("xpath", "/html/body/div[2]/div[2]/div[2]/section/div/div/div/div/div[1]/div/div/div[1]/div/div[2]/a")
        print(showcomment_link)
        showcomment_link.click()
        print("i = " + str(i))
        sleep(random.randint(5,10))
    except NoSuchElementException:
        print("No Such Element Exception!" + str(i))
        
#acc_link = driver.find_element(By.XPATH, "//div[@class='ru-row']")
print("get link account member")
# get link account member
elems = driver.find_elements(By.CSS_SELECTOR , ".ru-row [href]")
account_name = [elem.text for elem in elems]
# links = [elem.get_attribute('href') for elem in elems]

# get title and content comment
get_titles = driver.find_elements(By.CSS_SELECTOR,"a.rd-title.ng-binding.ng-scope")
titles = [get_title.text for get_title in get_titles]
get_contents = driver.find_elements(By.CSS_SELECTOR,".rd-des span")
contents = [content.text for content in get_contents]

# get score
get_scores = get_contents = driver.find_elements(By.CSS_SELECTOR,".review-points.ng-scope span")
scores = [score.text for score in get_scores]


df1 = pd.DataFrame(list(zip(account_name,titles,contents,scores)), columns = ['Tên tài khoản bình luận về nhà hàng này', 'Tiêu đề bình luận','Nội dung bình luận','Điểm đánh giá'])
df1.to_excel('data2.xlsx', index=True)

# 6. Đóng browser
driver.close()
