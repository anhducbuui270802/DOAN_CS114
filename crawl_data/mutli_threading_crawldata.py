from selenium import webdriver
from time import sleep
from selenium.webdriver.common.keys import Keys
import random
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import threading
from queue import Queue
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.common.by import By
import pandas as pd


# Khai báo luồng
n = 6
global filename 
filename = 1


# ===============================Define Logic==================================

def openMultiBrowsers(n):
    drivers = []
    for i in range(n):
        driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()))        
        drivers.append(driver)
    return drivers

def loadMultiPages(driver, link,i):
    # for driver in drivers:
    driver.maximize_window()
    driver.get(link[i])
    sleep(6)

def loadMultiBrowsers(drivers_rx, n):
    i = 0
    for driver in drivers_rx:
        t = threading.Thread(target=loadMultiPages, args = (driver, n,i))
        i += 1
        t.start()
        
        
def getData(driver):
    global filename 
    # 1. Khai báo browser
    # driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()))

    # 2. Mở URL của post
    # driver.get("https://www.foody.vn/ho-chi-minh/banh-cuon-gao-lut-74-vu-tong-phan/binh-luan")
    # sleep(random.randint(5,10))

    review_count = driver.find_element("xpath","/html/body/div[2]/div[2]/div[2]/section/div/div/div/div/div[1]/div/div/div[1]/div/div[1]/div/ul/li[1]/a/span")
    review_count = int(review_count.text)
    click_count = int(review_count)
    # 3. Lấy link hiện comment
    print("show comment link")
    for i in range(click_count):
        try:
            showcomment_link = driver.find_element("xpath", "/html/body/div[2]/div[2]/div[2]/section/div/div/div/div/div[1]/div/div/div[1]/div/div[2]/a")
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
    print("get comment")
    get_titles = driver.find_elements(By.CSS_SELECTOR,"a.rd-title.ng-binding.ng-scope")
    titles = [get_title.text for get_title in get_titles]
    get_contents = driver.find_elements(By.CSS_SELECTOR,".rd-des span")
    contents = [content.text for content in get_contents]

    # get score
    get_scores = get_contents = driver.find_elements(By.CSS_SELECTOR,".review-points.ng-scope span")
    scores = [score.text for score in get_scores]


    df1 = pd.DataFrame(list(zip(account_name,titles,contents,scores)), columns = ['Tên tài khoản bình luận về nhà hàng này', 'Tiêu đề bình luận','Nội dung bình luận','Điểm đánh giá'])
    df1.to_excel('data/data_{}.xlsx'.format(filename), index=True)
    filename += 1

    # 6. Đóng browser
    driver.close()

def runInParallel(func, drivers_rx):
    for driver in drivers_rx:  
        que = Queue()
        print("-------Running parallel---------")
        t1 = threading.Thread(target=lambda q, arg1: q.put(func(arg1)), args=(que, driver))
        t1.start()
    # try:    
    #     ouput = que.get()
    # except:
    #     ouput = [] 

    # return ouput


# ===========================Run/Execute=======================================
# Đường dẫn đến file Excel chứa danh sách đường link
excel_file = 'links.xlsx'

# Đọc file Excel vào một DataFrame
df = pd.read_excel(excel_file)

# print(df.info())

# Chuyển cột 'Link' thành một mảng
links_array = df['Link'].values

for i in range((len(links_array)//n)+1):
    drivers_r1 = openMultiBrowsers(n)
    loadMultiBrowsers(drivers_r1, links_array[n*i:n*i+n])  
    sleep(10)

    # Tạo danh sách luồng
    threads = []

    # Chạy 2 luồng cùng một lúc
    for j in range(n):
        t = threading.Thread(target=getData, args=(drivers_r1[j],))
        t.start()
        threads.append(t)

    # Đợi cho tất cả các luồng hoàn thành công việc
    for t in threads:
        t.join()

# drivers_r1 = openMultiBrowsers(4)
# loadMultiBrowsers(drivers_r1, links_array)  
# sleep(10)

# # ===== GET link/title

# title_link2 = runInParallel(getData, drivers_r1)