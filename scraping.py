import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
t=0
#proxy

#proxy = "67.209.67.231"
#port = 5555
# 
#fp = webdriver.FirefoxProfile()
#fp.set_preference('network.proxy.ssl_port', int(port))
#fp.set_preference('network.proxy.ssl', proxy)
#fp.set_preference('network.proxy.http_port', int(port))
#fp.set_preference('network.proxy.http', proxy)
#fp.set_preference('network.proxy.type', 1)
# 
#driver = webdriver.Firefox(firefox_profile=fp)

driver = webdriver.Firefox(executable_path="/home/dipesh/Desktop/geckodriver")

driver.implicitly_wait(20)

url = "https://us.soccerway.com/national/england/premier-league/20172018/regular-season/r41547/"

driver.get(url)

previous = lambda: driver.find_element_by_xpath('//*[@id="page_competition_1_block_competition_matches_summary_5_previous"]')

for i in range(t,38):
    
    if i!=0:
        
        for z in range(i):
            
            time.sleep(2)
            previous().click()
        
        
    print('krree')        
    
    each_match = lambda: driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div[2]/div[2]/table/tbody').find_elements_by_tag_name('tr')
    
    count=driver.find_element_by_xpath("/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div[2]/div[2]/table/tbody").find_elements_by_tag_name('tr')
    
    countt=len(count)
    
    for j in range(countt): 
        print(str(j)+'j')
        
        if i!=0 and j!=0:
        
           
            for k in range(i):
                
                time.sleep(2)
             
                previous().click()
         
            
            
#        each_match()[j].find_elements_by_tag_name('td')[6].click()
        
#        each_match = driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div[2]/div[2]/table/tbody').find_elements_by_tag_name('tr')
        for p in range(15):
            print(str(p)+"fuckk")
            time.sleep(2)
            
            try:
                each_match()[j].find_elements_by_tag_name('td')[6].click()
            
            except TimeoutException:
                
                continue;
         
            print("only fuckk")
            time.sleep(2)
            if(driver.current_url != url):
                break;
#   
        
        driver.execute_script("window.scrollTo(0, 1050)")
        
        
        #driver.get("https://us.soccerway.com/matches/2018/05/13/england/premier-league/liverpool-fc/brighton--hove-albion-fc/2463167/")
        
        hometeam = driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div/div/div[1]/div[1]/h3/a').text
        awayteam = driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div/div/div[1]/div[3]/h3/a').text
        scoreline = driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div/div/div[1]/div[2]/h3').text
        Date = driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div/div/div[2]/div[2]/div[1]/dl/dd[2]/a').text
        Game_week= driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div/div/div[2]/div[2]/div[1]/dl/dd[3]').text
        
        try:
            Attendance = driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div/div/div[2]/div[2]/div[3]/dl/dd[2]').text
        except NoSuchElementException:
            Attendance = ''
#            driver.back()
#            continue
#        Attendance = driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div/div/div[2]/div[2]/div[3]/dl/dd[2]').text
        #home_coach= driver.find_element_by_xpath("/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[5]/div[1]/table/tbody/tr[12]/td/a").text
        #away_coach= driver.find_element_by_xpath("/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[5]/div[2]/table/tbody/tr[12]/td/a").text
        
        
        f=open("premier.csv","a")
        
        f.write(hometeam+','+awayteam+','+Date+','+Game_week+','+Attendance+','+scoreline+',')
        
        #home 5 match
        
        match1=driver.find_element_by_xpath("/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div/div/div[1]/div[1]/div").find_elements_by_tag_name("a")
        for value in match1:
            f.write(value.text+',')
            
        #away 5 match    
        match2= driver.find_element_by_xpath("/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[1]/div/div/div/div[1]/div[3]/div").find_elements_by_tag_name("a")   
        for value in match2:
            f.write(value.text+',')    
            
          
        players=driver.find_element_by_css_selector("table.playerstats.lineups.table").find_element_by_tag_name('tbody').find_elements_by_tag_name("tr")
#            players=driver.find_element_by_xpath("/html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[5]/div[1]/table/tbody").find_elements_by_tag_name("tr")
#                                                    /html/body/div[5]/div[2]/div[2]/div/div/div[2]/div[4]/div[1]/table/tbody
        
        #home players and coach
        for l,value in enumerate(players):
            
             
            if l==12:
                continue
                
            print(str(l)+"what")
           
            
            if l>=0 and l<=10:
                print(str(l)+"vitra")
                f.write(value.find_elements_by_tag_name("td")[1].find_element_by_tag_name("a").text+',')
                
            if(l==11):
                print(str(l)+"bahira")
                
                try:
                    f.write(value.find_element_by_tag_name("td").find_element_by_tag_name("a").text+',')   
            
                except NoSuchElementException:
                    
                    f.write(' '+',') 
               
                
            
        #away players and coach
                
        players2=driver.find_elements_by_css_selector("table.playerstats.lineups.table")[1].find_element_by_tag_name('tbody').find_elements_by_tag_name("tr")
        
        for m,value in enumerate(players2):
            
            if m==12:
                continue
            if(m!=11):
                f.write(value.find_elements_by_tag_name("td")[1].find_element_by_tag_name("a").text+',') 
            if(m==11):
                try:
                     f.write(value.find_element_by_tag_name("td").find_element_by_tag_name("a").text+'\n')      
            
                except NoSuchElementException:
                    
                     f.write(' '+'\n')   
                
                
                 
        
        #print(hometeam)
        #print(awayteam)
        #print(scoreline)
        #print(Date)
        #print(Game_week)
        #print(Attendance)
        #print(home_coach)
        #print(away_coach)
        
        f.close()
        
        next=driver.current_url
        
        for q in range(15):
            
            
            time.sleep(2)
            print('backkk'+str(q))
            try:
                driver.back()
            
            except TimeoutException:
                
                continue;
           
            print("vitra chireko chaina")
            print(driver.current_url)
            print(next)
            if(driver.current_url != next):
                print('successbackk')
                break;
   
        
        
