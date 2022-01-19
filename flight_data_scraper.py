import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import datetime
from datetime import datetime as dt
from datetime import date
import time
import re
import csv
from regex import stripString

routes = []
with open('links.csv') as links_file:
    csv_reader = csv.reader(links_file,delimiter=",")
    for row in csv_reader:
        new_route = []
        new_route.extend([row[0],row[1],row[2]])
        routes.append(new_route)
#Uncomment if you want a new  file
# f = open('flight_data.csv','w')
# f.close()


options = {
    'freq': 6,
    'delay':40,
    'days_to_check': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,21,28,42,56]
}
driver = uc.Chrome()
while True:
    print("Running")
    for link in routes:
        from_city = link[0]
        to_city = link[1]
        today = date.today()
        for day in options['days_to_check']:
            d = today + datetime.timedelta(days=day)
            l = link[2].format(month=d.month,day=d.day,year=d.year)
            time.sleep(10)
            driver.get(l)
            try:
                WebDriverWait(driver,options['delay']).until(EC.presence_of_element_located((By.XPATH,"//li[@data-test-id='offer-listing']")))
                time.sleep(5)
            except TimeoutException:
                print('loading took too long')
            flights = driver.find_elements(By.XPATH,"//li[@data-test-id='offer-listing']")
            current_datetime = dt.now()
            for n,flight in enumerate(flights):
                data = stripString(flight.text,from_city,to_city,d,current_datetime,print_data=False,search_rank=n)
                if data != None:
                    df = pd.DataFrame.from_dict(data)
                    df.to_csv(path_or_buf='flight_data.csv',mode='a', sep=',', na_rep='', float_format=None, header=False, index=True, index_label=None, encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.')
                else:
                    print(data)

            time.sleep(30)
    print("Sleeping")
    time.sleep(60*60*options['freq'])
