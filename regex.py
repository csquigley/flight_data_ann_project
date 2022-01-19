import datetime
from datetime import datetime as dt
from datetime import date
import time
import re

#basic search with error handling
def check(search_params,string):
    try:
        data = re.search(search_params,string)
        if data != None:
            return data.group(0)
        else:
            return None
    except:
        print(search_params)
        print(string)
        raise ValueError(f'ERROR in check func\n{search_params}\n{string}')
def tickets_left(string):
    search_params = "[0-9]+\sleft\sat"
    tickets = check(search_params,string)
    if tickets != None:
        search_params = "[0-9]+"
        tickets = check(search_params,tickets)
        tickets = int(tickets)
    else:
        tickets = "UNKNOWN"

    return [tickets]
def dep_hour(string):
    search_params = "departing\sat\s[0-9]+"
    departure_hour = check(search_params=search_params,string=string)
    if departure_hour != None:
        departure_hour = check("[0-9]+",departure_hour)
        departure_hour = int(departure_hour)
    else:
        departure_hour = None
    return [departure_hour]
def dep_minute(string):
    search_params = "departing\sat\s[0-9]{1,2}:[0-9]+"
    departure_minute = check(search_params=search_params,string=string)
    if departure_minute != None:
        departure_minute = check(search_params=":[0-9]+",string=departure_minute)
        departure_minute = int(departure_minute[1:])
    else:
        departure_minute = None
    return [departure_minute]
def dep_ampm(string):
    search_params = "departing\sat\s[0-9]+:[0-9]+[a-z]+"
    departure_ampm = check(search_params=search_params,string=string)
    if departure_ampm != None:
        departure_ampm = departure_ampm[-2:]
    else:
        departure_ampm = None
    return [departure_ampm]
def carrier(string):
    search_params = "[A-Z]+[\S]+\s{0,1}[\S]{0,}"
    carrier = check(search_params=search_params,string=string)
    if carrier != None:
        carrier = carrier.replace(' flight','')
        if carrier == "Multiple":
            search_params = "•\s[A-Z]+[\S]+\s{0,1}[\S]{0,}"
            c = check(search_params=search_params,string=string)
            if c != None:
                c = c[2:]
                carrier = c
    return [carrier]
def price(string):
    search_params = "\$[1-9],[0-9]+|\$[0-9]{1,3}"
    price = check(search_params=search_params,string=string)
    if price != None:
        price = price.replace(",",'')
        price = int(price[1:])
    else:
        price = None
    return [price]
def arrival_time(string):
    arr_dict = {}
    search_params = "\-\s[0-9]+:[0-9]+am\+{0,1}[0-9]{0,1}|-\s[0-9]+:[0-9]+pm\+{0,1}[0-9]{0,1}"
    arrival_time = check(search_params=search_params,string=string)
    if arrival_time != None:
    #arr_hour - NEEDS ATTENTION
        search_params = "[0-9]+"
        arrival_hour = check(search_params=search_params,string=arrival_time)
        if arrival_hour != None:
            arrival_hour = int(arrival_hour)
        else:
            arrival_hour = None
        arr_dict['arrival_hour'] = [arrival_hour]
        #aarr_minute
        search_params = ":[0-9]+"
        arrival_minute = check(search_params=search_params,string=arrival_time)
        if arrival_minute != None:
            arrival_minute = int(arrival_minute[1:])
        else:
            arrival_minute = None
        arr_dict['arrival_minute'] = [arrival_minute]
        #arr_ampm
        search_params = "[am]+|[pm]+"
        arrival_ampm = check(search_params=search_params,string=arrival_time)
        if arrival_ampm != None:
            arrival_ampm = arrival_ampm
        else:
            arrival_ampm = None
        arr_dict['arrival_ampm'] = [arrival_ampm]
        #arr_plus_days
        search_params = "\+[0-9]+"
        plus_days = check(search_params=search_params,string=arrival_time)
        if plus_days != None:
            plus_days = plus_days[1:]
        else:
            plus_days = 0
        arr_dict['arrival_plus_days'] = [plus_days]
    else:
        arr_dict['arrival_hour'] = [None]
        arr_dict['arrival_plus_days'] = [None]
        arr_dict['arrival_ampm'] = [None]
        arr_dict['arrival_minute'] = [None]
    return arr_dict
def duration_hours(string):
    search_params = "[0-9]+h"
    duration_hours = check(search_params=search_params,string=string)
    if duration_hours != None:
        duration_hours = int(duration_hours[:-1])
    else:
        duration_hours = None
    return [duration_hours]
def duration_minutes(string):
    search_params = "[0-9]+m"
    duration_minutes = check(search_params=search_params,string=string)
    if duration_minutes!= None:
        duration_minutes = int(duration_minutes[:-1])
    else:
        duration_minutes = None
    return [duration_minutes]
def num_stops(string):
    search_params = "[0-9]\sstop"
    stops = check(search_params,string)
    if stops != None:
        search_params = "[0-9]+"
        stops = check(search_params,stops)
        if stops != None:
            stops = int(stops)
    else:
        stops = 0
    return [stops]
def layover(string):
    lay_list = []
    search_params = "[0-9]{0,2}m{0,1}\s{0,1}[0-9]{0,}h{0,1}\s{0,1}[0-9]{0,2}m{0,1}\sin\s[A-Z][a-z]+\s{0,1}[A-Z]{0,1}[a-z]{0,}\s\([A-Z]+\)"
    layover_locations = re.findall(search_params,string)
    t_lay_time = 0
    if len(layover_locations) > 0:
        for li in layover_locations:
            search_params = "\([A-Z]+"
            stop_airport = check(search_params,li)
            stop_airport = stop_airport[1:]
            search_params = "[0-9]+m"
            stop_minutes = check(search_params,li)
            if stop_minutes != None:
                stop_minutes = int(stop_minutes[:-1])
                t_lay_time += (stop_minutes/60)
            else:
                stop_minutes = 0
            search_params = "[0-9]+h"
            stop_hours = check(search_params,li)
            if stop_hours != None:
                stop_hours = int(stop_hours[:-1])
                t_lay_time += stop_hours
            else:
                stop_hours = 0
            lay_list.append(stop_airport)
    else:
        lay_list.append(None)
    return [lay_list],[t_lay_time]
def next_carrier(string):
    search_params = '(?!•\sLayover\sfor)•\s[^0-9]*\s[A-Z]*[a-z]*'
    next_carrier = check(search_params,string)
    if next_carrier != None:
        next_carrier = next_carrier.strip()
        #replace lower case flight because it isn't part of the airline name
        next_carrier.replace(' flight','')
        next_carrier = next_carrier[2:]
    else:
        next_carrier = 'UNKNOWN'
    return [next_carrier]

def stripString(string,dep_city,arr_city,departure_date,search_date,search_rank, print_data=False):
    string = string.rstrip()
    data_dict = {}
    #departure_date
    data_dict['search_rank'] = [search_rank]
    departure_date = departure_date
    data_dict['departure_date'] = [departure_date]
    #departure_city
    departure_city = dep_city
    data_dict['departure_city'] = [departure_city]
    #arrival_city
    arrival_city = arr_city
    data_dict['arrival_city'] = [arrival_city]
    #search_date
    search_date = search_date.strftime("%m/%d/%Y, %H:%M:%S")
    data_dict['search_date'] = [search_date]
    #departure_hour
    data_dict['departure_hour'] = dep_hour(string)
    #departure minute
    data_dict['departure_minute'] = dep_minute(string)
    #departure_ampm
    data_dict['departure_ampm'] = dep_ampm(string)
    #carrier
    data_dict['carrier'] = carrier(string)
    #price
    data_dict['price'] = price(string)
    data_dict['r_tickets'] = tickets_left(string)
    #arrival time (hour,minute,ampm,plus_days)
    arr_time = arrival_time(string)
    data_dict = {**data_dict,**arr_time}
    #duration_hours
    data_dict['duration_hours'] = duration_hours(string)
    #duration_minutes
    data_dict['duration_minutes'] = duration_minutes(string)
    #layover info

    data_dict['num_stops'] = num_stops(string)
    data_dict['layover_locations'],data_dict['layover_duration'] = layover(string)
    if data_dict['num_stops'][0] <= 0:
        data_dict['next_carrier'] = ['NONSTOP']
    else:
        data_dict['next_carrier'] = next_carrier(string)
    if data_dict['carrier'][0] == "Multiple":
        data_dict['multiple'] = True
    else:
        if  data_dict['next_carrier'][0] in [data_dict['carrier'][0],'NONSTOP']:
            data_dict['multiple'] = False
        else:
            data_dict['multiple'] = True
    if print_data == True:
        print(data_dict)
    return data_dict
