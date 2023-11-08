import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

csv_file_path = 'MTX00_data.txt'
data = pd.read_csv(csv_file_path)

date_time =  data.iloc[:, 0].tolist()
open_price =  data.iloc[:, 1].tolist()
high_price =  data.iloc[:, 2].tolist()
low_price =  data.iloc[:, 3].tolist()
close_price =  data.iloc[:, 4].tolist()
quantity =  data.iloc[:, 5].tolist()
days_1_min_k = 1139  # 一個交易日共1139根1分k

def trans_time_to_datetime_format():
    for i in range(len(date_time)):
        dt_object = datetime.strptime(date_time[i], "%Y/%m/%d %H:%M")
        date_time[i] = dt_object



def del_unmarking_times():  # https://www.thinkmarkets.com/tw/learn-to-trade/futures/futures-(1)/taiwan-index-futures/
    for index, row in data.iterrows():
        date_obj = datetime.strptime(row['date_time'], '%Y/%m/%d %H:%M')
        # 取出小時和分鐘
        hour = date_obj.hour
        minute = date_obj.minute
        time = int(hour) * 60 + int(minute)
        morning_start = 8 * 60 + 45     # 8:45
        morning_end = 13 * 60 + 45 # 13:20
        evening_start = 15 * 60  # 15:00
        evening_end = 5 * 60  # 5:00
        
        if((evening_end < time < morning_start) or (morning_end < time < evening_start)):
            print("超出交易時間")
            print(row['date_time'])
        
        
def cac_price_indicator(price_data): # 標記未來20分鐘，價格是否> close_price if >= 25 : 1  if <= 25 : 2 else 0  0不變1做多2做空

    open_prices = price_data['Open'].tolist()
    high_prices = price_data['High'].tolist()
    low_prices = price_data['Low'].tolist()
    close_prices = price_data['Close'].tolist()
    price_indicator = []
    for i in range(len(close_prices)-21):
        index = 0
        for j in range(1,21): # 未來20分鐘 close_prices[i] = 現在時間
            if((open_prices[j+i] - close_prices[i]) >= 25): # 未來20分鐘內開盤價格 >= 25
                index = 1
                break
            elif((open_prices[j+i] - close_prices[i]) <= -25): # 未來20分鐘內開盤價格 <= 25
                index = 2
                break
            elif(((high_prices[j+i] - close_prices[i]) >= 25) and ((low_prices[j+i] - close_prices[i]) <= -25)): # 未來20分鐘內其中一分鐘最大價格>25最低價又<25
                index = 0
                break
            elif((high_prices[j+i] - close_prices[i]) >= 25): # 未來20分鐘內最大價格 >= 25
                index = 1
                break
            elif((low_prices[j+i] - close_prices[i]) <= -25): # 未來20分鐘內最低價格 <= 25
                index = 2
                break
            
        price_indicator.append(index)
    
    while len(price_indicator) < len(price_data):
        price_indicator.append(0)
    return price_indicator



def cac_time_divisor(price_data): # 切割時間 8:45-09:45 = 0  ,9:45-14:00 = 1  ,15:00-20:50 = 2   ,20:50-21:50 = 3, 21:50-5:00 = 4
    _8_45  = 8  * 60 + 45 # 8:45
    _9_45  = 9  * 60 + 45 # 9:45
    _14_00 = 14 * 60      # 14:00
    _15_00 = 15 * 60      # 15:00
    _20_50 = 20 * 60 + 50 # 20:50
    _21_50 = 21 * 60 + 50 # 21:50
    _05_00 = 5 * 60      # 5:00
    time_index = []
    for index, row in data.iterrows():
        date_obj = datetime.strptime(row['date_time'], '%Y/%m/%d %H:%M')
        # 取出小時和分鐘
        hour = date_obj.hour
        minute = date_obj.minute
        time = int(hour) * 60 + int(minute)
        
        if(time >= _8_45 and time <= _9_45):
            time_index.append(0)
        elif(time >= _9_45 and time <= _14_00):
            time_index.append(1)
        elif(time >= _15_00 and time <= _20_50):
            time_index.append(2)
        elif(time >= _20_50 and time <= _21_50):
            time_index.append(3)
        elif(time >= _21_50 or time <= _05_00):
            time_index.append(4)
        else:
            time_index.append(5)
        
    return time_index




def cac_data():
    window_size = 600 # 100分鐘平均
    price_data = pd.DataFrame({'date_time': date_time})
    price_data['Open'] = data['open_price']
    price_data['High'] = data['high_price']
    price_data['Low'] = data['low_price']
    price_data['Close'] = data['close_price']
    price_data['quantity'] = data['quantity']
    price_indicator = cac_price_indicator(price_data)
    price_data['price_indicator'] = price_indicator
    time_divisor = cac_time_divisor(price_data)
    price_data['time_divisor'] = time_divisor
    # price_data = pd.DataFrame({'Quantity_n': quantity})
    # 計算移動平均和標準差
    price_data['Moving_Avg'] = price_data['Close'].rolling(window=window_size).mean()
    price_data['Std_Dev'] = price_data['Close'].rolling(window=window_size).std()
    # 計算布林通道的上線和下線
    price_data['Upper_Band'] = price_data['Moving_Avg'] + (price_data['Std_Dev'] * 1)
    price_data['Lower_Band'] = price_data['Moving_Avg']  - (price_data['Std_Dev'] * 1)
    
    price_data['5_Days'] = price_data['Close'].rolling(window = days_1_min_k * 5).mean() # 5日均線
    price_data['5_Days_slope'] = price_data['5_Days'] - price_data['5_Days'].shift(1)
    price_data['60_mins'] = price_data['Close'].rolling(window = 180).mean() # 180分均線
    price_data['60_mins_slope'] = price_data['60_mins'] - price_data['60_mins'].shift(1)
    price_data['5_mins'] = price_data['Close'].rolling(window = 5).mean() # 5分均線
    price_data['5_mins_slope'] = price_data['5_mins'] - price_data['5_mins'].shift(1)
    price_data['5_mins_quantity_Avg'] = price_data['quantity'].rolling(window = 5).mean() # 5分鐘平均交易量





    price_data.to_csv('MTX00_result_data.csv', index=False)
    
    return price_data
    

def plot(price_data):
    plt.figure(figsize=(12,6))
    plt.plot(price_data['Close'], label='Close Price', linewidth=1.0)  # 調整線寬
    plt.plot(price_data['Moving_Avg'], label='Moving Average', linewidth=0.5)  # 調整線寬
    plt.plot(price_data['Upper_Band'], label='Upper Bollinger Band', linewidth=0.5)  # 調整線寬
    plt.plot(price_data['Lower_Band'], label='Lower Bollinger Band', linewidth=0.5)  # 調整線寬
    plt.fill_between(price_data.index, price_data['Lower_Band'], price_data['Upper_Band'], color='gray', alpha=0.3)
    plt.title('Bollinger Bands')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

        
        



trans_time_to_datetime_format()
del_unmarking_times() # 目前用不到，歷史資料沒有包含盤前搓合
price_data = cac_data()
plot(price_data)