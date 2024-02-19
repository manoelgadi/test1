# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:33:51 2023

@author: manoe
"""
import time
import datetime
from binance import Client # pip install python-binance
import pandas as pd
import os
client = Client()


dir = '_data'
timeframes = ['1min', '5min', '15min', '1h', '4h', '1d', '1w']
assets = ['BTC','ETH','SOL','ADA','MATIC','DOT','UNI','FTM','LINK','XRP','ATOM','AVAX','HBAR','NEAR','ALGO','VET','ONE','DOGE',
          'ENJ','XTZ','BNB','TRX','LTC','SHIB','XLM','XMR','FIL','ARB','QNT','SAND','AXS','MANA','RNDR','PEPE','GALA','KLAY',
          'CAKE','LRC','QTUM','GMT','YFI','WAVES','FET','GLMR','AAVE']

if not dir in os.listdir():
    os.mkdir(dir)
for tf in timeframes:
    if not tf in os.listdir(dir):
        os.mkdir(dir+'/'+tf)

def get_minute_data(symbol, interval, start_str, end_str):
    """
    Retrieve minute-level historical data for a given trading symbol.
    Returns:
        pandas.DataFrame: A DataFrame containing the historical minute-level OHLCV data for the specified symbol.
    """
    df = pd.DataFrame(client.get_historical_klines(symbol='BTCUSDT', interval=interval, start_str=start_str,end_str=end_str))
    df = df.iloc[:-1, :6]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index, unit='ms')
    df = df.astype(float)
    return df


for asset in assets:
    start_dt = datetime.date(2023, 10, 10)
    data_1min = False
    
    while (start_dt < datetime.date.today()):
        print(start_dt)
        
        symbol = asset+'USDT'
        
        end_dt = start_dt + datetime.timedelta(days=1)
        data = get_minute_data(symbol=symbol, interval='1m', start_str=str(start_dt),end_str=str(end_dt))
        try:
            data_1min = pd.concat([data_1min, data], axis=0)
        except:
            data_1min = data
        path = dir+'/1min'
        data_1min.to_csv(f'{path}/{symbol}.csv')    
        start_dt = end_dt
    time.sleep(1)


#df = pd.DataFrame(client.get_historical_klines(symbol='BTCUSDT', interval='1m', start_str=str(start_dt),end_str=str(end_dt)))