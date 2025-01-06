# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:46:29 2025

@author: Pmondou
"""

from binance.client import Client
import datetime


from apiKeys import apiSecretKey
from apiKeys import apiKey

symbol = 'BTCUSDT'
start_time = datetime.datetime(2024, 9, 15, 0, 0, 0)
end_time = datetime.datetime(2025, 1, 5, 0, 0, 0)

binanceClient = Client(apiKey, apiSecretKey)
print("loading")
#klines = binanceClient.get_historical_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, start_str=str(start_time), end_str=str(end_time))

klines = binanceClient.get_historical_klines("BTCUSDT", "5m", "50475 minutes˓→ago UTC")
print(len(klines))