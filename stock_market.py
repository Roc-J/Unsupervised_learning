# -*- coding:utf-8 -*- 
# Author: Roc-J

import json
import numpy as np
import datetime
from sklearn import covariance, cluster
from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_yahoo

symbol_file = 'symbol_map.json'
with open(symbol_file, 'r') as f:
    symbol_dict = json.loads(f.read())

symbols, names = np.array(list(symbol_dict.items())).T

#
start_date = datetime.datetime(2004, 4, 5)
end_date = datetime.datetime(2007, 6, 2)

quotes = [quotes_yahoo(symbol, start_date, end_date, asobject=True) for symbol in symbols]

# 开盘价
opening_quotes = np.array([quote.open for quote in quotes]).astype(np.float)
closing_quotes = np.array([quote.close for quote in quotes]).astype(np.float)

delta_quotes = closing_quotes - opening_quotes

# 建立一个协方差
edge_model = covariance.GraphLassoCV()

# 数据标准化
X = delta_quotes.copy().T

X /= X.std(axis=0)

with np.errstate(invalid='ignore'):
    edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

for i in range(num_labels+1):
    print "Cluster", i+1, "-->", ', '.join(names[labels == i])