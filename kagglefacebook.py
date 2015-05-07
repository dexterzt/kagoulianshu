import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn import cross_validation
from sklearn import tree

#read file
train=pd.read_csv('train.csv', header=0)
test=pd.read_csv('test.csv', header=0)
bids=pd.read_csv('bids.csv', header=0)
train.groupby(train['outcome']).count() #class distribution
bids_null=bids.isnull() #country has null values


"""bid frequency
number of ip
number of country
number of referrer link
number of different auctions
number of device
"""
num_ip=bids['ip'].groupby(bids['bidder_id']).nunique()
num_country=bids['country'].groupby(bids['bidder_id']).nunique(dropna=False)
num_url=bids['url'].groupby(bids['bidder_id']).nunique()
num_device=bids['device'].groupby(bids['bidder_id']).nunique()
num_auction=bids['auction'].groupby(bids['bidder_id']).nunique()
num_bid=bids['bid_id'].groupby(bids['bidder_id']).nunique() #naturally unique
num_time=bids['time'].groupby(bids['bidder_id']).nunique()
timeg=bids['time'].groupby(bids['bidder_id'])
avg_time=(timeg.max()-timeg.min())/(timeg.size())
num_ip.name='num_ip'
num_country.name='num_country'
num_url.name='num_url'
num_device.name='num_device'
num_auction.name='num_auction'
num_bid.name='num_bid'
num_time.name='num_time'
avg_time.name='avg_time'

new_features=pd.concat([num_ip, num_country, num_url, num_device, num_auction, num_bid, num_time, avg_time], axis=1)
train_full=pd.merge(train, new_features, left_on='bidder_id', right_index=True, how='left')
test_full=pd.merge(test, new_features, left_on='bidder_id', right_index=True, how='left')

"""merchandise: factor of 10
country: factor of 199 and NaN
time frequency
"""
