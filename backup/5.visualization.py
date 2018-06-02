# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 11:53:53 2018

@author: DELL
"""
import pandas as pd
data = pd.read_excel('Excel/Laptop_data_processed.xls')

price_limit = 15000

#数据的截取，只截取 价格小于15000的 商品
data = data[data.view_price<price_limit]
data['Revenue'] = data['view_price']*data['sales']


import matplotlib.pyplot as plt

fig = plt.figure(figsize = (40,30))
plt.xticks(range(0,price_limit,1000))
#plt.yticks(fontsize = 30)

ax = fig.add_subplot(2,1,1)
ax.scatter(data['view_price'],data['sales'],s= 30,alpha = 0.5)
ax.set_title('Sales and Price Scatter Chart',fontsize = 25)
ax.set_ylabel('Sales Amount',fontsize = 20)
ax.set_xlabel('Price',fontsize = 20)



ax2= fig.add_subplot(2,1,2)
ax2.scatter(data['view_price'],data['view_price']*data['sales'],s= 30,alpha = 0.5)
ax2.set_title('Price and Revenue',fontsize = 25)
ax2.set_ylabel('Revenu(ten million)',fontsize = 20)
ax2.set_xlabel('Price',fontsize = 20)


fig.show()

#print(price_revenue['revenue'].max())

MatrixData = pd.DataFrame()
MatrixData['price'] = data['view_price']
MatrixData['sales'] = data['sales']
MatrixData['Revenue'] = data['view_price']*data['sales']

pd.plotting.scatter_matrix(MatrixData,marker = 'o',figsize = (40,30))

MaxRevenue = data['Revenue'].max()
print('The Maximun Revenue:{}'.format(MaxRevenue))
url = data.loc[data.Revenue >= MaxRevenue,['detail_url','raw_title']]
url = str(url)

with open('the top merchant.txt','w',encoding = 'utf-8') as f:
  f.write(url)
  
print('The Merchant with the maximun revenue:\n{}'.format(url))
  


