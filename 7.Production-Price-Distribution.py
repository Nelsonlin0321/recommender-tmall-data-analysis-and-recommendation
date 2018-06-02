# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:26:06 2018

@author: DELL
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data  = pd.read_excel('Excel/Laptop_data_processed.xls')
data_p = data[data['view_price'] < 25000]    

plt.figure(figsize=(14,8))
sns.distplot(data_p['view_price'],kde = True)
#plt.hist(data_p['view_price'] ,bins=20)   #分为15组  
plt.xlabel('价格',fontsize=12)
plt.ylabel('Product Density',fontsize=12)         
plt.title('KDA-Production-Price-Distribution',fontsize=15)  
plt.show()

plt.figure(figsize=(14,8))
#sns.distplot(data_p['view_price'],kde = False,fit=stats.gamma)
plt.hist(data_p['view_price'] ,bins=20)   #分为15组  
plt.xlabel('Price',fontsize=12)
plt.ylabel('Product Quantity',fontsize=12)         
plt.title('Production-Price-Distribution',fontsize=15)

plt.show()