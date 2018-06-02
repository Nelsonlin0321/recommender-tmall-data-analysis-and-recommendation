# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:54:03 2018

@author: DELL
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
font = {'family' : 'SimHei'} 
matplotlib.rc('font', **font)
data  = pd.read_excel('Excel/Laptop_data_processed.xls')
plt.figure(figsize=(12,6))
Quanlity = data.province.value_counts()


Quanlity.plot(kind='bar')
plt.xticks(rotation= 0)       
plt.xlabel('Province',fontsize = 12)
plt.ylabel('Quantity',fontsize = 12)
plt.title('Province-Quantity Distribution')
plt.show()
df_province_quantity = pd.DataFrame(Quanlity)
df_province_quantity.to_excel('province_quantity.xlsx',index = True,encoding = 'utf-8')

pro_sales = data.pivot_table(index = 'province', values = 'sales', aggfunc=np.mean)    #分类求均值
pro_sales.sort_values('sales', inplace = True, ascending = False)    #排序
pro_sales = pro_sales.reset_index()     #重设索引
 

index = np.arange(pro_sales.sales.size)
plt.figure(figsize=(12,6))
plt.bar(index, pro_sales.sales) 
plt.xticks(index, pro_sales.province, fontsize=11, rotation=0)
plt.xlabel('Province',fontsize = 12)
plt.ylabel('Mean_sales',fontsize = 12)
plt.title('Province-SaleAmount Distribution')
plt.show()
pro_sales.to_excel('pro_sales.xlsx', index = False)