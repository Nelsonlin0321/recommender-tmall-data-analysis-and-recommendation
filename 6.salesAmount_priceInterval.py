# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:00:02 2018

@author: DELL
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data  = pd.read_excel('Excel/Laptop_data_processed.xls')

data['price'] = data.view_price.astype('int')   #转为整型  

# 用 qcut 将price列分为12组
data['group'] = pd.qcut(data.price, 25)
        
df_group = data.group.value_counts().reset_index()   #生成数据框并重设索引 

# 以group列进行分类求sales的均值：
df_s_g = data[['sales','group']].groupby('group').mean().reset_index()  

# 绘柱形图：
index = np.arange(df_s_g.group.size)
plt.figure(figsize=(16,8))
plt.bar(index, df_s_g.sales)     
plt.xticks(index, df_s_g.group, fontsize=11, rotation=30) 
plt.xlabel('Group')
plt.ylabel('mean_sales')
plt.title('SaleAmount_PriceInterval')
plt.show()