# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 11:18:48 2018

@author: DELL
"""

import pandas as pd
TaobaoData = pd.read_excel('Excel/LaptopData.xls')

#提取指定的列
data = TaobaoData[['raw_title','item_loc','view_price','view_sales','detail_url']]

#添加新的列

data['province'] = data.item_loc.apply(lambda x:x.split()[0])
data['city'] = data.item_loc.apply(lambda x:x.split()[0] if len(x) <4 else x.split()[1])
#print(data['province'])
#print(data['city'])

data['sales'] = data.view_sales.apply(lambda x:x.split('人')[0])
#print(data['sales'])

#浏览数据类型

print('原始的数据类型：\n{}'.format(data.dtypes))

#将数据类型进行转换

data['sales'] = data.sales.astype('int')


#把身份和城市转换为 类别型

data['province'] = data['province'].astype('category')
data['city'] = data['city'].astype('category')

#删除不要的列
data = data.drop(['item_loc','view_sales'],axis = 1)

print('处理之后的数据：\n{}'.format(data.dtypes))

data.to_excel('Excel/Laptop_data_processed.xls',index = False)