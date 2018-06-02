# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 10:49:00 2018

@author: DELL
"""
import pandas as pd
TaobaoData = pd.read_excel('Excel/LaptopData.xls')

import missingno as msno

msno.bar(TaobaoData.sample(len(TaobaoData)),figsize = (10,4))

half_count = len(TaobaoData)/2
#len(TaobaoData) 有 44 *100 条数据
print(half_count)

#删除缺失值过磅的列
TaobaoData_half = TaobaoData.dropna(thresh = half_count,axis = 1)

#删除重复行
TaobaoData_half = TaobaoData_half.drop_duplicates()

msno.bar(TaobaoData_half.sample(len(TaobaoData_half)),figsize = (10,4))
