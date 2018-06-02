# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:53:09 2018

@author: DELL
"""
import pandas as pd
import re
data  = pd.read_excel('Excel/Laptop_data_processed.xls')

title_list = data.raw_title.values.tolist()

new_title_list = []
#对每一个title进行提取清洗, 删除所有符号
for title in title_list:
  new_title = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]',title))
  new_title_list.append(new_title)

  
import jieba
#添加自定义中文词
corpus = ['吃鸡','ThinkPad','MacBookAir','IdeaPad'] 
for word in corpus:
  jieba.add_word(word)


#分词
title_list= [] 
for title in new_title_list:
  title_cut = jieba.lcut(title) #list
  title_list.append(title_cut)
  
#创建停用词
stop_words = []
with open('Text/Stop words.txt') as f:
  for line in f:
    word = f.readline()
    word = word.strip()
    stop_words.append(word)
stop_words.append('笔记本电脑')
stop_words.append('156')
stop_words.append('14')

title_filter_list = []
#去除停用词
for title in title_list:
  title = [word for word in title if word not in stop_words
           and len(word) < 11 and len(word) > 1]
  title = [word for word in title if not isinstance(word,int)]
  title_filter_list.append(title)
  
  

#对一个标题下的重复词语要进行去重
title_list_unique = []
for title in title_filter_list:
  title_unique = []
  for word in title:
    if word not in title_unique:
      title_unique.append(word)
  title_list_unique.append(title_unique)

#把title_list_unique 转化为一个list：
allwords = []
for title in title_list_unique:
  for word in title:
    allwords.append(word)

#转化为DataFrame
allwords_df = pd.DataFrame({'allwords':allwords})

word_count = allwords_df.allwords.value_counts().reset_index()


word_count.columns = ['word','count']

import numpy as np
sales_sum_list = [] #不同商品 各种不同的总销售量

for word in word_count.word: #没word_count 进行遍历
  i = 0
  word_sales_list = []
  
  for title in title_list_unique: #对每一个title 列表 进行遍历
    if word in title: 
      word_sales_list.append(data.sales[i])
    i+=1# 第几次循环对应data 中对应第几个 sales
    #title_list_unique 与data 的元数据中的 顺序一致
  sales_sum_list.append(sum(word_sales_list)) #每一个word 加一个列表
  
#创建 sales_sum_list 的DataFrame
  
sales_sum_df = pd.DataFrame({'sales_sum':sales_sum_list})
#把word_count 和 saels_sum_list 合并

word_sum_df = pd.concat([word_count,sales_sum_df],axis = 1,ignore_index = True)
word_sum_df.columns = ['word','count','sales_sum']


word_sum_df.sort_values('sales_sum', inplace = True , ascending = True) 
#升序

#输出
word_sum_df.to_excel('word_sum.xls',index = False)


#进行可视化

df_top_30 = word_sum_df.tail(30)

df_top_30.to_excel('top_30.xls',index = False)


import matplotlib.pyplot as plt
import matplotlib


index = np.arange(df_top_30.word.size)
font = {'family' : 'SimHei'} 
matplotlib.rc('font', **font) 
plt.figure(figsize = (12,12))
plt.barh(index,df_top_30.sales_sum,
         align = 'center',
         alpha = 0.8)

plt.yticks(index ,df_top_30.word, fontsize = 11)

for y, x in zip(index,df_top_30.sales_sum):
  plt.text(x,y, '%.0f' %x,
           ha = 'left', va = 'center',fontsize = 11)
plt.title('Words:Total Sales Amount')
plt.show()  


  
  

