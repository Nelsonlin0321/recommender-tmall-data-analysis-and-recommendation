# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:24:20 2018

@author: DELL
"""

import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = pd.read_excel('Excel/Laptop_data_processed.xls')
row_title_list = data.raw_title.values.tolist()
#new_title = '笔记本电脑 轻薄 吃鸡 Apple' input data
browsed_title = input('The product title the user is browsing is:')

row_title_list.append(browsed_title)

new_title_list = []
for title in row_title_list:
    new_title = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]',title))
    new_title_list.append(new_title)
  
corpus = ['thinkpad','麦本本','小麦5','拯救者','吃鸡','游戏本','商务本','轻薄本','灵越','飞匣',
          '电竞','电竞屏','电竞本','微边框','窄边框','surface','i5','i7','i3','4g','8g','16g','ssd',
          '128g','256g','122英寸','125英寸','116英寸','156英寸','133英寸','173英寸','141英寸','141寸',
          '133','156','116','1代','一代','2代','二代','3代','iii代','三代','4代','四代','5代','五代',
          '6代','六代','7代','七代','8代','八代','13寸','11寸','14寸','15寸','17寸','13英寸','14英寸',
          '15英寸','11英寸','17英寸','dell','ThinkPad','MacBookAir','IdeaPad']
for item in corpus:
    jieba.add_word(item)

title_list = []
for line in new_title_list:
    title_word = jieba.lcut(line)
    title_list.append(title_word)

stop_words = []
with open('Text/Stop words.txt') as f:
  for line in f:
    word = f.readline()
    word = word.strip()
    stop_words.append(word)
    
title_list_filter = []
for line in title_list:
  line = [word for word in line if word not in stop_words 
          and len(word) < 11 and len(word) > 1]
  line = ' '.join(line)
  title_list_filter.append(line)
  
title_input_list = title_list_filter
#添加预测数据：
# 定义向量化参数
#max_df=0.9,min_df=0.1,
tfidf_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,3)) 
new_tfidf_matrix = tfidf_vectorizer.fit_transform(title_input_list)
new_cosine_similarity = cosine_similarity(new_tfidf_matrix)
print('*'*30 +'results'+'*'*30)
print_num = 5
index_num = -2
for i in range(print_num):
    last_vector = new_cosine_similarity[-1] #返回最后一行
    index = last_vector.argsort()[index_num]
    index_num = index_num - 1
    print('recommendation:%s' %(i+1))
    print('index:{}'.format(index))
    print('similarity:{}'.format(new_cosine_similarity[-1][index]))
    recommend_product_url = data.loc[index,'detail_url']
    recommend_product_raw_title = data.loc[index,'raw_title']
    print(recommend_product_raw_title)
    print(recommend_product_url)
    print('_'*50)

#吃鸡 商务 轻薄
#联想 i5 吃鸡游戏本笔记本电脑 Aero 15-V7