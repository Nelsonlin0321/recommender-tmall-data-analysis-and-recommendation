# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:34:35 2018

@author: past
"""

import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_excel('Excel/Laptop_data_processed.xls')
row_title_list = data.raw_title.values.tolist()
url_lilst = data.raw_title.values.tolist()

new_title_list = []
for title in row_title_list:
    new_title = "".join(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]',title))
    new_title_list.append(str.lower(new_title))
  
corpus = ['thinkpad','麦本本','小麦5','拯救者','吃鸡','游戏本','商务本','轻薄本','灵越','飞匣',
          '电竞','电竞屏','电竞本','微边框','窄边框','surface','i5','i7','i3','4g','8g','16g','ssd',
          '128g','256g','122英寸','125英寸','116英寸','156英寸','133英寸','173英寸','141英寸','141寸',
          '133','156','116','1代','一代','2代','二代','3代','iii代','三代','4代','四代','5代','五代',
          '6代','六代','7代','七代','8代','八代','13寸','11寸','14寸','15寸','17寸','13英寸','14英寸',
          '15英寸','11英寸','17英寸']
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


# 定义向量化参数
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, use_idf=True,ngram_range=(1,3))
 
tfidf_matrix = tfidf_vectorizer.fit_transform(title_list_filter) # 向量化关键词
print(title_list_filter)
print(tfidf_matrix)
terms = tfidf_vectorizer.get_feature_names()
print(terms)

dist = 1 - cosine_similarity(tfidf_matrix)

from sklearn.cluster import KMeans
from sklearn.externals import joblib

num_clusters = 5
 
km = KMeans(n_clusters=num_clusters)
 
km.fit(tfidf_matrix)

#joblib.dump(km,  'C:/Users/past/Desktop/7630 Web intelligence and its application/Group_project/laptop/doc_cluster.pkl')
#km = joblib.load('C:/Users/past/Desktop/7630 Web intelligence and its application/Group_project/laptop/doc_cluster.pkl')
clusters = km.labels_.tolist()


laptop = { 'url':url_lilst, 'title':title_list_filter, 'cluster': clusters }
 
frame = pd.DataFrame(laptop, index = [clusters] , columns = ['url', 'title', 'cluster'])

new_title = '笔记本电脑 轻薄 吃鸡 商务'
#input_data processing

new_title = jieba.lcut(new_title)


new_title_words_list = [word for word in new_title if word not in stop_words 
        and len(word) < 11 and len(word) > 1]

#把新的title 加入到词库中
predict_format = []
predict_format.extend(terms)
predict_format.extend(new_title_words_list)

input_title = tfidf_vectorizer.fit_transform(predict_format)
predict_cluster = km.predict(input_title)
print(predict_cluster)

  


