# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:31:34 2018

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

#allwords_df.to_excel('allword.xls',index = False)
#allwords_df.allwords.value_counts().to_excel('value_counts.xls',index = True)

word_count = allwords_df.allwords.value_counts().reset_index()
word_count.to_excel('word_count_without_index.xls',index = False)

word_count.columns = ['word','count']
print(type(word_count))
word_count.to_excel('word_count.xls',index = False)

from wordcloud import WordCloud
from scipy.misc import imread
pic = imread("images/laptop_background.jpg")

import matplotlib.pyplot as plt
plt.figure(figsize = (40,50))

word_cloud = WordCloud(font_path = "font/simhei.ttf",
                       max_font_size= 300, 
                       margin=1,
                       mask = pic,
                       background_color ='white')
wc = word_cloud.fit_words({x[0]:x[1] for x in word_count.head(1000).values})

plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()




  

