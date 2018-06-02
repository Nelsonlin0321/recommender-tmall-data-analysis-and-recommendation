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

###############################################################################
# 定义向量化参数
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.1, use_idf=True,ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(title_list_filter) # 向量化关键词
dist = 1 - cosine_similarity(tfidf_matrix)

from sklearn.cluster import KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

#from sklearn.externals import joblib
#joblib.dump(km,  'doc_cluster_k=5.pkl')
#km = joblib.load('doc_cluster_k=5.pkl')
# 保存模型
clusters = km.labels_.tolist()

laptop = { 'url':url_lilst, 'title':title_list_filter, 'cluster': clusters } 
frame = pd.DataFrame(laptop, index = [clusters] , columns = ['url', 'title', 'cluster']) #构建dataframe
# 使用tfidf向量化keyword并且使用kmeans进行clustering
###############################################################################
# 使用mds进行降维， 并且画出K=5时数据的2维可视化结果
import matplotlib.pyplot as plt
 
from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # 使用mds进行降维，保留每个点之间的相对距离
 
xs, ys = pos[:, 0], pos[:, 1] # mds后的2维坐标

cluster_names = {}
cluster_dict = {}
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()
for a in range(num_clusters):
    content = ""
    for ind in order_centroids[a, :3]:
        content += "%s " % terms[ind]
        cluster_names[a] = [content]
        cluster_dict[a] = a
# 得到个类别中频率最高的3个key words
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'} # 设置类别颜色

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(17, 9)) # 设置大小
ax.margins(0.05)# 可选项，只添加 5% 的填充（padding）来自动缩放（auto scaling）
plt.rcParams['font.sans-serif']=['SimHei']
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis= 'x',          # 使用 x 坐标轴
        which='both',      # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
        bottom='off',      # 取消底部边缘（bottom edge）标签
        top='off',         # 取消顶部边缘（top edge）标签
        labelbottom='off')
    ax.tick_params(
        axis= 'y',         # 使用 y 坐标轴
        which='both',      # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
        left='off',      # 取消底部边缘（bottom edge）标签
        top='off',         # 取消顶部边缘（top edge）标签
        labelleft='off')

ax.legend(numpoints=1)  # 图例（legend）中每项只显示一个点
plt.savefig('mds_clusters_small_noaxes_k=5.png', dpi=200)
plt.show() # 展示绘图

from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=1)
pca_pos = pca.fit_transform(dist)  # 使用pca进行降维
pca_xs, pca_ys = pca_pos[:, 0], pca_pos[:,1]  # pca 后的2维坐标

pca_df = pd.DataFrame(dict(x=pca_xs, y=pca_ys, label=clusters))
pca_groups = pca_df.groupby('label')

fig, ax = plt.subplots(figsize=(17, 9)) # 设置大小
ax.margins(0.05)# 可选项，只添加 5% 的填充（padding）来自动缩放（auto scaling）
plt.rcParams['font.sans-serif']=['SimHei']
for name, group in pca_groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis= 'x',          # 使用 x 坐标轴
        which='both',      # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
        bottom='off',      # 取消底部边缘（bottom edge）标签
        top='off',         # 取消顶部边缘（top edge）标签
        labelbottom='off')
    ax.tick_params(
        axis= 'y',         # 使用 y 坐标轴
        which='both',      # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
        left='off',      # 取消底部边缘（bottom edge）标签
        top='off',         # 取消顶部边缘（top edge）标签
        labelleft='off')

ax.legend(numpoints=1)  # 图例（legend）中每项只显示一个点
plt.savefig('pca_clusters_small_noaxes_k=5.png', dpi=200)
plt.show() # 展示绘图


###############################################################################
# 使用ward hierarchy算法绘制树状图

from scipy.cluster.hierarchy import ward, dendrogram
  
linkage_matrix = ward(dist) # 聚类算法处理之前计算得到的距离，用 linkage_matrix 表示
fig, ax = plt.subplots(figsize=(60, 300)) # 设置大小
ax = dendrogram(linkage_matrix, orientation="right", labels=title_list_filter);
 
import matplotlib.pyplot as plt
 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 
plt.tick_params(
        axis= 'x',          # 使用 x 坐标轴
        which='both',      # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
        bottom='off',      # 取消底部边缘（bottom edge）标签
        top='off',         # 取消顶部边缘（top edge）标签
    labelbottom='off')
  
plt.tight_layout() # 展示紧凑的绘图布局
  
# 注释语句用来保存图片
plt.savefig('ward_clusters.png', dpi=200) # 保存图片为 ward_clusters

###############################################################################
#使用新的K=4进行kmeans clustering
new_num_clusters = 4
new_km = KMeans(n_clusters=new_num_clusters)
new_km.fit(tfidf_matrix)
#from sklearn.externals import joblib
#joblib.dump(km,  'doc_cluster_k=4.pkl')
#km = joblib.load('doc_cluster_k=4.pkl')
# 保存模型
new_clusters = new_km.labels_.tolist()
new_laptop = { 'url':url_lilst, 'title':title_list_filter, 'cluster': new_clusters } 
new_frame = pd.DataFrame(laptop, index = [new_clusters] , columns = ['url', 'title', 'new_cluster'])

new_cluster_names = {}
new_cluster_dict = {}
new_order_centroids = km.cluster_centers_.argsort()[:, ::-1]
new_terms = tfidf_vectorizer.get_feature_names()
for a in range(new_num_clusters):
    content = ""
    for ind in new_order_centroids[a, :3]:
        content += "%s " % new_terms[ind]
        new_cluster_names[a] = [content]
        new_cluster_dict[a] = a

new_cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a'}

#使用mds进行降维，并且k=4的kmeans可视化结果
new_df = pd.DataFrame(dict(x=xs, y=ys, label=new_clusters)) 
new_groups = new_df.groupby('label')

fig, ax = plt.subplots(figsize=(17, 9)) # 设置大小
ax.margins(0.05)# 可选项，只添加 5% 的填充（padding）来自动缩放（auto scaling）
plt.rcParams['font.sans-serif']=['SimHei']
for name, group in new_groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=new_cluster_names[name], color=new_cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis= 'x',          # 使用 x 坐标轴
        which='both',      # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
        bottom='off',      # 取消底部边缘（bottom edge）标签
        top='off',         # 取消顶部边缘（top edge）标签
        labelbottom='off')
    ax.tick_params(
        axis= 'y',         # 使用 y 坐标轴
        which='both',      # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
        left='off',      # 取消底部边缘（bottom edge）标签
        top='off',         # 取消顶部边缘（top edge）标签
        labelleft='off')

ax.legend(numpoints=1)  # 图例（legend）中每项只显示一个点
 
plt.savefig('mds_clusters_small_noaxes_k=4.png', dpi=200)
plt.show() # 展示绘图

###############################################################################
#使用pca降维，并且k=4的kmeans可视化结果
new_pca_df = pd.DataFrame(dict(x=pca_xs, y=pca_ys, label=new_clusters)) 
new_pca_groups = new_pca_df.groupby('label')

fig, ax = plt.subplots(figsize=(17, 9)) # 设置大小
ax.margins(0.05)# 可选项，只添加 5% 的填充（padding）来自动缩放（auto scaling）
plt.rcParams['font.sans-serif']=['SimHei']
for name, group in new_pca_groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=new_cluster_names[name], color=new_cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis= 'x',          # 使用 x 坐标轴
        which='both',      # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
        bottom='off',      # 取消底部边缘（bottom edge）标签
        top='off',         # 取消顶部边缘（top edge）标签
        labelbottom='off')
    ax.tick_params(
        axis= 'y',         # 使用 y 坐标轴
        which='both',      # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
        left='off',      # 取消底部边缘（bottom edge）标签
        top='off',         # 取消顶部边缘（top edge）标签
        labelleft='off')

ax.legend(numpoints=1)  # 图例（legend）中每项只显示一个点
 
plt.savefig('pca_clusters_small_noaxes_k=4.png', dpi=200)
plt.show() # 展示绘图

###############################################################################
# =============================================================================
# 根据cosine similarity 进行推荐最相似的item 推荐
# #new_title = '笔记本电脑 轻薄 吃鸡 Apple' input data
# browsed_title = input('The product title the user is browsing is:')
# title_input_list = title_list_filter
# title_input_list.append(browsed_title)
# #添加预测数据：
# # 定义向量化参数
# tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
#                                    min_df=0.1, use_idf=True,ngram_range=(1,3)) 
# new_tfidf_matrix = tfidf_vectorizer.fit_transform(title_input_list)
# new_cosine_similarity = cosine_similarity(new_tfidf_matrix)
# 
# print_num = 5
# index_num = -2
# for i in range(print_num):
#     last_vector = new_cosine_similarity[-1] #返回最后一行
#     index = last_vector.argsort()[index_num]
#     index_num = index_num - 1
#     print('recommendation:%s' %(i+1))
#     print('index:{}'.format(index))
#     print('similarity:{}'.format(new_cosine_similarity[-1][index]))
#     recommend_product_url = data.loc[index,'detail_url']
#     recommend_product_raw_title = data.loc[index,'raw_title']
#     print(recommend_product_raw_title)
#     print(recommend_product_url)
#     print('_'*50)
# 
# #吃鸡 联想 商务 轻薄 苹果
# =============================================================================
############################################################################### 