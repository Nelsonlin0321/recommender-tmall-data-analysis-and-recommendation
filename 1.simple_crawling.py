# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 10:30:54 2018

@author: DELL
"""

import re
import time
import requests
import pandas as pd
from retrying import retry
from concurrent.futures import ThreadPoolExecutor

start = time.clock()

plist = []
for i in range(1,101):
  j = 44*(i-1) #44 页间隔 数值, 44为单位的分页
  plist.append(j)
  
listno = plist 
#每一个页的编号
datatmsp = pd.DataFrame(columns =[])

headers = {'User-Agent':
  'Mozilla/5.0 (Windows NT 10.9; WOW64)  \
  AppleWebKit 537.36(KHTML, like Gecko)  \
  Chrome/55.0.2883.87 Safari/537.36'}

@retry(stop_max_attempt_number = 8) #获取页面的重试次数为8次
def network_programming(num):#传入需要获取页面的 页编码, 例如 num =44 代表第二页。
#url = 'https://s.taobao.com/search?q=%E6%B2%99%E5%8F%91&imgfile=&commend=all&ssid=s5-e&search_type=item&sourceId=tb.index&spm=a21bo.2017.201856-taobao-item.1&ie=utf8&initiative_id=tbindexz_20170306&sort=sale-desc&filter=reserve_price%5B500%2C%5D&fs=1&filter_tianmao=tmall&bcoffset=0&p4ppushleft=%2C44&s=44'
#页码与参数关系 ：参数 = （页码-1）*44
  url = 'https://s.taobao.com/search?q=笔记本电脑&imgfile=&js=1&stats_click=search_radio_tmall%3A1&initiative_id=staobaoz_20180402&tab=mall&cd=false&ie=utf8&sort=sale-desc&filter=reserve_price%5B1500%2C%5D&bcoffset=0&p4ppushleft=%2C44&s='+str(num)
  #关键词：笔记本电脑， Tmall ， 销量从高到低，最低价格1500
  web = requests.get(url,headers = headers)
  #网址发出请求
  web.encoding = 'utf-8'
  return web


def multihreading():
  number = listno
  #页码列表W
  event = [] 
  #创建一个可以容纳10个task是的线程池
  with ThreadPoolExecutor(max_workers = 10) as executor: #with as 一旦完成，进程池销毁
    for result in executor.map(network_programming,number, chunksize = 10):
      # 用map函数，传入function 函数，参数列表，chunkisze 表示，任务参数序列分为10组，chunksize表示每组中的任务个数
      #map函数，生成按照传入参数顺序的迭代器，生成迭代器的同时，调用network_programming 执行，假如有返回值， 则该进程为返回值。
      #异步执行，但是生成有序列表
      event.append(result) #把每一个线程，添加到event列表中，再调用该列表再对 返回为web页面进行处理。
  return event

listpg = []
event = multihreading()

for i in event :
 json =re.findall('"auctions":(.*?),"recommendAuctions"', i.text)#获取web的文本信息,['[44个商品信息]']
    #淘宝html文本结构分析。
 if len(json):
   table = pd.read_json(json[0]) # 用pandas 读取 json字符串
   datatmsp = pd.concat([datatmsp,table],axis=0, ignore_index = True)
   #获取的table， 以x axis = 0， 拼接
   pg = re.findall('"pageNum":(.*?),"p4pbottom_up"',i.text)[0]
   listpg.append(pg) #获取每一个web 的网页
   
datatmsp.to_excel('Excel/LaptopData.xls',index = False)

end = time.clock()

print('The number of pages accessed:{}'.format(len(listpg)))
print('Use Time:{:.2f}'.format(end-start))