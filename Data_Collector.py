import os
from pathlib import Path
import json
import io
from TwitterAPI import TwitterAPI, TwitterOAuth
import pandas
import shutil


o = TwitterOAuth.read_file('skpdragon.twitterapi_credentials')
api = TwitterAPI(o.consumer_key, o.consumer_secret, o.access_token_key, o.access_token_secret)
tweet_count = 500
file_loc = "Data//"+str(tweet_count)+"//"

if not os.path.exists(os.path.dirname(file_loc)):
	os.makedirs(file_loc)

def searchTwitter(query,feed="search/tweets",api=api,n=tweet_count):
  r = []
  qs = 0
  if len(r)==0:
    r.extend([t for t in api.request("search/tweets",{'q':query,'count':n})])
    qs +=1
  while len(r) < n:
    last = r[-1]['id']
    r.extend([t for t in api.request("search/tweets",{'q':query,'count':n,
                                                        'max_id':last})])
    qs += 1
    if qs > 180:
      time.sleep(840)
      qs = 0
  return r[:n]

def checkFileExists(filename):
  return (not (Path(filename).is_file()))

def createFile(filename):
	file = open(filename,'a')
	file.close
	return (Path(filename).is_file())

prefix_list = ["cat","dog","catdog"]
search_criteri = ["#cat -#dog","#dog -#cat","#dog #cat"]
for i in range(len(prefix_list)):
	new_n = 0
	old_n = 0
	prefix = prefix_list[i]
	count_file_name = file_loc + prefix + "_count.txt"
	tweet_file_name = file_loc + prefix + "_tweet" + ".json"
	tweets = searchTwitter(search_criteri[i])
	df = pandas.read_json(json.dumps(tweets))
	txt = [x for x in df['text']]
	new_n = (len(txt))

	if (checkFileExists(count_file_name)):
		old_n = 0
	else:
		count_file = open(count_file_name,'r')
		old_n = int(count_file.read())
		count_file.close()	
	print (old_n,new_n,old_n+new_n)
	count_file = open(count_file_name,mode = 'w')
	tweet_count = old_n+new_n
	tweet_file = open(tweet_file_name,mode = "a+",encoding = "utf-8")
	tweet_file.write(json.dumps(tweets))
	count_file.write(str(tweet_count))
	tweet_file.close()
	count_file.close()
