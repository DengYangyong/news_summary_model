# coding:utf-8
import os
import pathlib

'''
为了代码更简洁，把所有文件路径放在这个文件，且保证跨系统时路径也正确
把哈工大停用词也在这边提前准备好
'''

# 代码的根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 提前准备好停用词
stop_words_path = os.path.join(root,'data','stopwords','哈工大停用词表.txt')
stop_words = [word.strip() for word in open(stop_words_path,encoding='utf-8')]

# 汉语新闻的文件路径
news_path = os.path.join(root,'data','raw_data','zh_news.csv')
# 中文维基的文件路径
zhwiki_path = os.path.join(root,'data','raw_data','zhwiki.json')
# 自定义字典的文件路径
user_dict_path = os.path.join(root,'data','processed_data','user_dict.txt')
# 汉语新闻和中文维基分词后，整合在一起的文件
merged_zhnews_zhwiki_path = os.path.join(root,'data','processed_data','merged_zhnews_zhwiki_seg.csv')

# 为计算句子向量，提前准备好的4个文件
word_id_path =os.path.join(root,'data','processed_data','word_id.json')
id_embed_path =os.path.join(root,'data','processed_data','id_embed.pickle')
word_freq_path =os.path.join(root,'data','processed_data','word_freq.json')
id_weight_path =os.path.join(root,'data','processed_data','id_weight.pickle')

# 把word2vec保存为model和txt两种格式的路径
word2vec_model_path = os.path.join(root,'data','word2vec','wor2vec_300features.model')
word2vec_txt_path = os.path.join(root,'data','word2vec','wor2vec_300features.txt')

 



