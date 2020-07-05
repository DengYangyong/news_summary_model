# coding:utf-8
import jieba,json
import pandas as pd
from config import *
from multi_proc_utils import parallelize
from collections import Counter
from itertools import chain
from data_utils import *
import gc
    
def generate_data():
    '''
    对中文维基的文章和汉语新闻进行分词，然后整合起来，保存为一个文件，用于训练word2vec
    观察分词的结果发现，中文维基里有些文章的分词结果不太好，需要把title字段整理为自定义字典。
    此外，为了提高分词的速度，用了多进程，cpu有多少核，就开多少个进程，充分利用计算资源。
    :return:
    '''
    zhwiki_file = open(zhwiki_path,encoding='utf-8')
    text_list = []
    for line in zhwiki_file:
        if not line.strip():
            continue
        article = json.loads(line)
        text = article['text'].strip().replace('\n\n','')
        text_list.append(text)
    zhwiki_texts = pd.DataFrame(data=text_list, columns=['text'])
    ''''开多进程进行分词'''
    zhwiki_texts_seg = parallelize(seg_df,zhwiki_texts)     
    
    zh_news = pd.read_csv(news_path,encoding='gb18030')
    zh_news.dropna(inplace=True,subset=['title','content'])
    '''只保留title和content两个字段的内容，并且拼接成一篇文章，以便于和中文维基的数据进行整合'''
    zh_news['text'] = zh_news[['title','content']].apply(lambda x: ' '.join(x),axis=1)
    zh_news_merged = zh_news[['text']]
    '''开多进程进行分词'''
    zh_news_merged_seg = parallelize(seg_df,zh_news_merged)
    '''把中文维基和汉语新闻的数据整合'''
    merged_zhnews_zhwiki = pd.concat([zhwiki_texts_seg[['text']],zh_news_merged_seg[['text']]],axis=0)
    merged_zhnews_zhwiki.to_csv(merged_zhnews_zhwiki_path,index=False,header=True)
    zhwiki_file.close()

def prepare_required_file():
    '''
    准备好做文本摘要所需要的文件，包括word2id，id2embed, 词频文件和id2weight
    为了后台起服务时更快，把这些文件保存为json对象或pickle对象，既方便加载，又节省内存。
    json对象的key必须是字符串，所以id2embed和id2weight就没保存为json格式，而是pickle格式。
    '''
    _,vocab,embeddings = load_word2vec()
    word_id, id_embed = calcu_word_id_embed(vocab,embeddings)
    save_pickle(id_embed,id_embed_path)
    del embeddings,id_embed
    gc.collect()
    
    corpus = pd.read_csv(merged_zhnews_zhwiki_path)
    corpus = chain.from_iterable([text.split() for text in corpus['text']])
    '''计算词频，并根据论文的公式计算词的权重'''
    word_freq = calcu_word_freq(corpus, min_freq=5)
    word_weight = calcu_word_weight(word_freq,param=1e-3)
    '''词的id和权重相对应的字典'''
    id_weight = calcu_id_weight(word_id, word_weight)
    
    dump_json(word_id,word_id_path)
    dump_json(word_freq,word_freq_path) 
    save_pickle(id_weight,id_weight_path) 
    
def prepare_userdict():
    '''
    通过观察中文维基文件里的title字段，发现是分好的词，可以准备好作为jieba的自定义字典。
    不加入自定义字典，发现确实很多专有的词被分错了。
    不过需要对title字段进行清理，把 (*) 去除掉。
    :return:
    '''
    user_dict_file = open(user_dict_path,'w',encoding='utf-8')
    user_dict = []
    with open(zhwiki_path,encoding='utf-8') as zhwiki_file:    
        for line in zhwiki_file: 
            if not line.strip():
                continue
            article = json.loads(line)
            title = article['title'].strip()
            word = re.sub(r'\(.+\)','',title)
            user_dict.append(word.strip())
    user_dict = set(user_dict)
    user_dict_file.write('\n'.join(user_dict))    
    user_dict_file.close()
            
if __name__ == '__main__':
    # generate_data()
    # prepare_required_file()
    # prepare_userdict()
    

    
    
    
    
    