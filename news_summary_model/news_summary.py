#coding:utf-8
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))

from news_summary_model.utils.SIF_embedding import sif_embedding
from news_summary_model.utils.data_utils import *
import numpy as np
import pandas as pd
from news_summary_model.utils.config import *
from sklearn.metrics.pairwise import cosine_similarity

class NewsSummary(object):
    def __init__(self):
        '''
        提前准备好计算句向量必备的文件：word2id，id2embed，id2weight
        '''
        self.word_id = load_json(word_id_path)
        self.id_embed = load_pickle(id_embed_path)
        self.id_weight = load_pickle(id_weight_path)
    
    def sentence_embedding(self,sentences):
        '''
        :param sentences: 分好词的n条句子
        :return: n条句子的sif句向量
        '''
        sents_id_array = calcu_sents_id(sentences, self.word_id) 
        sents_weight_array = calcu_sents_weight(sents_id_array, self.id_weight) 

        sents_sif_embeds = sif_embedding(self.id_embed, sents_id_array, sents_weight_array) 
        return sents_sif_embeds    
      
    def calcu_scores(self,embeddings):
        '''
        根据句向量计算每个句子的得分，再进行平滑。
        每个句子与标题的余弦相似度占0.2的权重，句子与正文的余弦相似度占0.8的权重，进行加权求和。
        :param embeddings: n条句子的sif句向量
        :return: n条句子平滑后的得分
        '''
        cosine_scores = []
        title_emb,news_emb = embeddings[-2:]
        for emb in embeddings[:-2]:
            title_similar = cosine_similarity(emb.reshape(1,-1), title_emb.reshape(1,-1))
            news_similar = cosine_similarity(emb.reshape(1,-1), news_emb.reshape(1,-1))
            score_weighted = 0.2 * title_similar + 0.8 * news_similar
            cosine_scores.append(score_weighted[0][0])
        scores_smoothed = smooth_scores(cosine_scores)
        return scores_smoothed
    
    def choose_topk(self,sentences,scores,lens):
        '''
        根据句子得分，挑选得分最高的前几条作为摘要。
        摘要的数量与新闻中句子的数量有关，是一个变动的值。
        同时考虑作为摘要的句子得分必须不低于平均值。
        :param sentences: 划分好的原始句子（未分词）
        :param scores: 所有句子的得分
        :param lens: 句子的数量
        :return: 新闻摘要
        '''
        # 摘要的数量是和lens有关的分段函数
        topk_list = [round(lens * 0.3),9]
        topk = np.select([lens<=30,lens>30],topk_list)
        topk_score_by_lens = sorted(scores,reverse=True)[topk-1]
        topk_score_by_mean = np.mean(scores)
        topk_score = np.max([topk_score_by_lens,topk_score_by_mean])
        return topk_by_score(sentences,scores,topk_score)
            
    def news_summary(self,title,news):
        '''
        输入新闻标题和正文，得到新闻摘要
        如果句子数量在3以下，不做摘要，直接输出
        第一句话非常重要，必须输出，不参与句向量和得分的计算
        :param title: 新闻标题
        :param news: 新闻正文
        :return: 新闻摘要
        '''
        title_seg,news_seg,sentences_seg,sentences_list = prepare_sentences(title,news)
        lens = len(sentences_list[1:])
        if lens <= 1:
            return sentences_list
        
        sentences_all = sentences_seg[1:] + [title_seg,news_seg]    # 把标题和正文的分词结果加入，一起计算sif句向量
        embeddings = self.sentence_embedding(sentences_all)     # 除第一句外所有句子（包括标题和正文）的sif句向量
        scores_smoothed = self.calcu_scores(embeddings)         # 经过平滑后的句子的得分（不含标题和正文）
        
        sents_topk = self.choose_topk(sentences_list[1:],scores_smoothed,lens)  
        sents_topk = [sentences_list[0]] + sents_topk           # 加上首句
        return sents_topk
    

if __name__ == '__main__':
    news_corpus = pd.read_csv(news_path,encoding='gb18030')
    model = NewsSummary()
    
    for i in range(news_corpus.shape[0]):
        title = news_corpus.title[i]
        news = news_corpus.content[i]
        if isinstance(title,float) or isinstance(news,float):
            print('请输入完整的新闻（包含标题和内容）！')
        else:
            sentence_topk = model.news_summary(title,news)
            print('\n'.join(sentence_topk))
            print()
            # time.sleep(0.5)
    
    
