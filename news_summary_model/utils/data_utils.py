#coding:utf-8
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))

import numpy as np
import pickle,json,jieba,re
from gensim.models import word2vec
from news_summary_model.utils.config import *
from collections import Counter
from itertools import chain

'''
中文维基文件里的title字段，准备成了自定义字典，可以加载到jieba里，提高分词准确率。
每次起服务时都导入自定义字典，导致速度有点慢。
'''
# jieba.load_userdict(user_dict_path)

# 加载json文件
def load_json(filename):
    return json.load(open(filename,'r',encoding='utf-8'))

# 保存为json文件
def dump_json(s,filename):
    json.dump(s,open(filename,'w',encoding='utf-8'),indent=2,ensure_ascii=False)

# 保存为pickle对象
def save_pickle(s,file_path):
    pickle.dump(s,open(file_path,'wb'))

# 加载pickle对象
def load_pickle(file_path):
    return pickle.load(open(file_path,'rb'))

def clean_sent(sentence):
    '''
    函数没有return时会返回None，为了避免这种情况，空值返回''
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”~@#￥%……&*（）]+',
            '', sentence)
    else:
        return ''

def filter_stopwords(words):
    return [word for word in words if word not in stop_words]

def sentence_seg(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    sentence = clean_sent(sentence.strip())
    words = jieba.lcut(sentence)
    words = filter_stopwords(words)
    return ' '.join(words)   

def seg_df(df):
    '''
    :param df: pd.DataFrame格式的数据
    :return: 分词后的数据
    '''
    for col in df.columns.tolist():
        df[col] = df[col].apply(sentence_seg)
    return df

def load_word2vec(word2vec_path):
    '''
    以model二进制文件的形式加载，占用内存大大减少
    ：return： w2v模型，所有词和对应的embedding
    '''
    w2v_model = word2vec.Word2Vec.load(word2vec_model_path)
    vocab = model.wv.index2word
    embeddings = model.wv.vectors
    return w2v_model,vocab,embeddings

def calcu_word_freq(corpus,min_freq):
    '''
    :param corpus: 未去重的词库
    :param min_freq: 过滤词频低于阈值的词
    :return: 词与词频的字典
    '''
    counter = Counter(corpus)
    counter = sorted(counter.items(),key=lambda x:x[1],reverse=True)
    word_freq = {word:freq for word,freq in counter if freq >= min_freq}
    return word_freq

def calcu_word_weight(word_freq,param=1e-3):
    '''
    :param word_freq: 词频字典
    :param param: 计算权重的一个参数
    :return: 词与权重的字典
    '''
    if param <= 0: 
        param = 1.0
    freq_sum = sum(word_freq.values())
    word_weight = {}
    for word, freq in word_freq.items():
        word_weight[word] = param / (param + freq / freq_sum)
    return word_weight

def calcu_word_id_embed(vocab,embeddings):
    '''
    :param vocab: 去重的词表
    :param embeddings: 词对应的词向量
    :return: 词与id的字典、id与词向量对应的字典
    '''
    word_id = dict(zip(vocab,range(len(vocab))))
    id_embed = dict(zip(range(len(vocab)),embeddings))
    return word_id,id_embed

def calcu_id_weight(word_id, word_weight):
    '''
    :param word_id:  词与id的字典
    :param word_weight: 词与权重的字典
    :return: id与权重的字典
    '''
    id_weight = {}
    for word,ind in word_id.items():
        if word in word_weight:
            id_weight[ind] = word_weight[word]
        else:
            id_weight[ind] = 1.0
    return id_weight

def calcu_sents_id(sentences,word_id):
    '''
    :param sentences: n条句子组成的列表
    :param word_id: 词与id的字典
    :return: n条句子中词的id
    '''
    sents_id = []
    for sent in sentences:
        sent_id = [word_id[word] for word in sent.split() if word in word_id]
        if sent_id:
            sents_id.append(sent_id)
    sents_id_array = array_pad(sents_id)
    return sents_id_array

def array_pad(id_lists):
    '''
    在获得n条句子中词的id时，将不同长度的句子统一长度，长度为其中最长句的长度。
    由于0本身就是词的id，所以用-1来填充其他句子。
    :param id_lists:n条句子的词的id
    :return: pad后统一长度的id的array数组
    '''
    lengths = [len(ids) for ids in id_lists]
    n_samples = len(id_lists)
    maxlen = np.max(lengths)
    id_pad_array = np.zeros((n_samples, maxlen),dtype=np.int32) - 1
    for ind, ids in enumerate(id_lists):
        id_pad_array[ind,:lengths[ind]] = ids
    return id_pad_array

def calcu_sents_weight(sents_id, id_weight):
    '''
    根据句子中词的id的array数组，得到句子中词的权重的array数组。
    同样要统一长度，用0来填充。
    :param sents_id: 句子中词的id的array数组
    :param id_weight: 提前准备好的id和权重对应的字典
    :return: 句子中词的权重的array数组
    '''
    sents_weight_array = np.zeros(sents_id.shape)
    for i in range(sents_id.shape[0]):
        for j in range(sents_id.shape[1]):
            if sents_id[i,j] >= 0:
                sents_weight_array[i,j] = id_weight[sents_id[i,j]]
    return sents_weight_array

def prepare_sentences(title,news):
    '''
    输入新闻标题和正文，进行数据预处理：划分句子，分词。
    :param title: 新闻标题
    :param news: 新闻的正文
    :return: 分词后的标题，分词后的正文，正文划分句子后分词的结果，正文划分句子的结果
    '''
    title_seg = sentence_seg(title.strip())
    news_split = news.strip().split('\n+')
    sentences_list = []
    for sent in news_split:
        # 正则：把双引号中的句子提取处理，不是句子的（专有名词）的则不提取
        # 正则：把不含双引号的句子，按[。！？!?]进行切分，得到句子。
        # 可能存在英文，但英文句号可能为小数点，就不根据.进行划分了。
        pat_talk = re.compile(r'(“[^”]+”[。！？!?]|“[^”]+[。！？!?]”)')
        pat_punc = re.compile(r'([。！？!?])')
        sent_split_1 = pat_talk.split(sent)
        for sent in sent_split_1:
            if not sent.strip():
                continue
            if not pat_talk.findall(sent):
                # 如果不是含有双引号的句子，就再进行划分。
                sent_split_2 = pat_punc.split(sent.strip())
                lens = len(sent_split_2) // 2
                # 按[。！？!?]进行切分，这些符号也会切分为列表中的元素，再和前面的句子拼接起来
                sent_split_2 = [''.join(sent_split_2[2*i:2*(i+1)]) for i in range(lens) if lens >= 1]
                if sent_split_2:
                    sentences_list.append(sent_split_2)
            else:
                # 如果是含有双引号的句子，就不再进行划分。
                sentences_list.append([sent])
    sentences_list = list(chain.from_iterable(sentences_list))
    # 用汉语新闻测试时，发现有很多 ???? ，所以加上这个正则来去掉,这是英文中的问号。同时去掉空字符。
    sentences_list = [re.sub(r'[?{4}]','',sent).strip() for sent in sentences_list if (re.sub(r'[?{4}]','',sent)).strip()]
    sentences_seg = [sentence_seg(sent).strip() for sent in sentences_list if (sentence_seg(sent)).strip()]
    
    news_seg  = ' '.join(sentences_seg)
    return title_seg,news_seg,sentences_seg,sentences_list

def smooth_scores(cosine_scores):
    '''
    把新闻中句子的分数进行平滑。平滑是指某个词与前后2个词（一共5个词）的分数进行加权求和。
    window为5,3和7的效果应该不太好。
    :param cosine_scores: n条句子的分数
    :return: n条句子平滑后的分数
    '''
    window = 5
    pad = window // 2
    scores_pad = np.pad(cosine_scores,pad,'constant')              # 借鉴ngram的做法，先前后补0，再取5gram
    scores_ngram = [scores_pad[i:i+window] for i in range(len(scores_pad)-window+1)]
    targed_weight = 0.7                                 # 该句子的权重取0.7，其他4句合计0.3
    others_weights = [0.05,0.1,0.1,0.05]                # 其他4句的权重
    scores_smoothed = []
    for scores in scores_ngram:
        target = scores[pad]
        others = np.delete(scores,pad)
        if np.count_nonzero(others) == 4:
            score_smoothed = targed_weight * target + sum(np.multiply(others_weights,others))
        else:
            # 对于开头前两个权重，和末尾两个权重，由于补0，存在0元素，为了保证权重之和为1，非0元素平分0.3的权重。
            weight = 0.3 / np.count_nonzero(others)
            score_smoothed = sum([targed_weight * target] + [weight * score for score in others])
        scores_smoothed.append(score_smoothed)
    return scores_smoothed

def topk_by_score(sentences,scores,topk_score):
    '''
    根据平滑后的句子分数，取分数最高的前几个作为摘要，同时保持句子原来的顺序。
    :param sentences: 划分好的n条句子
    :param scores: n条句子的平滑分数
    :param topk_score: 得分阈值
    :return: 新闻的摘要
    '''
    pairs = zip(sentences,scores)
    sentences_topk = [sent for sent,score in pairs if score >= topk_score]
    return sentences_topk

if __name__ == '__main__':
    # 测试提取双引号句子的正则。
    title = '使这个给分点更具操作性'
    news = '细化评分细则。更具操作性。“每个人”细化评分细则。细化评分细则“每个人”。“做一遍，再结合标准答案 ”。使这个更具操作性。“做一遍”使这个更具操作性。“做一遍之后，再结合标准答案 。” 细化评分细则？'
    prepare_sentences(title, news)