# coding:utf-8
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import font_manager 
from config import *
import logging,os,gc
from multiprocessing import cpu_count
from data_utils import *

"""打印日志信息"""
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
'''计算cpu核数，用于开最大进程训练'''
cores = cpu_count()

def train_wor2vec():
    """"设定词向量训练的参数，开始训练词向量"""
    num_features = 300      # 词向量取300维
    min_word_count = 5      # 词频小于5个单词就去掉
    num_workers = cores     # 开最大进程，并行训练
    epoch = 10              # 训练10轮
    context = 5             # 上下文滑动窗口的大小
    model_ = 1              # 使用skip-gram模型进行训练
    
    logger.info('开始训练word2vec...')
    model = word2vec.Word2Vec(LineSentence(merged_zhnews_zhwiki_path),size=num_features, min_count = min_word_count, \
                              iter = epoch,workers = num_workers,window = context, sg=model_)
    # 保存模型
    # 第一种方法保存的文件不能利用文本编辑器查看，但是保存了训练的全部信息，可以在读取后追加训练
    # 后一种方法保存为word2vec文本格式，但是保存时丢失了词汇树等部分信息，不能追加训练
    model.save(word2vec_model_path)
    model.wv.save_word2vec_format(word2vec_txt_path,binary = False)
    logger.info('模型训练完毕并保存')
    del model
    gc.collect()    

# 根据词向量找同义词
def synonym(word,topn=10):
    return model.wv.most_similar(word,topn=topn)

# 根据词向量进行类比，类似于：美国-奥巴马+金正恩=韩国
def analogy(x1,x2,y1):
    result = model.most_similar(positive=[y1,x2],negative=[x1])
    return result[0][0]

def visualize_wor2vec(vocab,embeddings):
    '''
    词向量的可视化
    :param vocab: 词表
    :param embeddings: 词向量矩阵
    :return:
    '''
    # tsne: 一个降维的方法，降维后维度是2维，使用'pca'来初始化。
    # 取出了前500个词的词向量，把300维减低到2维。
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)         
    plot_only = 500
    two_dim_embs = tsne.fit_transform(embeddings[:plot_only,:])           
    labels = vocab[:plot_only]     
    filename='../font/tsne.png'
    plt.figure(figsize=(18, 18)) 
    myfont = font_manager.FontProperties(fname='../font/SimHei.ttf')  
    for i, label in enumerate(labels):
        x, y = two_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),                           
                 textcoords='offset points',
                 ha='right',
                 va='bottom',
                 fontproperties=myfont)
    
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    train_wor2vec()
    
    # print(synonym('勇敢'))
    # print(synonym('美女'))
    # print(analogy('中国','汉语','美国'))
    # print(analogy('美国','奥巴马','美国'))  
    
    # _,vocab,embeddings = load_word2vec()
    # visualize_wor2vec(vocab,embeddings)