## 一、项目介绍

构建非监督文本自动摘要模型，用于对新闻文本提取摘要。

用word2vec训练词向量，训练语料为汉语新闻语料和中文wiki语料。

句向量使用SIF句向量，为词向量加权求和所得。

计算句子和新闻标题、新闻正文的余弦相似度，然后进行平滑得到句子的得分，再取得分最高的topk条句子作为摘要。

## 二、代码环境

python版本为python3.7，系统为ubuntu 18.04。

## 三、项目结构（模型部分）

news_summary_model/

├── data
│   ├── processed_data                               // 预处理好的文件，包括word2id，id2weight，id2embed等，保存为json或pickle对象备用
│   ├── raw_data                                     // 汉语新闻语料文件，用于模型测试
│   ├── stopwords                                    // 停用词表
│   └── word2vec                                     // 训练好的word2vec
├── font                                             // word2vec可视化需要的字体和保存的可视化结果
│   ├── SimHei.ttf
│   └── tsne.png
├── news_summary.py                                  // 自动摘要模型模块
├── README.md
├── requirements.txt			             // 需要安装的依赖包
└── utils
    ├── build_w2v.py                                 // 训练word2vec和可视化
    ├── config.py                                    // 全部文件的路径
    ├── data_loader.py                               // 数据预处理模块，分词和准备word2id，id2weight，id2embed等
    ├── data_utils.py                                // 数据处理函数的模块，包括分词、划分句子、句子id化等 
    ├── multi_proc_utils.py                          // 分词时进行多进程处理的函数
    ├── __pycache__
    └── SIF_embedding.py                             // 计算SIF句子向量的函数

## 四、项目运行

### 1、安装依赖包

    pip install -r requirements.txt

### 2、前端页面

可访问的外网链接：http://117.51.152.213:10446/

或者 python app.py    然后打开网页，输入：http://0.0.0.0:6006/

### 3、自动摘要模型

    python new_summary.py

用汉语新闻语料做自动摘要的测试。

在数据预处理时，把中文wiki的title字段整理成自定义字典，加到jieba里，所以模型初始化时，比较慢。

### 4、数据预处理

调用data_loader.py的函数，进行语料分词和保存，准备word2id，id2weight，id2embed等文件。

在数据预处理时，把中文wiki的title字段整理成了自定义字典，调用模型时，会把它加载到jieba里，提高分词的准确率。

### 5、训练word2vec

训练word2vec词向量和可视化

    python build_w2v.py

