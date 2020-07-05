#coding:utf-8
from flask import Flask, render_template, request

import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))

from news_summary_model.news_summary import NewsSummary
from news_summary_model.utils.data_utils import *

app = Flask(__name__)
app.secret_key = 'development key'

model = NewsSummary()

@app.route('/', methods=['POST', 'GET'])
def index():
    news = ""
    abstract = ""
    title = ""
    if request.method == 'POST':
        result = request.form
        for key, value in result.items():
            print(key)
            print(value)
            if(key == "News"):
                news = value
            elif(key == "Abstract"):
                abstract = value
            elif (key == "Title"):
                title = value
        # 输出摘要
        abstract = model.news_summary(title, news)
        abstract = '\n\n'.join(abstract)
    elif request.method == 'GET':
        pass

    print(title)
    print(news)
    print(abstract)
    return render_template("index.html", title = title, news = news, abstract = abstract)

app.debug = True

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6006, debug = True)
