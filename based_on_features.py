#/usr/bin/env python3
# coding:utf-8
import codecs
import jieba.posseg as pseg
from collections import Counter
import numpy as np

# 文本分句
def sentence_cut(text, punctuation_list=',.!?;~，。！？；～… '):
    sentences = []
    sentiment_word_position = 0
    word_position = 0
    #punctuation_list = punctuation_list.decode('utf8')
    for words in text:
        word_position += 1
        if words in punctuation_list:
            nextWord = list(text[sentiment_word_position:word_position+1]).pop()
            if nextWord not in punctuation_list:
                sentences.append(text[sentiment_word_position:word_position])
                sentiment_word_position = word_position
    if sentiment_word_position < len(text):
        sentences.append(text[sentiment_word_position:])
    return sentences


#人名，地名，名词，动词且长度大于1的词作为关键词
def get_key_words(sentences, types=['ns', 'nr', 'n', 'v']):
    # 统计词频
    keywords_weights = {}
    keywords = []  # postagger = Postagger() postagger.load('') title_words = pseg.cut(title)
    for line in sentences:
        for word, flag in pseg.cut(line):
            if flag in types and len(word) > 1:
                keywords.append(word)
    keywords_fre = Counter(keywords)
    len_sen = len(sentences)
    # 计算关键词在所有句子中出现的频次
    for word in keywords_fre.keys():
        count = 0
        for line in sentences:
            if word in line:
                count += 1
        keywords_weights[word] = count*1.0 / len_sen
    return keywords_weights,keywords_fre

#计算每个句子的权重:将句子中的重要词语的权重加和
def compute_sentences_weigths(keywords, sentences):
    content_weights = {}
    for sentence in sentences:
        sentence = sentence[:-1]
        content_weights[sentence] = 0.
        # print(content_weights)
        for word in keywords.keys():
            if word in sentence:
                # content_weights.get(sentence,0.)
                content_weights[sentence] += keywords[word]

    content_weights = sorted(content_weights.items(),
                             key=lambda d: d[1], reverse=True)
    content_weights = [list(result)[0] for result in content_weights]
    return content_weights


def runFeature(text):
    sentences = sentence_cut(text,punctuation_list=u'，。！：;；?？')

    keywords_weights,keywords_fre = get_key_words(sentences)

    content_weights = compute_sentences_weigths(keywords_weights, sentences)

    summary = content_weights[0]
    # print (summary)
    return summary

def rouge_n(auto_summary, manu_summary):
    import jieba
    rouge_1_p = 0.0; rouge_1_r = 0.0; rouge_1_f = 0.0
    rouge_2_p = 0.0; rouge_2_r = 0.0; rouge_2_f = 0.0
    rouge_3_p = 0.0; rouge_3_r = 0.0; rouge_3_f = 0.0

    results = np.zeros(9)
    i = 0.
    for word in manu_summary:
        if(word in auto_summary):
            i = i + 1
    rouge_1_p = i / len(auto_summary); results[0] = round(rouge_1_p,4)
    rouge_1_r = i / len(manu_summary); results[1] = round(rouge_1_r,4)
    try:
        rouge_1_f = 2*rouge_1_p*rouge_1_r / (rouge_1_r + rouge_1_p)
    except:
        rouge_1_f = 0.
    results[2] = round(rouge_1_f,4)

    manu_cut = list(jieba.cut(manu_summary))
    auto_cut = list(jieba.cut(auto_summary))

    i = 0.
    for word in manu_cut:
        if((word in auto_cut) and len(word) >=2 ):
            i = i + int(len(word)/2)

    rouge_2_p = i / int(len(auto_summary)/2); results[3] = round(rouge_2_p,4)
    rouge_2_r = i / int(len(manu_summary)/2);results[4] = round(rouge_2_r,4)
    try:
        rouge_2_f = 2*rouge_2_p*rouge_2_r / (rouge_2_r + rouge_2_p)
    except:
        rouge_2_f = 0.
    results[5] = round(rouge_2_f,4)

    i = 0.
    for word in manu_cut:
        if((word in auto_cut) and len(word) >=3 ):
            i = i + int(len(word)/3)
            manu_cut.remove(word)
    for j in range(0,len(manu_cut)-1,2):
        if(manu_cut[j]+manu_cut[j+1] in auto_summary):
            i = i + 1

    rouge_3_p = i / int(len(auto_summary)/3); results[6] = round(rouge_3_p,4)
    rouge_3_r = i / int(len(manu_summary)/3); results[7] = round(rouge_3_r,4)
    try:
        rouge_3_f = 2*rouge_3_p*rouge_3_r / (rouge_3_r + rouge_3_p)
    except:
        rouge_3_f = 0.
    results[8] = round(rouge_3_f,4)

    results = list(results)
    return results

def plot(in_data):
    import matplotlib.pyplot as plt
    import pandas as pd 
    header = ['doc_id', 'rouge_1_p', 'rouge_1_r', 'rouge_1_f', 'rouge_2_p', 
            'rouge_2_r', 'rouge_2_f','rouge_3_p', 'rouge_3_r', 'rouge_3_f']
    df = pd.read_csv(in_data, sep = '\t', header = None)
    df = df[[0, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    df.columns = header
    df = df.sort_values(by='rouge_1_f', ascending=False)

    x = [i for i in range(725)]
    y1_f = df['rouge_1_f']
    y1_p = df['rouge_1_p']#.tolist() 
    y1_p = sorted(y1_p,reverse=True)
    y1_r = df['rouge_1_r']#.tolist()
    y1_r = sorted(y1_r,reverse=True)
    plt.plot(x,y1_f,'r-',color = 'red',label = 'rouge_1_f')
    plt.plot(x,y1_p,'r--',color = 'blue',label = 'rouge_1_p')
    plt.plot(x,y1_r,'-',color = 'green',label = 'rouge_1_r')
    plt.xlabel("Diffrents weibo doc_id")
    plt.ylabel("ROUGE scores value")
    plt.title('ROUGE-1 pricision/recall/F1-measure test scoring')
    plt.legend()
    plt.savefig('./data/ROUGE_1_scoring.jpg')
    plt.show()

    y1_f = df['rouge_1_f']
    y2_f = df['rouge_2_f'].tolist()
    y2_f = sorted(y2_f,reverse=True)
    y3_f = df['rouge_3_f'].tolist()
    y3_f = sorted(y3_f,reverse=True)
    plt.plot(x,y1_f,'-',color = 'red',label = 'rouge_1_f')
    plt.plot(x,y2_f,'-',color = 'blue',label = 'rouge_2_f')
    plt.plot(x,y3_f,'-',color = 'green',label = 'rouge_3_f')
    plt.xlabel("Diffrents weibo doc_id")
    plt.ylabel("ROUGE scores value")
    plt.title('ROUGE-1/2/3 test F1-measure')
    plt.legend()
    plt.savefig('./data/ROUGE_1_2_3_scoring.jpg')
    plt.show()


if __name__ == '__main__':
    res = []
    data_set_file = './data/data_set.txt'
    with open(data_set_file,'r',encoding = 'utf8') as fi:
        for line in fi.readlines():
            rouge_res = []
            text = line.strip().split('\t')
            # print(text)
            auto_summary = runFeature(text[3])
            manu_summary = text[2]
            socres = rouge_n(auto_summary,manu_summary)
            rouge_res.append(text[0])
            rouge_res.append(text[1])
            rouge_res.append(auto_summary)
            rouge_res.extend(socres)
            res.append(rouge_res)

    out_results = './data/summary_and_rouge_scores.txt'
    with codecs.open(out_results,'w+',encoding='utf8') as fo:
      for i in range(len(res)):
          fo.writelines([str(res[i][0]),'\t',str(res[i][1]),'\t',str(res[i][2]),'\t',str(res[i][3]),
            '\t',str(res[i][4]),'\t',str(res[i][5]),'\t',str(res[i][6]),'\t',str(res[i][7]),
            '\t',str(res[i][8]),'\t',str(res[i][9]),'\t',str(res[i][10]),'\t',str(res[i][11]),'\n'])

    plot(out_results)


