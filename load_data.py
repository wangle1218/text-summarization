#/usr/bin/env python3
# coding:utf-8
import codecs
import re
import linecache
import jieba.posseg as pseg
from collections import Counter

def deal_data(in_file):
    output = []
    dataSet = []
    doc_id = 0
    human_label = 0
    summary = r'20U'
    short_text = r'200U'

    with codecs.open(in_file,'r',encoding='utf8') as fi:
        for line in fi.readlines():
            text = line.strip()
            if text[:8] == '<doc id=':
                p = re.compile(r'\d+')
                doc_id = p.findall(text)
                output.append(doc_id[0])
            elif text[:13] == '<human_label>':
                p = re.compile(r'\d+')
                human_label = p.findall(text)
                if int(human_label[0]) < 3:
                    break
                else:
                    output.append(human_label[0])
            elif text == '<summary>':
                summary = linecache.getline(in_file,int(doc_id[0])*9+4)
                summary = summary.strip()
                output.append(summary)
            elif text == '<short_text>':
                short_text = linecache.getline(in_file,int(doc_id[0])*9+7)
                short_text = short_text.strip()
                output.append(short_text)
            elif text == '</doc>':
                dataSet.append(output)
                output = []
            else:
                continue
    return dataSet

def write_data(data,out_file):
    with codecs.open(out_file,'a',encoding='utf8') as fo:
        # for i in range(len(data)):
        #     for word, flag in pseg.cut(data[i][2] + data[i][3]):
        #         fo.writelines([word,' '])
        #     fo.write('\n')
        for i in range(len(data)):
            fo.writelines([str(data[i][0]),'\t',str(data[i][1]),'\t',
                            data[i][2],'\t',data[i][3],'\n'])

if __name__ == '__main__':
    in_file = './data/PART_III.txt'
    out_file = './data/data_set.txt'
    data = deal_data(in_file)
    write_data(data, out_file)