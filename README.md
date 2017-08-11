# 文本自动摘要text automatic summarization
自然语言处理任务--文本自动摘要的一个实践。

采用基于启发式的规则文本摘要算法，给定一条微博，通过算法，从这个微博中抽取出一条句子作为该条微博的摘要。

./data/PART_III.txt 文件是原始数据集，经过load_data.py脚本，将数据集处理成更加适合算法处理的形式的数据格式，输出为data_set.txt。

based_on_features.py 脚本将data_set.txt输入，将每一条微博生成一个摘要，并且对每一条摘要和参考摘要进行对比，利用ROUGE-N评测度量自动摘要的性能好坏。

## 脚本使用方法：
cd ./weibo_text_summarization #进入项目文件夹

python load_data.py #处理原始数据集

python based_on_features.py #自动提取摘要，输出评测结果
