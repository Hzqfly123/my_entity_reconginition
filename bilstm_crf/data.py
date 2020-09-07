import sys
import pickle
import os
import random
import numpy as np

#第一步：数据处理
#pikle是一个将任意复杂的对象转成对象的文本或二进制表示的过程。

# tags, BIO
# tag2label = {"O": 0,
#              "B": 1,
#              "I": 2
#              }


tag2label = {'B-Department':0, 'B-Symptom':1, 'B-Check':2, 'B-Operation':3, 'B-Disease':4, 'B-Relative_disease':5, 'B-Drug':6, 
           'B-Registration_department':7, 'B-Body':8, 'I-Department':9, 'I-Symptom':10, 'I-Check':11, 'I-Operation':12, 
           'I-Disease':13, 'I-Relative_disease':14,'I-Drug':15, 'I-Registration_department':16, 'I-Body':17, 'O':18 }
label2tag = {0:'B-Department', 1: 'B-Symptom', 2:'B-Check', 3:'B-Operation', 4:'B-Disease', 5:'B-Relative_disease', 6:'B-Drug', 
           7:'B-Registration_department', 8:'B-Body',      9:'I-Department', 10:'I-Symptom', 11:'I-Check', 12:'I-Operation', 
           13:'I-Disease', 14:'I-Relative_disease',15:'I-Drug', 16:'I-Registration_department', 17:'I-Body', 18:'O' }


#输入train_data文件的路径，读取训练集的语料，输出train_data,返回 [[文字],[标签]]形式
def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: raw_data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    
    sent_, tag_ = [], []
    line_num = 0
    for line in lines:
        #if line != '\n':  # 每句话以换行符区分
            # char 与 label之间有个空格
            # line.strip()的意思是去掉每句话句首句尾的空格
            # .split()的意思是根据空格来把整句话切割成一片片独立的字符串放到数组中，同时删除句子中的换行符号\n
            # line.strip().split('\t') 这样就把每行的每个字符一个个分开，变成一个list，按照制表符切割字符串
        fields = line.split()
        char = fields[0]
        label = fields[-1]
        line_num = line_num + 1
        # 取120字为一段话加入训练
        sent_.append(char)
        tag_.append(label)
        # 一段话结束加入到data
        if line_num % 120 == 0:
            data.append((sent_,tag_))
            sent_, tag_ = [], []
    data.append((sent_,tag_))
    return data   


#由train_data来构造一个(统计非重复字)字典{'第一个字':[对应的id,该字出现的次数],'第二个字':[对应的id,该字出现的次数], , ,}
#去除低频词，生成一个word_id的字典并保存在输入的vocab_path的路径下，
#保存的方法是pickle模块自带的dump方法，保存后的文件格式是word2id.pkl文件

def vocab_build(vocab_path, corpus_path, min_count):
    """
    BUG: I forget to transform all the English characters from full-width into half-width... 
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    #sent_的形状为['我','在','北','京']，对应的tag_为['O','O','B-LOC','I-LOC']
    for sent_ in data[0]:
        for word in sent_:
            # 如果字符串只包含数字则返回 True 否则返回 False
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:# 新词
                 # [len(word2id)+1, 1]用来统计[位置标签，出现次数]，第一次出现定为1
                word2id[word] = [len(word2id)+1, 1]
            else:#不是新词
            # word2id[word][1]实现对词频的统计，出现次数累加
                word2id[word][1] += 1

    # 前面的任务都是统计词频，而它的目的是为了删除低频词，删除完就不必统计词频了
    # 用来统计低频词
    low_freq_words = []
    # 寻找低于某个数字的低频词
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    # 删除词频出现较少的单词
    for word in low_freq_words:
        del word2id[word]

    # 剔除词频出现较少的单词，不再统计词频
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    # 分词后不在词典内的词经常被标为<unk>，处理为相同长度通常会在前或后补<pad>   猜测
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    # 将word2id存储进入文件中
    with open(vocab_path, 'wb') as fw:
        # 序列化到名字为word2id.pkl文件
        pickle.dump(word2id, fw)


#输入一句话，生成一个 sentence_id
'''sentence_id的形状为[1,2,3,4,...]对应的sent为['当','希','望','工',程'...]'''

def sentence2id(sent, word2id):
    """
    将一个句子进行编号
    :param sent: 表示一个句子
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id
    


#通过pickle模块自带的load方法(反序列化方法)加载输出word2id
def read_dictionary(vocab_path):
    """
    读取之前存入文件中的word2id词典
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:   # ../
        #反序列化方法加载输出
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    
    return word2id
'''word2id的形状为{'腹': 1, '痛': 2, '，': 3, '的': 4, '挂': 5,。。'<UNK>': 3904, '<PAD>': 0}
   总共1339个字'''

#输入vocab，vocab就是前面得到的word2id，embedding_dim=300
def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    # 返回一个len(vocab)*embedding_dim=3905*300的矩阵(每个字投射到300维)作为初始值
    #numpy.random.uniform(low,high,size)功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    # 参数介绍:
    #     
    #     low: 采样下界，float类型，默认值为0；
    #     high: 采样上界，float类型，默认值为1；
    #     size: 输出样本数目，为int或元组(tuple)
    # 类型，例如，size = (m, n, k), 则输出m * n * k个样本，缺省时输出1个值。
    #
    # 返回值：ndarray类型，其形状和参数size中描述一致。ndarray为n为数组对象
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


#padding,输入一句话，不够标准的样本用pad_mark来补齐,pad_sequences序列填充
''' 
输入：seqs的形状为二维矩阵，形状为[[33,12,17,88,50]-第一句话
                                 [52,19,14,48,66,31,89]-第二句话
                                                    ] 
输出：seq_list为seqs经过padding后的序列
      seq_len_list保留了padding之前每条样本的真实长度
      seq_list和seq_len_list用来喂给feed_dict
'''
def pad_sequences(sequences, pad_mark=0):
    """
    补齐，将数据sequences内的word变成长度相同
    :param sequences:
    :param pad_mark:
    :return:
    """
    # 返回一个序列中长度最长的那条样本的长度s
    # max_len = 300
    seq_list, seq_len_list = [], []
    max_len = max(map(lambda x: len(x), sequences))
    for seq in sequences:
        # 由元组格式()转化为列表格式[]
        # seq = list(seq)
        # 不够最大长度的样本用0补上放到列表seq_list
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        #seq_list为sequences经过padding后的序列
        seq_list.append(seq_)
        # seq_len_list用来统计每个样本的真实长度
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list



#生成batch
''' seqs的形状为二维矩阵，形状为[[33,12,17,88,50....]...第一句话
                                [52,19,14,48,66....]...第二句话
                                                    ] 
   labels的形状为二维矩阵，形状为[[0, 0, 3, 4]....第一句话
                                 [0, 0, 3, 4]...第二句话
                                             ]
'''
def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    #if shuffle:
    #    random.shuffle(data) # 将所有元素随机打乱
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels
    
def get_entity(tag_seq, char_seq, keys):  # y_pred [B-body,,,], contents[腹,,,], entity[body,disease,,,]
    """
    返回实体类别
    :param tag_seq:
    :param char_seq:
    :param keys: key为list，表示返回需要的类别名称
    :return:
    """
    # entity = get_entity_one_(tag_seq, char_seq)
    # return entity
    entity = []
    for key in keys:
        entity.append(get_entity_key(tag_seq, char_seq, key))
    return entity


def get_entity_key(tag_seq, char_seq, key): 
    entities = []
    entity = ''
    for (char, tag) in zip(char_seq, tag_seq):
        if tag == 'B-' + key or tag == 'I-' + key :
            entity += char
        else:
            if len(entity) != 0:
                entities.append(entity)
                entity = ''
    if len(entity) != 0:
        entities.append(entity)
    return entities


if __name__ == '__main__':
    vocab_build('./train_test_data/char2id_bio.pkl', './train_test_data/train_bio_char.txt', 5)
    #with open('./train_test_data/char2id_bio.pkl', 'rb') as fr:
    #char2id = pickle.load(fr)
    #print('vocab_size:', len(char2id))
    #for key, value in word2id.items():
    #    print(key, value)
    # raw_data = read_corpus('data_path/train_new.txt')
    # print(len(raw_data))
    # data = read_corpus('./train_test_data/train_bio_char.txt') # /根目录 ./当前目录 ../上级目录



