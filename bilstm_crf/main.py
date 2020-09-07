import tensorflow as tf
import numpy as np
import os
import argparse
import time
import random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity_keys
from data import read_corpus, read_dictionary, tag2label, random_embedding
# 添加部分，解决gpu被占满问题
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)
# 第四步，运行  

# hyperparameters

## hyperparameters超参数设置
#使用argparse的第一步就是创建一个解析器对象，并告诉它将会有些什么参数。
#那么当你的程序运行时，该解析器就可以用于处理命令行参数
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')

#方法add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
# 其中：
# name or flags：命令行参数名或者选项，如上面的address或者-p,--port.​
# 其中命令行参数如果没给定，且没有设置defualt，则出错。但是如果是选项的话，则设置为None
# nargs：命令行参数的个数，
# 一般使用通配符表示，其中，'?'表示只用一个，'*'表示0到多个，'+'表示至少一个
# default：默认值
# type：参数的类型，默认是字符串string类型，还有float、int等类型
# help：和ArgumentParser方法中的参数作用相似，出现的场合也一致
# 最常用的地方就是这些，其他的可以参考官方文档
parser.add_argument('--data_dir', type=str, default='train_test_data', help='raw_data dir source')
parser.add_argument('--dictionary', type=str, default='char2id_bio.pkl', help='dictionary source')
parser.add_argument('--train_data', type=str, default='train_bio_char.txt', help='train raw_data source')  
parser.add_argument('--test_data', type=str, default='test.txt', help='test raw_data source')
parser.add_argument('--dev_data', type=str, default='dev.txt', help='test raw_data source')   # 添加
parser.add_argument('--batch_size', type=int, default=50, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=50, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')  # 解决梯度爆炸的影响
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random',
                    help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training raw_data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='random_char_300', help='model for test and demo')
parser.add_argument('--embedding_dir', type=str, default='../word2vector', help='embedding files dir')

#传递参数送入模型中解析
args = parser.parse_args()

# get char embeddings

'''word2id的形状为{'腹': 1, '痛': 2, '，': 3, '的': 4, '挂': 5, '号': 6, '科': 7,'<UNK>': 1338, '<PAD>': 0}
   train_data总共1338个去重后的字'''
word2id = read_dictionary('./'+ args.data_dir + '/' + args.dictionary)  #.\../train_test_data\word2id.pkl   修改 .\train_test_data\word2id.pkl

#通过调用random_embedding函数返回一个len(vocab)*embedding_dim=1338*300的矩阵(矩阵元素均在-0.25到0.25之间)作为初始值
if args.pretrain_embedding == 'random':   # 每个字投射300维作为初始值
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = os.path.join(os.path.curdir, args.embedding_dir, args.pretrain_embedding)
    embeddings = np.array(np.load(embedding_path), dtype='float32')

# read corpus and get training raw_data
# 读取数据
if args.mode != 'demo':
    # 设置train_path的路径为data_path下的train_data文件
    train_path = os.path.join('.', args.data_dir, args.train_data)  # \train_test_data\train_bio_char.txt
    # 设置test_path的路径为data_path下的test_path文件
    test_path = os.path.join('.', args.data_dir, args.test_data)
    # 通过read_corpus函数读取出train_data
    # 添加
    dev_path = os.path.join('.', args.data_dir, args.dev_data)
    """ train_data的形状为[(['我',在'北','京'],['O','O','B-LOC','I-LOC'])...第一句话
                         (['我',在'天','安','门'],['O','O','B-LOC','I-LOC','I-LOC'])...第二句话  
                          ( 第三句话 )  ] 总共有50658句话"""
    train_data = read_corpus(train_path) # 返回 [[文字],[标签]]形式
    test_data = read_corpus(test_path) 
    dev_data = read_corpus(dev_path)  
    test_size = len(test_data)

# paths setting
# 生成输出目录
#如果是训练就获取最新时间，否则就=args.demo_model
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
#输出地址,默认是./data_path_save/时间戳
output_path = os.path.join('.', "bio_model", timestamp)
#如果地址不存在就新建
if not os.path.exists(output_path):
    os.makedirs(output_path)

#./data_path_save/时间戳/summaries
summary_path = os.path.join(output_path, "summaries")

#如果地址不存在就新建
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
#./data_path_save/时间戳/checkpoints/
model_path = os.path.join(output_path, "checkpoints/")  # 去掉checkpoints后的/
if not os.path.exists(model_path):
    os.makedirs(model_path)
#./data_path_save/时间戳/checkpoints/model
ckpt_prefix = os.path.join(model_path, "model")

#./data_path_save/时间戳/results
result_path = os.path.join(output_path, "results")

#如果不存在就新建
if not os.path.exists(result_path):
    os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
#把调用的函数及各个参数写入日志文件
#2019-07-26 08:45:40,081:INFO: Namespace(CRF=True, batch_size=64, clip=5.0, demo_model='1521112368', 
# dropout=0.5, embedding_dim=300, epoch=40, hidden_dim=300, lr=0.001, mode='demo',
#  optimizer='Adam', pretrain_embedding='random', shuffle=True, test_data='data_path', train_data='data_path', update_embedding=True)
get_logger(log_path).info(str(args))
# training model
if args.mode == 'train':
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_prefix, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embedding)
    model.build_graph()

    # hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train raw_data: {0}\ndev raw_data: {1}".format(train_size, dev_size))
    # model.train(train_data, dev_data)

    # train model on the whole training raw_data
    print("train raw_data: {}".format(len(train_data)))
    # 训练
    model.train(train_data, dev_data)  # we could use test_data as the dev_data to see the overfitting phenomena


# testing model
elif args.mode == 'test':
    model_path = 'C:\\Users\\Administrator\\Desktop\\my_entity_reconginition\\bio_model\\1597547614\\checkpoints'
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embedding)
    model.build_graph()
    # test_size = 63177
    print("test raw_data: {}".format(test_size))
    # 测试
    model.test(test_data)


## demo
elif args.mode == 'demo':
    # 更改路径,路径问题
    print(model_path)  # .\bio_model\random_word_300\checkpoints/
    model_path = 'C:\\Users\\Administrator\\Desktop\\my_entity_reconginition\\bio_model\\1597256561\\checkpoints'
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim,
                       embeddings=embeddings,
                       dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip,
                       tag2label=tag2label, vocab=word2id, shuffle=args.shuffle,
                       model_path=ckpt_file, summary_path=summary_path, log_path=log_path, result_path=result_path,
                       CRF=args.CRF, update_embedding=args.update_embedding)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)  # 读入已经训练好的模型

        # 测试一份文件内容
        sent = []
        with open('./content_4.txt','r',encoding='utf-8-sig') as fr:
            for line in fr:
                for char in line:
                    sent.append(char)
        sent_data = [(sent, ['O'] * len(sent))]
        tag = model.demo_one(sess, sent_data)
        for s, t in zip(sent, tag):   # # 打包为元组的列表
            print(s,t)    
        keys = ['Symptom', 'Check', 'Operation', 'Disease', 'Relative_disease','Drug', 'Registration_department', 'Body']
        symptom, check, operation, disease, relative_disease, drug, registration_department, body= get_entity_keys(tag, sent, keys )
        print('symptom:{}\ncheck:{}\noperation:{}\ndisease:{}\nrelative_disease:{}\ndrug:{}\nregistration_department:{}\nbody:{}\n'
        .format(symptom, check, operation, disease,relative_disease,drug, registration_department,body))

        # while(1): 
        #     print('Please input your sentence:')
        #     demo_sent = input()
        #     if demo_sent == '' or demo_sent.isspace():
        #         print('See you next time!')
        #         break
        #     else:
        #         demo_sent = list(demo_sent.strip())
        #         # label全都是'O'
        #         demo_data = [(demo_sent, ['O'] * len(demo_sent))]
        #         tag = model.demo_one(sess, demo_data)
        #         # entities = get_entity(tag, demo_sent)
        #         # print('ENTITY: {}\n'.format(entities))
        #         body, chec, cure, dise, symp = get_entity(tag, demo_sent)
        #         print('body:{}\nchec:{}\ncure:{}\ndise:{}\nsymp:{}\n'.format(body, chec, cure, dise, symp))
