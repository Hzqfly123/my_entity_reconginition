import numpy as np
import os
import time
import sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield, get_entity,label2tag
from utils import get_logger
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

#第二步:设置模型
class BiLSTM_CRF(object):
    def __init__(self, batch_size, epoch_num, hidden_dim, embeddings,
                 dropout_keep, optimizer, lr, clip_grad,
                 tag2label, vocab, shuffle,
                 model_path, summary_path, log_path, result_path,
                 CRF=True, update_embedding=True):
        # 批次大小
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        #drop操作参数
        self.dropout_keep_prob = dropout_keep
        self.optimizer = optimizer
        self.lr = lr
        self.clip_grad = clip_grad
        self.tag2label = tag2label
        #tag2label = {"O": 0,
             # "B-PER": 1, "I-PER": 2,
             # "B-LOC": 3, "I-LOC": 4,
             # "B-ORG": 5, "I-ORG": 6
             # }
        self.num_tags = len(tag2label)
        self.vocab = vocab  # word2id
        self.shuffle = shuffle
        self.model_path = model_path
        self.summary_path = summary_path
        self.logger = get_logger(log_path)
        self.result_path = result_path
        self.CRF = CRF
        self.update_embedding = update_embedding

    def build_graph(self):
        # 占位符
        self.add_placeholders()   
        self.lookup_layer_op()  # 返回句子*最大字个数*embedding
        self.biLSTM_layer_op() 
        self.softmax_pred_op()  # 如果不采用CRF方法的话，用argmax获取行中最大值的索引
        # 损失函数build
        self.loss_op()
        self.trainstep_op() # 优化器，默认adam
        # 初始化所有变量
        self.init_op()

    # 添加占位符
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        # 真实的标签序列
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        # 一个样本的真实序列长度
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        # dropout
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        # 学习率
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    # 找到每个word对应的编码
    def lookup_layer_op(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope("words"):
                _word_embeddings = tf.Variable(self.embeddings,  #3905*300的矩阵，矩阵元素均在-0.25到0.25之间
                                               dtype=tf.float32,
                                               #默认是True，如果为True，则会默认将变量添加到图形集合GraphKeys.TRAINABLE_VARIABLES中。#
# 此集合用于优化器Optimizer类优化的的默认变量列表【可为optimizer指定其他的变量集合】，可就是要训练的变量列表。这样的话在训练的过程中就会改变值
                                               trainable=self.update_embedding,
                                               name="_word_embeddings")
                word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                         ids=self.word_ids,
                                                         name="word_embeddings")
            self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)
            # params: 表示完整的嵌入张量，或者除了第一维度之外具有相同形状的P个张量的列表，表示经分割的嵌入张量
            #
            # ids: 一个类型为int32或int64的Tensor，包含要在params中查找的id
            #
            # partition_strategy: 指定分区策略的字符串，如果len（params） > 1，则相关。当前支持“div”和“mod”。 默认为“mod”
            #
            # name: 操作名称（可选）
            #
            # validate_indices:  是否验证收集索引
            #
            # max_norm: 如果不是None，嵌入值将被l2归一化为max_norm的值
            #
            #  
            #
            # tf.nn.embedding_lookup()
            # 函数的用法主要是选取一个张量里面索引对应的元素
            #
            # tf.nn.embedding_lookup(tensor, id)：即tensor就是输入的张量，id
            # 就是张量对应的索引


#先定义前后向RNN，再用tf.nn.bidirectional_dynamic_rnn来创建双向递归神经网络的动态版本，再dropout处理，
#定义权值w和b做矩阵乘法，再reshape一下（和W矩阵做乘法）
    def biLSTM_layer_op(self):
        #关于tf.variable_scope和tf.get_variable：https://blog.csdn.net/zSean/article/details/75057806
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim) #隐藏层神经元，默认300
            cell_bw = LSTMCell(self.hidden_dim)
            # def bidirectional_dynamic_rnn(
            #         cell_fw,  # 前向RNN
            #         cell_bw,  # 后向RNN
            #         inputs,  # 输入
            #         sequence_length=None,  # 输入序列的实际长度（可选，默认为输入序列的最大长度）
            #         initial_state_fw=None,  # 前向的初始化状态（可选）
            #         initial_state_bw=None,  # 后向的初始化状态（可选）
            #         dtype=None,  # 初始化和输出的数据类型（可选）
            #         parallel_iterations=None,
            #         swap_memory=False,
            #         time_major=False,
            #         # 决定了输入输出tensor的格式：如果为true, 向量的形状必须为 `[max_time, batch_size, depth]`.
            #         # 如果为false, tensor的形状必须为`[batch_size, max_time, depth]`.
            #         scope=None
            # )
            #outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的二元组。
            # 如果time_major == False(默认值), 则output_fw将是形状为[batch_size, max_time, cell_fw.output_size]
            # 的张量, 则output_bw将是形状为[batch_size, max_time, cell_bw.output_size]
            # 的张量；
            # 如果time_major == True, 则output_fw将是形状为[max_time, batch_size, cell_fw.output_size]
            # 的张量；output_bw将会是形状为[max_time, batch_size, cell_bw.output_size]
            # 的张量.
            # 与bidirectional_rnn不同, 它返回一个元组而不是单个连接的张量.如果优选连接的, 则正向和反向输出可以连接为tf.concat(outputs, 2).
 
            # output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的二元组。 
            # output_state_fw和output_state_bw的类型为LSTMStateTuple。 
            # LSTMStateTuple由（c，h）组成，分别代表memory
            # cell和hidden（即c,h矩阵）
            # state。
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(   # 创建双向递归神经网络的动态版本
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                # 输入  [batch_szie, max_time, depth]depth=self.hidden_dim=300，max_time可以为句子的长度
                #（一般以最长的句子为准，短句需要做padding），depth为输入句子词向量的维度
                inputs=self.word_embeddings, 
                sequence_length=self.sequence_lengths, # 输入序列的实际长度（可选，默认为输入序列的最大长度）
                dtype=tf.float32) # 数据类型
                #则output_fw将是形状为[batch_size, max_time, cell_fw.output_size],
            # 的张量, 则output_bw将是形状为[batch_size, max_time, cell_bw.output_size]
            # 维持行数不变，后面的行接到前面的行后面  示例程序在tt.py
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)#[batch_size, max_time, 600]-1是按行的意思
            # model 143
            # (?, ?, 300)
            # (?, ?, 300)
            # model 151
            # (?, ?, 600)
            #经过droupput处理
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags], #[600,7]
                                 # 该函数返回一个用于初始化权重的初始化程序 “Xavier” 。
                                # 这个初始化器是用来保持每一层的梯度大小都差不多相同
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags], #[7]
                                 # tf.zeros_initializer()，也可以简写为tf.Zeros()
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            # output的形状为[batch_size,steps,cell_num]批次大小，步长，神经元个数=600
            s = tf.shape(output)
            # print(output.shape)
            # reshape的目的是为了跟w做矩阵乘法
            output = tf.reshape(output, [-1, 2*self.hidden_dim]) #-1就是未知值，是批次大小
            pred = tf.matmul(output, W) + b  #[batch_size,self.num_tags]
            # s[1]=batch_size
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])


    def loss_op(self):
        if self.CRF:
            # crf_log_likelihood作为损失函数
            # inputs：unary potentials,就是每个标签的预测概率值
            # tag_indices，这个就是真实的标签序列了
            # sequence_lengths,一个样本真实的序列长度，为了对齐长度会做些padding，但是可以把真实的长度放到这个参数里
            # transition_params,转移概率，可以没有，没有的话这个函数也会算出来
            # 输出：log_likelihood:标量;transition_params,转移概率，如果输入没输，它就自己算个给返回
            log_likelihood, self.transition_params = crf_log_likelihood(
                inputs=self.logits, tag_indices=self.labels, sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            # 交叉熵做损失函数
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=self.labels)
            #张量变换函数 
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        #添加标量统计结果
        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)  #-1表示按行取值最大的索引
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            #minimize()实际上包含了两个步骤，即compute_gradients和apply_gradients，前者用于计算梯度，后者用于使用计算得到的梯度来更新对应的variable 
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)


    def train(self, train, dev):
        """

        :param train:
        :param dev:
        :return:
        """ 
        #train_data的形状为[(['我',在'北','京'],['O','O','B-LOC','I-LOC'])...第一句话
        #(['我',在'天','安','门'],['O','O','B-LOC','I-LOC','I-LOC'])...第二句话
        #( 第三句话 )  ] 总共有50658句话
        saver = tf.train.Saver(tf.global_variables()) # saver引入已经训练好的模型，然后输入句子，保存和加载模型
        # 如果句子合法，首先以字为单位划分，然后初始化标签为O
        #[(['小', '明', '的', '大', '学', '在', '北', '京', '的', '北', '京', '大', '学'],
        # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])]
        with tf.Session() as sess:
            sess.run(self.init_op) 
            self.add_summary(sess)
            # #epoch_num=100
            for epoch in range(self.epoch_num):
                # ['B-PER', 'I-PER', 0, 0, 0, 0, 'B-LOC', 'I-LOC', 0, 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            # self.evaluate(label_list, seq_len_list, test)
            self.evaluate_(label_list, test)

    # 用模型测试一个句子
    # 一得到预测标签id，第二转化为label
    def demo_one(self, sess, sent):   #[['疾',,,]['O',,,]]
        """

        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            # print('model 268行:')
            # model 268 行:以  小明的大学在北京的北京大学  为例
            # [[841, 37, 8, 55, 485, 73, 87, 74, 8, 87, 74, 55, 485]]
            # 可见batch_yield就是把输入的句子每个字的id返回，以及每个标签转化为对应的tag2label的值
            # [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            label_list_, _ = self.predict_one_batch(sess, seqs) # seqs为['疾',,,]*64,['O',,,]*64     #为空
            # print('model 275行')
            # print(_)
            # model 275 行
            # [[1, 2, 0, 0, 0, 0, 3, 4, 0, 5, 6, 6, 6]]
            # [13]
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        # print(label2tag)  {0: 0, 1: 'B-Symptom', 2: 'B-Check', 3: 'B-Operation'}
        
        tag = [label2tag[label] for label in label_list[0]]

        # model 304
        # {0: 0, 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC', 5: 'B-ORG', 6: 'I-ORG'}
        # model 307
        # ['B-PER', 'I-PER', 0, 0, 0, 0, 'B-LOC', 'I-LOC', 0, 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']
        # print(tag)
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver): # 训练数据，测试数据，tag2label，迭代索引，saver
        """
        每次训练的一个epoch，在每个epoch里面有多个batch_size
        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        # # 计算出多少个batch，计算过程：(50658+300-1)//300=792
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        # # 记录开始训练的时间
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 产生每一个batch，分批训练
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            # sys.stdout 是标准输出文件，write就是往这个文件写数据
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            # step_num=epoch*792+step+1
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            #开头后每相隔300记录一次，最后再记录一次
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))
            # 可视化
            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                 # 训练的最后一个batch保存模型
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)  #将test_data传过去
        # self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)
        self.evaluate_(label_list_dev, dev)

    #占位符赋值
    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        # seq_len_list用来统计每个样本的真实长度
        # word_ids就是seq_list，padding后的样本序列
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
             # labels经过padding后，喂给feed_dict
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        # seq_len_list用来统计每个样本的真实长度
        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        #获取一个批次的句子中词的id以及标签
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        # seq_len_list用来统计每个样本的真实长度
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)   # seqs 为[300]*64 [300]*64

        if self.CRF:
             # transition_params代表转移概率，由crf_log_likelihood方法计算出
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            # 打包成元素形式为元组的列表[(logit,seq_len),(logit,seq_len),( ,),]
            #如果是demo情况下，输入句子，那么只有一个句子，所以只循环一次，训练模式下就不会
            #对logits解析得到一个数
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else: #如果不用CRF，就是把self.logits每行取最大的
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            # tag2label = {"O": 0,
            #              "B-PER": 1, "I-PER": 2,
            #              "B-LOC": 3, "I-LOC": 4,
            #              "B-ORG": 5, "I-ORG": 6
            #              }
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                # tag是真实值，tag_是预测值
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)

    # 使用sklearn进行模型评估
    def evaluate_(self, label_list, data):
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag
        contents = []
        y_test = []
        y_pred = []
        for content, label in data:
            contents.extend([con for con in content])
            y_test.extend([y for y in label])
        for label in label_list:
            y_pred.extend([label2tag[y] for y in label])
        entity = ['Symptom', 'Check', 'Operation', 'Disease', 'Relative_disease',
               'Drug', 'Registration_department', 'Body']
        entities_pred = get_entity(y_pred, contents, entity)
        entities_true = get_entity(y_test, contents, entity)
        pre_all = 0
        rec_all = 0
        f1_all = 0
        for i, (pred, true) in enumerate(zip(entities_pred, entities_true)):
            pre = 0
            rec = 0
            for p in pred:
                if p in true:
                    pre += 1
            for t in true:
                if t in pred:
                    rec += 1
            precision = 0.1
            recall = 0.1
            if len(pred) != 0:
                precision = pre * 1.0 / len(pred)
            if len(true) != 0:
                recall = rec * 1.0 / len(true)
            if precision == 0 and recall == 0:
                recall = 0.1
                precision = 0.1
            f1 = 2 * precision * recall / (precision + recall)
            pre_all += precision
            rec_all += recall
            f1_all += f1
            print('{:10s}: precision:{:.2f}, recall:{:.2f}, f1-score:{:.2f}'.format(entity[i], precision, recall, f1))

        print('{:10s}: precision:{:.2f}, recall:{:.2f}, f1-score:{:.2f}'.format(
            'average', pre_all / len(entity), rec_all / len(entity), f1_all / len(entity)))