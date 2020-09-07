"""工具类函数，存放代码块"""
import logging    # 处理日志
import sys      # 处理python运行时环境不同的部分
import argparse    # 处理命令行参数

# 第三步

def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # 首先被内层IOError异常捕获，打印“inner exception”, 然后把相同的异常再抛出，
        # 被外层的except捕获，打印"outter exception"
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 根据tag_seq和char_seq得到实体
def get_entity_keys(tag_seq, char_seq, keys):
    # entity = get_entity_one_(tag_seq, char_seq)
    # return entity
    entities = []
    for key in keys:
        entities.append(get_entity_key(tag_seq, char_seq, key))
    return entities



def get_entity_key(tag_seq, char_seq, key):
    entities = []
    entity = ''
    last_char = ''
    last_tag = ''
    for (char, tag) in zip(char_seq, tag_seq):
        if tag == 'B-' + key:
            if last_tag == 'I-' + key :
                entities.append(entity)
                entity = ''
                entity += char
            elif last_tag == '':
                entity = ''
                entity += char
            else:
                entity = ''
                entity += char

        elif tag == 'I-' + key:
            if last_tag == 'I-' + key or last_tag == 'B-' + key:
                entity += char
        #　如果Ｏ前元素大于等于２则判断为实体
        elif tag == 'O':
            if len(entity)>=2:
                entities.append(entity)
                entity = '' 

        last_tag = tag 

    if len(entity) != 0:
        entities.append(entity)
    return entities


# 将实体提取出来
def get_entity_one_(tag_seq, char_seq):
    sequence = []
    seq = ''
    for i, tag in enumerate(tag_seq):
        if tag == 'B' or tag == 'I':
            seq += char_seq[i]
        else:
            if len(seq) != 0:
                sequence.append(seq)
                seq = ''
    if len(seq) != 0:
        sequence.append(seq)
    return sequence

# 记录日志
def get_logger(filename):   # 返回日志文件
    logger = logging.getLogger('logger')   # 进行初始化
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

