import pickle
def test_pkl(self):
    with open('./train_test_data/char2id_bio.pkl', 'rb') as ff:
        data1 = pickle.load(ff)
        print(data1)
def trans_data_2():
    seqs = []
    with open('./content_3.txt','r',encoding='utf-8-sig') as fr:
        for line in fr:
            for char in line:
                seqs.append(char)




num = 0
def t_d():
    with open('./train_test_data/test_bio_char.txt','r',encoding='utf-8-sig') as fr:
        with open('test.txt','w') as f:
            for line in fr:
                if num > 31000:
                    f.write(line)
                    num = num + 1
                
            
            f.close()  
with open('./test.txt','r',encoding='utf-8-sig') as fr:
    with open('content_4.txt','w') as f:
        for line in fr:
            if num<=200:
                f.write(line[0])
                num = num + 1
        f.close()  


