import xlrd
data_tag = ['科室', '相关症状', '相关检查', '相关手术', '疾病', '相关疾病', '相关药品', '挂号科室', '发病部位']
english_tag = ['Department', 'Symptom', 'Check', 'Operation', 'Disease', 'Relative_disease',
               'Drug', 'Registration_department', 'Body']
entity_col = [1, 0, 10, 8, 20, 9, 3, 2, 5]  # 疾病 科室 挂号科室 相关疾病 药品 相关手术 相关检查 相关症状 身体部位
add_data = [[['属', '于', '的', '科', '室', '是'], ['O', 'O', 'O', 'O', 'O', 'O']], [['，'], ['O']], [['的', '挂', '号', '科', '室', '是'], ['O', 'O', 'O', 'O', 'O', 'O']], [['。'], ['O']], [['的', '相', '关', '疾', '病', '有'], ['O', 'O', 'O', 'O', 'O', 'O']], [[','], ['O']], [['应', '该', '吃', '的', '药', '品', '是'], ['O', 'O', 'O', 'O', 'O', 'O', 'O']], [['。'], ['O']], [['需', '要', '做', '的', '相', '关', '手', '术', '有'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']], [[','], ['O']], [['相', '关', '检', '查', '有'], ['O', 'O', 'O', 'O', 'O']], [['，'], ['O']], [['的', '相', '关', '症', '状', '有'], ['O', 'O', 'O', 'O', 'O', 'O']], [[','], ['O']], [['发', '生', '的', '身', '体', '部', '位', '在'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']], [['。'], ['O']]]
train_data =open('train_data.txt', mode='w')
book = xlrd.open_workbook('all_test.xlsx')
sheet1 = book.sheets()[0]
datas = []
for row in range(1, 2371):    # 读取第一列
    data = []
    for col in entity_col:
        value = sheet1.cell(row, col).value  # 取出当前行列的值
        if col == 1:
            entities = []
            entities.append(value.split())
            entities.append('Disease')
        if col == 10:    # 挂号科室
            entities = []
            entities.append(value.split())
            entities.append('Registration_department')
        if col == 8:
            entities = []
            entities.append(value.split())
            entities.append('Relative_disease')
        if col == 9:
            entities = []
            entities.append(value.split('、'))
            entities.append('Operation')
        if col == 3:
            entities = []
            entities.append(value.split('、'))
            entities.append('Check')
        if col == 2:
            entities = []
            entities.append(value.split('、'))
            entities.append('Symptom')
        if col == 0:
            entities = []
            entities.append(value.split())
            entities.append('Department')
        if col == 5:
            entities = []
            if '及' in value:
                entities.append(value.split('及'))
            elif '\\' in value:
                entities.append(value.split('\\'))
            else:
                entities.append(value.split())
            entities.append('Body')
        if col == 20:
            entities=[]
            if value == '无':
                entities.append('无'.split())   # 没有药品标记为O
                entities.append('O')
                continue
            if not value:
                continue
            else:
                entities.append(value.split())
                entities.append('Drug')
        data.append(entities)
    datas.append(data)  # 返回[[['腹痛'], 'Disease'], [['外科'], 'Department']]]


    """
        标注每个实体
        返回形式
        ['腹', '痛'] ['B-Disease', 'I-Disease']
        ['外', '科'] ['B-Department', 'I-Department']
    """
    p = 0
    disease_name = 0  # 记录第一个疾病名称
    disease_name_1 = []
    tt = 0  # 标记是否为第一次出现
    for entity_names, tag in data:
        if tt == 1:  # 为第一个疾病跳过
            tt = 0
            continue
        if disease_name == 1:
            for i in range(len(disease_name_1[0])):  # 写入疾病名称
                write2txt = disease_name_1[0][i] + ' ' + disease_name_1[1][i] + '\n'
                print(write2txt)
                train_data.write(write2txt)
            for j in range(len(add_data[p][0])):  #
                write2txt = add_data[p][0][j] + ' ' + add_data[p][1][j] + '\n'  # 写入付辅助语言
                print(write2txt)
                train_data.write(write2txt)
        for entity_name in entity_names:  # 在实体列表中查找
            chs = []
            names = []
            flag = 0
            if entity_name == '无':   # 如果没有药品不用标记
                chs.append('无')
                names.append('O')
            else:
                for ch in entity_name:
                    if flag == 0:    # 首字母标为B
                        new_tag = 'B-' + tag
                        flag = 1
                    else:
                        new_tag = 'I-' + tag
                    chs.append(ch)
                    names.append(new_tag)
                    if disease_name ==0:
                        disease_name_1.append(chs)
                        disease_name_1.append(names)  # [['腹', '痛'], ['B-Disease', 'I-Disease']]
                        disease_name = 1
                        tt = 1

            #print(chs, names)
            for i in range(len(chs)):  # 写入主要信息
                write2txt = chs[i] + ' ' + names[i] + '\n'
                print(write2txt)
                train_data.write(write2txt)
            # print(chs, names)
            # ['腹', '痛'] ['B-Disease', 'I-Disease']

        for j in range(len(add_data[p+1][0])):
            write2txt = add_data[p+1][0][j] + ' ' + add_data[p+1][1][j] + '\n'  # 写入付辅助语言
            train_data.write(write2txt)

            p = p + 2
        if p > 15:
            break


        print('======================')



#print(data)
train_data.close()

