import json
import codecs
import os
import pickle


puncs = [',', '.', '?', '<', '>', '\'', '"', '`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=',
         '+', ';', ':', '/', '|', '\\', '[', ']', '{', '}']


def process(s):
    return ''.join(c for c in s if c not in puncs).lower()


def create_directory(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_data(in_data_dir, out_data_dir, out_fname, q_fname, a_fname=None, k=1000):
    create_directory(out_data_dir)
    out_file_data = out_data_dir + out_fname + '_data.csv'
    qfile = in_data_dir + q_fname
    qf = codecs.open(qfile, 'r', encoding='utf-8')
    ques = json.load(qf)['questions']

    if a_fname is not None:
        out_file_ans = out_data_dir + 'aid_ans_' + str(k) + '.pickle'
        afile = in_data_dir + a_fname
        af = codecs.open(afile, 'r', encoding='utf-8')
        ans = json.load(af)['annotations']

    if a_fname is None:
        data = []
        for q in ques:
            qtext = q['question']
            q_id = q['question_id']
            image_id = q['image_id']
            data.append((image_id, process(qtext)))

        print("Length of Final Data : ", len(data))
        f = open(out_file_data, 'w')
        for d in data:
            f.write(str(d[0]) + ',' + d[1] + '\n')
        f.close()

    else:
        qid_q_dict = {}
        ans_count = {}
        data = []

        for q in ques:
            qtext = q['question']
            q_id = q['question_id']
            qid_q_dict[q_id] = qtext

        for a in ans:
            atext = a['multiple_choice_answer']
            image_id = a['image_id']
            q_id = a['question_id']
            qtext = qid_q_dict[q_id]
            if atext not in ans_count:
                ans_count[atext] = 0
            ans_count[atext] += 1
            data.append((image_id, process(qtext), process(atext)))

        if os.path.exists(out_file_ans):
            with open(out_file_ans, 'rb') as f:
                ans_dict = pickle.load(f)
        else:
            ans_freq = []
            top_ans = []
            for a in ans_count:
                ans_freq.append((a, ans_count[a]))
            ans_freq = sorted(ans_freq, key=lambda x: x[1], reverse=True)
            for i in range(k):
                top_ans.append(ans_freq[i][0])

            ans_dict = {}
            ctr = 0
            for a in top_ans:
                if a not in ans_dict:
                    ans_dict[a] = ctr
                    ctr += 1
            with open(out_file_ans, 'wb') as f:
                pickle.dump(ans_dict, f)

        fin_data = []
        for d in data:
            if d[2] not in ans_dict:
                continue
            fin_data.append((d[0], d[1], ans_dict[d[2]]))
        print("Length of Final Data : ", len(fin_data))
        f = open(out_file_data, 'w')
        for d in fin_data:
            f.write(str(d[0]) + ',' + d[1] + ',' + str(d[2]) + '\n')
        f.close()


k = 1000
in_data_dir = 'Raw_Data/'
train_q_fname = 'v2_OpenEnded_mscoco_train2014_questions.json'
train_a_fname = 'v2_mscoco_train2014_annotations.json'
val_q_fname = 'v2_OpenEnded_mscoco_val2014_questions.json'
val_a_fname = 'v2_mscoco_val2014_annotations.json'
test1_q_fname = 'v2_OpenEnded_mscoco_test-dev2015_questions.json'
test2_q_fname = 'v2_OpenEnded_mscoco_test2015_questions.json'
out_data_dir = 'datasets/'
train_out_fname = 'train_' + str(k)
val_out_fname = 'val_' + str(k)
test1_out_fname = 'test-dev'
test2_out_fname = 'test'

generate_data(in_data_dir, out_data_dir, train_out_fname, train_q_fname, train_a_fname, k)
generate_data(in_data_dir, out_data_dir, val_out_fname, val_q_fname, val_a_fname, k)
generate_data(in_data_dir, out_data_dir, test1_out_fname, test1_q_fname)
generate_data(in_data_dir, out_data_dir, test2_out_fname, test2_q_fname)
