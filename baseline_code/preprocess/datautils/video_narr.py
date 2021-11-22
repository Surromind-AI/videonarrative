import os
import pandas as pd
import json
from datautils import utils
import nltk

import pickle
import numpy as np

import torch
import re
from konlpy.tag import Mecab

# nltk.download("punkt")

# 차원 확인용 함수
def check_glove_dimension():
    glove = pickle.load(open("glove.korean.pkl", 'rb'),encoding="cp949")
    dim_word = len(glove['그'])
    length = len(glove)
    print('dimension: ' + str(dim_word))
    print('key-length: ' + str(length))

def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    video_paths = []

    vid = []

    with open('{}/라벨링데이터/생활안전/대본X/output.json'.format(args.video_dir),'r') as annotation_file:
        instances = json.load(annotation_file)
        video_ids = []
    for instance in instances:
        video_ids.append(instance['vid'])
    for video_id in video_ids:
        video_paths.append(os.path.join(" {}/원천데이터/생활안전/대본X/".format(args.video_dir),video_id))

    with open('{}/라벨링데이터/생활안전/대본O/output.json'.format(args.video_dir),'r') as annotation_file:
        instances = json.load(annotation_file)
        video_ids = []
    for instance in instances:
        video_ids.append(instance['vid'])
    for video_id in video_ids:
        video_paths.append(os.path.join("{}/원천데이터/생활안전/대본O/".format(args.video_dir),video_id))

 
    with open('{}/라벨링데이터/스포츠/대본X/output.json'.format(args.video_dir),'r') as annotation_file:
        instances = json.load(annotation_file)
        video_ids = []
    for instance in instances:
        video_ids.append(instance['vid'])
    for video_id in video_ids:
        video_paths.append(os.path.join("{}/원천데이터/스포츠/대본X/".format(args.video_dir),video_id))


    with open('{}/라벨링데이터/예능교양/대본O/output.json'.format(args.video_dir),'r') as annotation_file:
        instances = json.load(annotation_file)
        video_ids = []
    for instance in instances:
        video_ids.append(instance['vid'])
    for video_id in video_ids:
        video_paths.append(os.path.join("{}/원천데이터/예능교양/대본O/".format(args.video_dir),video_id))


    with open('{}/라벨링데이터/예능교양/대본X/output.json'.format(args.video_dir),'r') as annotation_file:
        instances = json.load(annotation_file)
        video_ids = []
    for instance in instances:
        video_ids.append(instance['vid'])
    for video_id in video_ids:
        video_paths.append(os.path.join("{}/원천데이터/예능교양/대본X/".format(args.video_dir),video_id))


    return video_paths


def multi_encoding_data(args, vocab, questions, question_id, video_id, answers, answer_candidates, mode = 'train'):

    #Encode all questions
    print('Encoding data')
    m = Mecab().morphs
    questions_encoded = []
    questions_len =[]
    questions_id = question_id
    all_answer_candidate_encoded = []
    all_answer_candidate_lens = []
    video_id_tbw=[]
    correct_answers = []

    for idx, question in enumerate(questions):
        question_tokens = m(question)
        question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        questions_id.append(questions_id[idx])
        video_id_tbw.append(video_id[idx])

        # ground truth
        answer = int(answers[idx])
        correct_answers.append(answer)

        # answer candidates
        candidates = answer_candidates
        candidates_encoded = []
        candidates_len = []

        for answer in candidates:
            answer_tokens = m(answer)
            candidate_encoded = utils.encode(answer_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            candidates_encoded.append(candidate_encoded)
            candidates_len.append(len(candidate_encoded))
        all_answer_candidate_encoded.append(candidates_encoded)
        all_answer_candidate_lens.append(candidates_len)

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_answer_token_to_idx']['<NULL>'])

    questions_encoded =  np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    # Pad encoded answer candidates
    max_answer_candidate_length = max(max(len(x) for x in candidate) for candidate in all_answer_candidate_encoded)
    for ans_cands in all_answer_candidate_encoded:
        for ans in ans_cands:
            while len(ans) < max_answer_candidate_length:
                ans.append(vocab['question_answer_token_to_idx']['<NULL>'])

    all_answer_candidate_encoded = np.asarray(all_answer_candidate_encoded, dtype=np.int32)
    all_answer_candidate_lens = np.asarray(all_answer_candidate_lens, dtype=np.int32)
    print(all_answer_candidate_encoded.shape)

    glove_matrix = None
    print('Writing ', args.output_pt.format(args.dataset, args.dataset, mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': questions_id,
        'video_ids': np.asarray(video_id_tbw),
        'ans_candidates': all_answer_candidate_encoded,
        'ans_candidates_len': all_answer_candidate_lens,
        'answers': correct_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.dataset, args.dataset, mode), 'wb') as f:
        pickle.dump(obj,f)

def multichoice_encoding_data(args, vocab, questions,question_id, video_id, answers, answer_candidates, mode = 'train'):
    #Encode all questions
    print('Encoding data')
    m = Mecab().morphs
    questions_encoded = []
    questions_len =[]
    questions_id = question_id
    all_answer_candidate_encoded = []
    all_answer_candidate_lens = []
    video_id_tbw=[]
    correct_answers = []

    for idx, question in enumerate(questions):
        question_tokens = m(question)
        question_encoded = utils.encode(question_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        questions_id.append(questions_id[idx])
        video_id_tbw.append(video_id[idx])

        # ground truth
        answer = int(answers[idx])
        correct_answers.append(answer)

        # answer candidates
        candidates = answer_candidates[idx]
        candidates_encoded = []
        candidates_len = []

        for answer in candidates:
            answer_tokens = m(answer)
            candidate_encoded = utils.encode(answer_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            candidates_encoded.append(candidate_encoded)
            candidates_len.append(len(candidate_encoded))
        all_answer_candidate_encoded.append(candidates_encoded)
        all_answer_candidate_lens.append(candidates_len)

    # Pad encoded questions
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_answer_token_to_idx']['<NULL>'])

    questions_encoded =  np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    # Pad encoded answer candidates
    max_answer_candidate_length = max(max(len(x) for x in candidate) for candidate in all_answer_candidate_encoded)
    for ans_cands in all_answer_candidate_encoded:
        for ans in ans_cands:
            while len(ans) < max_answer_candidate_length:
                ans.append(vocab['question_answer_token_to_idx']['<NULL>'])

    all_answer_candidate_encoded = np.asarray(all_answer_candidate_encoded, dtype=np.int32)
    all_answer_candidate_lens = np.asarray(all_answer_candidate_lens, dtype=np.int32)
    print(all_answer_candidate_encoded.shape)

    glove_matrix = None
    if mode in ['train']:
        token_itow = {i: w for w, i in vocab['question_answer_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))

        dim_word = glove['the'].shape[0]

        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing ', args.output_pt.format(args.dataset, args.dataset, mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': questions_id,
        'video_ids': np.asarray(video_id_tbw),
        'ans_candidates': all_answer_candidate_encoded,
        'ans_candidates_len': all_answer_candidate_lens,
        'answers': correct_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.dataset, args.dataset, mode), 'wb') as f:
        pickle.dump(obj,f)

def process_question_multiChoices(args):
    print('Loading data')
    question_id = list([29201])
    questions = list([args.que])
    correct_idx = list([3])
    video = args.vid
    answer_candidates = np.asarray(args.answers)
    video_id = []
    script =[]

    if 'A' in video:
        sample_text = '1' + video[9:-4]
    elif 'B' in video:
        sample_text = '2' + video[9:-4]
    elif 'C' in video:
        sample_text = '3' + video[9:-4]
    elif 'D' in video:
        sample_text = '4' + video[9:-4]
    elif 'E' in video:
        sample_text = '5' + video[9:-4]
    elif 'F' in video:
        sample_text = '6' + video[9:-4]
    elif 'G' in video:
        sample_text = '7' + video[9:-4]
    elif 'H' in video:
        sample_text = '8' + video[9:-4]
    elif 'J' in video:
        sample_text = '9' + video[9:-4]
    elif 'K' in video:
        sample_text = '10' + video[9:-4]
    elif 'L' in video:
        sample_text = '11' + video[9:-4]
    elif 'M' in video:
        sample_text = '12' + video[9:-4]
    elif 'I' in video:
        sample_text = '13' + video[9:-4]
    video_id.append(int(sample_text))

    with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
        vocab = json.load(f)
    multi_encoding_data(args, vocab, questions, question_id, video_id, correct_idx, answer_candidates, mode='test')



def process_questions_mulchoices(args):
    print('Loading data')
    if args.mode in ["train", "val"]:
        json_data = pd.read_json('{}/train/라벨링데이터/생활안전/대본X/output.json'.format(args.video_dir))
        json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/생활안전/대본O/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/스포츠/대본X/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/예능교양/대본O/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/예능교양/대본X/output.json'.format(args.video_dir)))

    else:
        json_data = pd.read_json('{}/test/라벨링데이터/생활안전/대본X/output.json'.format(args.video_dir))
        json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/생활안전/대본O/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/스포츠/대본X/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/예능교양/대본O/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/예능교양/대본X/output.json'.format(args.video_dir)))

    # data 랜덤하게 split하기 위해서 permutation 사용.
    json_data = json_data.iloc[np.random.permutation(len(json_data))]
    question_id = list(json_data['qid'])
    questions = list(json_data['que'])
    correct_idx = list(json_data['correct_idx'])
    video_name = list(json_data['vid'])
    answer_candidates = np.asarray(json_data['answers'])
    summary = list(json_data['sum'])
    m = Mecab().morphs
    video_id = []
    script=[]

    # script 정보 load
    init_script = list(json_data['script'])
    script_exi = list(json_data['script_exi'])
    for idx, exi in enumerate(script_exi):
        if exi == 1:
            script.append(init_script[idx])

    # video_id
    for idx, video in enumerate(video_name):
        if 'A' in video:
            sample_text = '1' + video[11:-4]
        elif 'B' in video:
            sample_text = '2'+ video[11:-4]
        elif 'C' in video:
            sample_text = '3'+ video[11:-4]
        elif 'D' in video:
            sample_text = '4'+ video[11:-4]
        elif 'E' in video:
            sample_text = '5' + video[11:-4]
        elif 'F' in video:
            sample_text = '6' + video[11:-4]
        elif 'G' in video:
            sample_text = '7' + video[11:-4]
        elif 'H' in video:
            sample_text = '8' + video[11:-4]
        elif 'J' in video:
            sample_text = '9' + video[11:-4]
        elif 'K' in video:
            sample_text = '10' + video[11:-4]
        elif 'L' in video:
            sample_text = '11' + video[11:-4]
        elif 'M' in video:
            sample_text = '12' + video[11:-4]
        elif 'I' in video:
            sample_text = '13' + video[11:-4]

        print("print sample_text"+sample_text)
        video_id.append(int(sample_text))

    print(answer_candidates.shape)

    print('number of questions: %s' % len(questions))

    if args.mode in ['train']:
        print('Building vocab')

        answer_token_to_idx = {'<UNK0>':0, '<UNK1>':1 } # anwer_candidate에 대한 token 저장할 dictionary
        question_token_to_idx = {'<NULL>':1, '<UNK>':1} # questions에 대한 token 저장할 dictionary
        summ_token_to_idx = {'<NULL>':1, '<UNK>':1} # sum에 대한 token 저장할 dictionary
        question_answer_token_to_idx = {'<NULL>':0 , '<UNK>': 1} # question, answer, sum에 대한 token 저장할 dictionary
        script_token_to_idx = {'<NULL>':1, '<UNK>':1} # script에 대한 token 저장할 dictionary

        # 정답 후보에 대한 tokenize
        for candidates in answer_candidates:
            for answer in candidates:
                for token in m(answer):
                    if token not in answer_token_to_idx:
                        answer_token_to_idx[token] = len(answer_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
        print('Get answer_token_to_idx, num %d' % len(answer_token_to_idx))

        # 질문에 대한 tokenize
        for question in questions:
            for token in m(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(answer_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

        # 요약문에 대한 tokenize
        for summ in summary:
            for token in m(summ):
                if token not in summ_token_to_idx:
                    summ_token_to_idx[token] = len(summ_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)                  
        
        # 대본에 대한 tokenize
        for sc in script:
            if script is not None or script is not "NaN":
                for token in m(sc):
                    if token not in script_token_to_idx:
                        script_token_to_idx[token] = len(script_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
        

        print('Get answer_token_to_idx')
        print(len(answer_token_to_idx))
        print('Get summ_token_to_idx')
        print(len(summ_token_to_idx))
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))
        print('Get question_answer_token_to_idx')
        print(len(question_answer_token_to_idx))
        print('Get script_token_to_idx')
        print(len(script_token_to_idx))

        # vocab 생성
        vocab = {
        'question_token_to_idx': question_token_to_idx,
        'answer_token_to_idx': answer_token_to_idx,
        'sum_token_to_idx': summ_token_to_idx,
        'question_answer_token_to_idx' : question_answer_token_to_idx,
        'script_token_to_idx': script_token_to_idx
        }

        vocab_save_directory = args.vocab_json.format(args.dataset, args.dataset)
        print("Write into %s" % vocab_save_directory)

        with open(vocab_save_directory, 'w') as file:
            json.dump(vocab, file, indent=4)


        split = int(0.9*len(questions))
        train_questions = questions[:split]
        train_question_id = question_id[:split]
        train_answers = correct_idx[:split]
        train_video_id = video_id[:split]
        train_answer_candidates=answer_candidates[:split]


        val_questions = questions[split:]
        val_question_id = question_id[split:]
        val_answers = correct_idx[split:]
        val_video_id = video_id[split:]
        val_answer_candidates = answer_candidates[split:]

        print("number of train questions %s" % len(train_questions))
        print("number of validate questions %s" % len(val_questions))


        multichoice_encoding_data(args, vocab, train_questions, train_question_id, train_video_id, train_answers, train_answer_candidates, mode='train')
        multichoice_encoding_data(args, vocab, val_questions, val_question_id, val_video_id, val_answers,val_answer_candidates, mode='val')

    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset,args.dataset),'r') as f:
            vocab = json.load(f)
        multichoice_encoding_data(args, vocab, questions, question_id, video_id, correct_idx, answer_candidates, mode = 'test')
