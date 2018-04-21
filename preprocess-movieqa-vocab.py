import argparse
from collections import Counter
import os
import pickle
import sys
sys.path.insert(0, '/Users/iammrhelo/Courses/10701/MovieQA_benchmark')

import spacy
from tqdm import tqdm

from data_loader import DataLoader

subtt_base = './movieqa/subtt'

nlp = spacy.load('en')

_PAD_IDX = 0
_UNK_IDX = 1

def tokenize(text):
    doc = nlp(text, disable=['parser', 'tagger', 'ner'])
    tokens = [ token.text for token in doc ]
    return tokens

def tokenize_and_index(text, vocab):
    tokens = tokenize(text)
    word_indices = []
    for word in tokens:
        index = vocab.get(word,_UNK_IDX)
        word_indices.append(index)
    return word_indices

def load_subtitles(video_clips):
    # subtitles
    subtt = []
    # Get subtitles
    for clip in video_clips:
        sub_abs_path = '{}/{}.p'.format(subtt_base, clip)
        with open(sub_abs_path,'rb') as fin:
            try:
                lines = pickle.load(fin)
            except ValueError:
                lines = pickle.load(fin, encoding='latin1') 
            # Preprocess subtitles
            lines = [ sent.replace('-',' ').strip() for sent in lines ]
        subtt.extend(lines)
    
    subtt = " ".join(subtt)

    return subtt

def build_vocab(vl_qa, qas, min_freq):

    counter = Counter()

    for qa in tqdm(qas):

        subtitles = load_subtitles(qa.video_clips)

        text = ' '.join([ qa.question.lower(),' '.join(qa.answers).lower(), subtitles.lower() ])

        tokens = tokenize(text)

        for word in tokens:
            counter.update([word])

    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    
    print("vocab",len(words_and_frequencies))

    itos = []

    for word, freq in words_and_frequencies:
        if freq < min_freq:
            break
        itos.append(word)

    stoi = dict()
    stoi["<pad>"] = _PAD_IDX # Pad token as 0
    stoi["<unk>"] = _UNK_IDX # Unknown word as 1

    stoi.update({tok: i for i, tok in enumerate(itos)})
    print("min_freq", min_freq,'vocab',len(stoi))
    return stoi

def preprocess_vocab(vl_qa, qas, vocab):
    # The data to return
    data = {}
    for qa in tqdm(qas):
        # subtitles
        subtt = []
        
        # Get subtitles
        for clip in qa.video_clips:
            sub_abs_path = '{}/{}.p'.format(subtt_base, clip)
            with open(sub_abs_path,'rb') as fin:
                try:
                    lines = pickle.load(fin)
                except ValueError:
                    lines = pickle.load(fin, encoding='latin1') 
                # Preprocess subtitles
                lines = [ sent.replace('-',' ').strip() for sent in lines ]
            subtt.extend(lines)
        
        subtt = " ".join(subtt)

        # Start tokenizing
        question = tokenize_and_index(qa.question, vocab)
        answers = [ tokenize_and_index(ans, vocab) for ans in qa.answers ]
        subtitles = tokenize_and_index(subtt, vocab)

        data[ qa.qid ] = {
            'question': question,
            'answers': answers,
            'subtitles': subtitles,
            'correct_index': qa.correct_index
        }
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str,default='movieqa')
    parser.add_argument('--min_freq',type=int,default=10)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    vocab_pickle = os.path.join(args.dir,'movieqa.vocab')

    mqa = DataLoader()

    if not os.path.exists(vocab_pickle):
        train_vl_qa, training_qas = mqa.get_video_list('train', 'qa_clips')
        vocab = build_vocab(train_vl_qa, training_qas, args.min_freq)
        with open(vocab_pickle,'wb') as fout:
            pickle.dump(vocab,fout)
    else:
        with open(vocab_pickle,'rb') as fin:
            vocab = pickle.load(fin)

    for split in ['train', 'val','test']:
        print("Processing",split + "...")
        vl_qa, qas = mqa.get_video_list(split, 'qa_clips')
        data = preprocess_vocab(vl_qa, qas, vocab)

        data_pickle = os.path.join(args.dir,"movieqa.{}.pickle".format(split))
        with open(data_pickle,'wb') as fout:
            pickle.dump(data, fout)