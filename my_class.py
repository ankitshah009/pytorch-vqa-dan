from pickle import load

import sys
from data_generator import DataGenerator
sys.path.insert(0, '/media/shijie/OS/Users/WUSHI/github/MovieQA_benchmark')
import data_loader

mqa = data_loader.DataLoader()
vl_qa, training_qas = mqa.get_video_list('train', 'qa_clips')
import pdb; pdb.set_trace()
for qa in training_qas[:4]:
    video_base = '/media/shijie/OS/Users/WUSHI/github/Multiple-Attention-Model-for-MovieQA/data/data_processed'
    subtt_base = '/media/shijie/OS/Users/WUSHI/github/Multiple-Attention-Model-for-MovieQA/data/subtt'

    # videos
    video = []
    # subtitles
    subtt = []
    # qas
    qas = []

    for clip in qa.video_clips:
        video_abs_path = '{}/{}features.p'.format(video_base, clip)
        frame_feature = load(open(video_abs_path, 'rb'))
        video.extend(frame_feature)

        sub_abs_path = '{}/{}.p'.format(subtt_base, clip)
        line = load(open(sub_abs_path, 'rb'))
        subtt.extend(line)

    question = qa.question
    answers = qa.answers
    print("{}\n{}\n{}\n{}\n".format([v.shape for v in video], subtt, question, answers))
    import pdb; pdb.set_trace()
