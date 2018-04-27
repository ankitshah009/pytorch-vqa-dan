import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import movie
from dan import TextEncoder, MovieDAN
import utils

torch.backends.cudnn.enabled = False

def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

total_iterations = 0

def run(net, dataset, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
    else:
        net.eval()

    niter = int(len(dataset) / dataset.batch_size)
    tq = tqdm(dataset.loader(), desc='{} E{:03d}'.format(prefix, epoch), ncols=0, total=niter)
    #loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    #acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    criterion = nn.CrossEntropyLoss().cuda()
   
    total_count = 0
    total_acc = 0
    total_loss = 0

    for q, s, la, c in tq:
        var_params = {
            'volatile': not train,
            'requires_grad': False,
        }
        q = Variable(q.cuda(async=True))
       	s = Variable(s.cuda(async=True))
        la = [ Variable(a.cuda(async=True)) for a in la ]
        c = Variable(c.cuda(async=False)) # correct answers
        
        out = net(q, s, la)
        loss = criterion(out, c)

     
        # Compute our own accuracy
        _, pred = out.data.max(dim=1)
        acc = (pred == c.data).float()

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iterations += 1
        
        total_count += acc.shape[0]
        total_loss += loss.data[0] * acc.shape[0]
        total_acc += acc.sum()
	
    
    acc = total_acc / total_count
    loss = total_loss / total_count
    print("loss",loss,"acc",acc)    

def main():
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_dataset = movie.get_dataset(train=True)
    val_dataset = movie.get_dataset(val=True)
    
    # Build Model
    vocab_size = len(train_dataset.vocab)
    model = MovieDAN(num_embeddings=vocab_size, 
                embedding_dim=config.embedding_dim, 
                hidden_size=config.hidden_size, 
                answer_size=config.movie_answer_size)

    net = nn.DataParallel(model).cuda()
    
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        run(net, train_dataset, optimizer, tracker, train=True, prefix='train', epoch=i)
        run(net, val_dataset, optimizer, tracker, train=False, prefix='val', epoch=i)

if __name__ == '__main__':
    main()
