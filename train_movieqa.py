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

def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

total_iterations = 0

def run(net, dataset, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        accs = []
    
    tq = tqdm(dataset.loader(), desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()
   
    for q, s, la, c in tq:
        var_params = {
            'volatile': not train,
            'requires_grad': False,
        }
        #q = Variable(q.cuda(async=True), **var_params)
        #s = Variable(s.cuda(async=True), **var_params)
        #la = [ Variable(a.cuda(async=True), **var_params) for a in la ]
        #c = Variable(c.cuda(async=False), **var_params) # correct answers

        q = Variable(q, **var_params)
        s = Variable(s, **var_params)
        la = [ Variable(a, **var_params) for a in la ]
        c = Variable(c, **var_params) # correct answers
        
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
        else:
            # store information about evaluation of this minibatch
            #_, answer = out.data.cpu().max(dim=1)
            _, answer = out.data.max(dim=1)
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))

        loss_tracker.append(loss.data[0])
        acc_tracker.append(acc.mean())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        accs = list(torch.cat(accs, dim=0))
        return answ, accs


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

    #net = nn.DataParallel(model).cuda()
    net = nn.DataParallel(model)
    
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        _ = run(net, train_dataset, optimizer, tracker, train=True, prefix='train', epoch=i)
        r = run(net, val_dataset, optimizer, tracker, train=False, prefix='val', epoch=i)

        results = {
            'name': name,
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            'eval': {
                'answers': r[0],
                'accuracies': r[1],
            },
            'vocab': train_dataset.vocab,
        }
        torch.save(results, target_name)


if __name__ == '__main__':
    main()
