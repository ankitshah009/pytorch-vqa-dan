import sys
import torch
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_acc(acc, figname):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.axhline(y=0.643, color='r', linestyle='-',label='Paper test acc')
    plt.plot(acc,label='Our val acc')
    plt.legend()
    plt.savefig(figname)

def main():
    path = sys.argv[1]
    results = torch.load(path,map_location={'cuda:0': 'cpu'})

    val_acc = torch.FloatTensor(results['tracker']['val_acc'])
    val_acc = val_acc.mean(dim=1).numpy()
    
    
    plot_acc(val_acc, 'val_acc.png')

if __name__ == '__main__':
    main()
