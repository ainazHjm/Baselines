import csv
import model
import json
import copy
import torch as th
import numpy as np
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from time import ctime

def get_acc(args, gt, prds):
    with th.no_grad():
        n_gt = gt.data.clone()
        n_prds = prds.data.clone()
        n_gt[n_gt>=args.threshold]=1
        n_gt[n_gt<args.threshold]=0
        n_prds[n_prds>=args.threshold]=1
        n_prds[n_prds<args.threshold]=0
        return th.sum(n_prds==n_gt)

def validate(args, test_loader):
    with th.no_grad():
        sig = nn.Sigmoid()
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        if args.model == 'LinearLayer':
            trained_model = model.LinearLayer(args.feature_num)
        trained_model.load_state_dict(th.load(args.load_model))
        trained_model.to(device)
        print('[{}]: model is successfully loaded.'.format(ctime()))

        acc, res, label_pr, id_, label = 0, [], [], [], []
        test_iter = iter(test_loader)
        for _ in range(len(test_iter)):
            batch = test_iter.next()
            prds = sig(trained_model.predict(batch['data'].to(device)))
            id_.append(batch['id'])
            res.append(prds)
            gt = batch['gt'].to(device)
            label_pr.append(gt)
            gt_binary = gt.data.clone()
            gt_binary[gt_binary>=args.threshold]=1
            gt_binary[gt_binary<args.threshold]=0
            label.append(gt_binary)
            acc += get_acc(args, gt, prds)
        total = len(test_iter)*args.batch_size
        print('[%s]: accuracy is %f >>> total true predictions: %d, number of test points: %d' %(ctime(), float(acc)/float(total), acc, total))
    
        f_id = th.cat((th.stack(id_[:-1]).view(-1, 1), id_[-1]), 0).data.numpy()
        f_res = th.cat((th.stack(res[:-1]).view(-1, 1), res[-1]), 0).data.numpy()
        f_gt_pr = th.cat((th.stack(label_pr[:-1]).view(-1, 1), label_pr[-1]), 0).data.numpy()
        f_gt = th.cat((th.stack(label[:-1]).view(-1, 1), label[-1]), 0).data.numpy()
        return f_id, f_res, f_gt_pr, f_gt

def write_results(args, id_, res, gt_pr, gt):
    path = args.save_res_to + 'predictions.csv'
    names = ['id', 'prediction', 'label_pr', 'label_binary']
    with open(path, 'w') as f:
        f_w = csv.writer(f)
        f_w.writerow(names)
        for row in range(res.shape[0]):
            write = []
            write.extend([int(id_[row])])
            write.extend(res[row])
            write.extend(gt_pr[row])
            write.extend([int(gt[row])])
            f_w.writerow(write)

def plot_curves(args, preds, gt):
    print('double checking the preds shape: (%d, %d)' %preds.shape)
    print('postive rate in the data: %f' % (np.sum(gt==1)/gt.shape[0]))
    fpr, tpr, threshold = metrics.roc_curve(gt, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
