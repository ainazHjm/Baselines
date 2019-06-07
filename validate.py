import torch as th
import numpy as np
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def validate_on_ones(args, testData):
    sig = nn.Sigmoid()
    model = th.load(args.load_model)
    pos_indices = np.load(args.pos_indices)
    num_samples = 4
    samples = np.random.choice(pos_indices, num_samples)
    for i in range(num_samples):
        data = testData[samples[i]]['data']
        gt = testData[samples[i]]['gt']
        prds = sig(model.predict(data.view(1, args.feature_num, args.ws, args.ws).cuda()))[0, 0, :, :]
        np.save(args.save_res_to+str(i)+'.npy', prds.cpu().data.numpy())
        np.save(args.save_res_to+str(i)+'_gt.npy', gt.data.numpy())

def validate(args, test_loader):
    sig = nn.Sigmoid()
    model = th.load(args.load_model)
    acc = 0
    res = []
    label = []
    test_loader_iter = iter(test_loader)
    for _ in range(len(test_loader_iter)):
        batch = test_loader_iter.next()
        prds = sig(model.predict(batch['data'].cuda()))
        res.append(prds)
        gt = batch['gt'].cuda()
        gt[gt >= args.threshold] = 1
        gt[gt < args.threshold] = 0
        label.append(gt)
        n_prds = prds
        n_prds[n_prds >= args.threshold] = 1
        n_prds[n_prds < args.threshold] = 0
        acc += th.sum(n_prds == gt)
    total = len(test_loader_iter)*args.batch_size
    print('accuracy is %f >>> total true predictions: %d, number of test points: %d' %(float(acc)/float(total), acc, total))
    f_res = th.cat((th.stack(res[:-1]).view(-1, 1), res[-1]), 0)
    f_gt = th.cat((th.stack(label[:-1]).view(-1, 1), label[-1]), 0)
    return f_res, f_gt
        
def plot_curves(args, preds, gt):
    print('double checking the preds shape: (%d, %d)' %preds.shape)
    print('postive rate in the data: %f' % (np.sum(gt==1)/gt.shape[0]))
    fpr, tpr, threshold = metrics.roc_curve(gt, preds)
    print(threshold)
    print(np.sum(preds>1))
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
