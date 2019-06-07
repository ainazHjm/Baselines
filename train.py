import model
import torch.nn as nn
import torch.optim as to
import torch as th
from tensorboardX import SummaryWriter
from time import ctime

def validate_model(args, model, data_loader):
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.tensor([args.weight]).cuda())
    data_loader_iter = iter(data_loader)
    running_loss = 0
    for _ in range(len(data_loader_iter)):
        batch = data_loader_iter.next()
        prd = model.predict(batch['data'].cuda())
        gt = batch['gt'].cuda()
        gt[gt >= args.threshold] = 1
        gt[gt < args.threshold] = 0
        loss = criterion(prd, gt)
        running_loss += loss.item()
    return running_loss/len(data_loader_iter)

def train_area(args, data_loader):
    train_model = model.Area(args.gt_path).cuda()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([args.loss_weight]).cuda())
    print('%s --- created the model.' %ctime())
    running_loss = 0
    num_iters = 0
    for _ in range(args.n_epochs):
        data_loader_iter = iter(data_loader)
        for _ in range(len(data_loader_iter)):
            batch = data_loader_iter.next()
            if args.loss_weight != 1:
                normalized_weight = 1-1/args.loss_weight # both are the same as we consider 1-y in the loss function
                print('%s --- normalized weights (pos, neg): %f, %f' %(ctime(), normalized_weight[0], 1-normalized_weight[0]))
                pw = (normalized_weight*batch['gt']).cuda()            
            else:
                pw = th.tensor([1.]).cuda()
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            # import ipdb; ipdb.set_trace()
            print('%s --- created the criterion with the normalizing weights.' %ctime())
            prds = train_model.predict(batch['data'].cuda())
            loss = criterion(prds.cuda(), batch['gt'].cuda())
            running_loss += loss.item()
            num_iters += 1
            print('loss: %f ... [%d]/[%d]' %(loss.item(), num_iters, args.n_epochs*len(data_loader_iter)), end='\r')
    th.save(train_model, args.save_model_to+args.model+'.pt')
    print('average running loss for %s: %f' %(args.model, running_loss/num_iters))
    return running_loss/num_iters

def train(args, train_data_loader, test_data_loader):
    print('starting to train ...')
    # criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([args.loss_weight]).cuda())
    if args.model == 'Area':
        loss = train_area(args, train_data_loader)
    else:
        if args.model == 'NoNghbr':
            train_model = model.NoNghbr(args.feature_num).cuda()
        elif args.model == 'Nghbr':
            train_model = model.Ngbhr(args.feature_num).cuda()
        elif args.model == 'LinearLayer':
            train_model = model.LinearLayer(args.feature_num).cuda()
        
        if args.load_model:
            train_model.load_state_dict(th.load(args.load_model).state_dict())

        writer = SummaryWriter()
        optimizer = to.Adam(train_model.parameters(), lr=args.lr, weight_decay=args.decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=th.tensor([args.weight]).cuda())
        scheduler = to.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

        running_loss = 0
        cum_train_loss = 0
        num_iters = 0
        for epoch in range(args.n_epochs):
            train_data_loader_iter = iter(train_data_loader)
            for iter_ in range(len(train_data_loader_iter)):
                optimizer.zero_grad()
                batch = train_data_loader_iter.next()
                prds = train_model.predict(batch['data'].cuda())
                gt = batch['gt'].cuda()
                gt[gt >= args.threshold] = 1
                gt[gt < args.threshold] = 0
                loss = criterion(prds, gt)
                num_iters += 1
                running_loss += loss.item()
                cum_train_loss += loss.item()
                # import ipdb; ipdb.set_trace()
                loss.backward()
                optimizer.step()
                writer.add_scalar('loss/train_iter', loss.item(), epoch*len(train_data_loader_iter)+iter_+1)
                print('loss: %f ... [%d]/[%d]' %(loss.item(), epoch*len(train_data_loader_iter)+iter_+1, args.n_epochs*len(train_data_loader_iter)), end='\r')
                if (epoch*len(train_data_loader_iter) + iter_ + 1) % 100 == 0:        
                    v_loss = validate_model(args, train_model, test_data_loader)
                    scheduler.step(v_loss)
                    writer.add_scalars('validation', {'validate': v_loss, 'train': cum_train_loss/100}, epoch*len(train_data_loader_iter)+iter_+1)
                    cum_train_loss = 0
            if (epoch+1)%args.s == 0:
                th.save(train_model, args.save_model_to+args.model+'_'+str(epoch+1)+'.pt')
        print('average running loss for %s: %f' %(args.model, running_loss/num_iters))
