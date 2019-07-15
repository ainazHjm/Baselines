import model
import os
import torch.nn as nn
import torch.optim as to
import torch as th
from tensorboardX import SummaryWriter
from time import ctime

def create_dir(dir_name):
    model_dir = dir_name+'/model/'
    res_dir = dir_name+'/result/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    return model_dir, res_dir

def validate_model(args, model, data_loader):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.tensor([args.pos_weight]).to(device))
    data_loader_iter = iter(data_loader)
    running_loss = 0
    for _ in range(len(data_loader_iter)):
        batch = data_loader_iter.next()
        prd = model.predict(batch['data'].to(device))
        gt = batch['gt'].to(device)
        gt[gt >= args.threshold] = 1
        gt[gt < args.threshold] = 0
        loss = criterion(prd, gt)
        running_loss += loss.item()
    return running_loss/len(data_loader_iter)

def train(args, train_data_loader, test_data_loader):
    print('[{}]: starting to train ...'.format(ctime()))
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    if args.model == 'LinearLayer':
        train_model = model.LinearLayer(args.feature_num) 
    if args.load_model:
        train_model.load_state_dict(th.load(args.load_model).state_dict())
    train_model.to(device)
    print('[{}]: model is loaded.'.format(ctime()))

    writer = SummaryWriter()
    model_dir, _ = create_dir(writer.file_writer.get_logdir())
    optimizer = to.Adam(train_model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.tensor([args.pos_weight]).to(device))
    scheduler = to.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
    print('[{}]: optimizer, loss fucntion, and the scheduler are instantiated.'.format(ctime()))
    
    running_loss, cum_train_loss = 0, 0
    for epoch in range(args.n_epochs):
        train_iter = iter(train_data_loader)
        for iter_ in range(len(train_iter)):
            optimizer.zero_grad()

            batch = train_iter.next()
            prds = train_model.predict(batch['data'].to(device))
            gt = batch['gt'].to(device)
            gt[gt >= args.threshold] = 1
            gt[gt < args.threshold] = 0
            loss = criterion(prds, gt)
            running_loss += loss.item()
            cum_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('loss/train_iter', loss.item(), epoch*len(train_iter)+iter_+1)
            if (epoch*len(train_iter) + iter_ + 1) % 100 == 0:        
                v_loss = validate_model(args, train_model, test_data_loader)
                scheduler.step(v_loss)
                writer.add_scalars('loss', {'validate': v_loss, 'train': cum_train_loss/100}, epoch*len(train_iter)+iter_+1)
                print('[%s]: training loss is %0.4f and validation loss is %0.4f at [%d]/[%d]' %(
                    ctime(),
                    cum_train_loss/100,
                    v_loss,
                    epoch*len(train_iter)+iter_+1,
                    args.n_epochs*len(train_iter))
                )  
                cum_train_loss = 0
        if (epoch+1)%args.save == 0:
            th.save(train_model.state_dict(), model_dir+'modelat{}.pt'.format(str(epoch+1)))
    th.save(train_model.state_dict(), model_dir+'final_model.pt')
    print('[%s]: average running loss for %s is %0.4f' %(ctime(), args.model, running_loss/(args.n_epochs*len(train_iter))))