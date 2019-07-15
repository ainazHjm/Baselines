import argparse
from loader import LandslideDataset, Sea2SkyDataset
from torch.utils.data import DataLoader
from train import train
from validate import write_results, validate, plot_curves

def get_args():
    parser = argparse.ArgumentParser(description="Training a CNN-Classifier for landslide prediction")
    parser.add_argument('--pos_weight', type=float, default=1.)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--decay', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default="LinearLayer")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=9)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--region', type=str, default='Veneto')
    parser.add_argument('--pix_res', type=int, default=10)
    parser.add_argument('--save', type=int, default=1) # save the model at how many epochs
    parser.add_argument('--feature_num', type=int, default=136)
    parser.add_argument('--save_res_to', type=str, default='')
    parser.add_argument('--validate', type=bool, default=False)
    parser.add_argument('--path2json', type=str, default='')
    return parser.parse_args()

def main():
    args = get_args()
    trainData = Sea2SkyDataset(args.data_path, 'train')
    train_loader = DataLoader(
        trainData,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    testData = Sea2SkyDataset(args.data_path, 'test')
    test_loader = DataLoader(
        testData,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    if args.validate:
        id_, preds, gt_pr, gt = validate(args, test_loader)
        write_results(args, id_, preds, gt_pr, gt)
        plot_curves(args, preds, gt)
        # tpr, tgt = validate(args, train_loader)
        # plot_curves(args, tpr.cpu().detach().numpy(), tgt.cpu().detach().numpy())
    else:
        train(args, train_loader, test_loader)

if __name__=='__main__':
    main()
