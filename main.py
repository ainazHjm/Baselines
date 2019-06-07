import argparse
from loader import LandslideDataset, Sea2SkyDataset
from torch.utils.data import DataLoader
from train import train
from validate import validate_on_ones, validate, plot_curves

def get_args():
    parser = argparse.ArgumentParser(description="Training a CNN-Classifier for landslide prediction")
    parser.add_argument("--sea2sky", type=bool, default=False)
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument('--decay', type=float, default=1e-5)
    parser.add_argument("--model", type=str, default="Area")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=9)
    parser.add_argument("--num_workers", type=int, default=4)
    # parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument("--load_model", type=str, default='')
    parser.add_argument("--validate", type=bool, default=False)
    parser.add_argument("--data_path", type=str, default="/tmp/data/lzf_compressed.h5")
    parser.add_argument("--save_model_to", type=str, default="../models/baselines/")
    parser.add_argument("--region", type=str, default='Veneto')
    parser.add_argument("--pix_res", type=int, default=10)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--ws", type=int, default=200)
    parser.add_argument("--s", type=int, default=5) #save the model at how many epochs
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--feature_num", type=int, default=94)
    # parser.add_argument("--oversample_pts", action='append', type=__range)
    parser.add_argument('--gt_path', type=str, default='/home/ainaz/Projects/Landslides/image_data/Veneto/gt.tif')
    parser.add_argument('--loss_weight', type=float, default=1)
    parser.add_argument('--pos_indices', type=str, default='/home/ainaz/Projects/Landslides/image_data/pos_indices.npy')
    parser.add_argument('--save_res_to', type=str, default='results/Area/')
    return parser.parse_args()

def main():
    args = get_args()
    if args.sea2sky:
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
            preds, gt = validate(args, test_loader)
            plot_curves(args, preds.cpu().detach().numpy(), gt.cpu().detach().numpy())
            # tpr, tgt = validate(args, train_loader)
            # plot_curves(args, tpr.cpu().detach().numpy(), tgt.cpu().detach().numpy())
        else:
            train(args, train_loader, test_loader)

    else:
        testData = LandslideDataset(args.data_path, args.region, args.ws, 'test', args.pad)
        test_loader = DataLoader(
            testData,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        trainData = LandslideDataset(
                args.data_path,
                args.region,
                args.ws,
                'train',
                args.pad
        )
        train_loader = DataLoader(
            trainData,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        print('created the data loader.')
        if args.validate:
            validate_on_ones(args, testData)
        else:
            train(args, train_loader, test_loader)

if __name__=='__main__':
    main()
