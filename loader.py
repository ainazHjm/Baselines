from torch.utils.data import Dataset
import torch as th
import h5py

class Sea2SkyDataset(Dataset):
    def __init__(self, path, data_flag):
        self.path = path
        self.data_flag = data_flag

    def __len__(self):
        with h5py.File(self.path, 'r') as f:
            (h, _) = f[self.data_flag]['gt'].shape
            return h

    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as f:
            sample = {
                'data': f[self.data_flag]['data'][index, :],
                'gt': f[self.data_flag]['gt'][index, :],
                'id': f[self.data_flag]['id'][index, :]
            }
            return sample

class LandslideDataset(Dataset):
    '''
    This class doesn't support different stride sizes and oversampling.
    When testing, we don't need to have stride smaller than ws.
    Also, we don't need to oversample.
    '''
    def __init__(self, path, region, ws, data_flag, pad=64):
        super(LandslideDataset, self).__init__()
        self.path = path
        self.ws = ws
        self.region = region
        self.pad = pad
        self.data_flag = data_flag

    def __len__(self):
        with h5py.File(self.path, 'r') as f:
            (_, h, w) = f[self.region][self.data_flag]['gt'].shape
            hnum = h//self.ws
            wnum = w//self.ws
            return hnum*wnum
    
    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as f:
            dataset = f[self.region][self.data_flag]['data']
            gt = f[self.region][self.data_flag]['gt']
            (_, _, wlen) = gt.shape
            wnum = wlen//self.ws
            row = index//wnum
            col = index - row*wnum
            sample = {
                'data': th.tensor(
                    dataset[
                        :,
                        row*self.ws:(row+1)*self.ws+2*self.pad,
                        col*self.ws:(col+1)*self.ws+2*self.pad
                    ]
                ),
                'gt': th.tensor(
                    gt[
                        :,
                        row*self.ws:(row+1)*self.ws,
                        col*self.ws:(col+1)*self.ws
                    ]
                ),
                'index': (row, col)
            }
            return sample
