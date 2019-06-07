import csv
import json
import h5py
import numpy as np

def normalize(f, feature_idx):
    mean = np.mean(f['train']['data'][:, feature_idx])
    std = np.std(f['train']['data'][:, feature_idx])
    # import ipdb; ipdb.set_trace()
    f['train']['data'][:, feature_idx] = (f['train']['data'][:, feature_idx]-mean)/std
    f['test']['data'][:, feature_idx] = (f['test']['data'][:, feature_idx]-mean)/std
    # import ipdb; ipdb.set_trace()
    return f

def create_dataset(path, shape):
    wr_path = '/'.join(path.split('.')[0].split('/')[:-1])
    print(wr_path)
    f = h5py.File(wr_path+'sea2sky.h5', 'a')
    (h, w) = shape
    idx = 0
    data_mat = np.zeros(shape)
    with open(path, 'r') as g:
        gr = csv.reader(g)
        for row in gr:
            assert len(row) == 137
            n_row = np.array([float(e) for e in row])
            if n_row[133] == 0.: #ignore 0 precipitation
                continue
            data_mat[idx, :] = n_row
            idx += 1
    print(idx, h) # idx is the new h
    filtered_data_mat = data_mat[0:idx, :]
    np.random.shuffle(filtered_data_mat)
    n_h = idx
    f.create_dataset('train/data', (n_h-n_h//5, w-1))
    f.create_dataset('train/gt', (n_h-n_h//5, 1))
    f.create_dataset('test/data', (n_h//5, w-1))
    f.create_dataset('test/gt', (n_h//5, 1))
    f['test']['data'][:, :] = filtered_data_mat[0:n_h//5, :-1]
    f['test']['gt'][:, :] = filtered_data_mat[0:n_h//5, -1].reshape(n_h//5, 1)
    f['train']['data'][:, :] = filtered_data_mat[n_h//5:, :-1]
    f['train']['gt'][:, :] = filtered_data_mat[n_h//5:, -1].reshape(n_h-n_h//5, 1)
    # import ipdb; ipdb.set_trace()
    f = normalize(f, 133)
    f.close()

def create_csv_table():
    features = {
        2: 'FID_River_',
        4: 'FID_Lake',
        5: 'FID_Ocean',
        6: 'FID_Permaf',
        9: 'MAT1',
        10: 'MAT2',
        11: 'TEXT1',
        12: 'TEXT2',
        13: 'EXP1',
        14: 'EXP2',
        15: 'GEOP_LBL',
        18: 'ROCK_TYPE',
        25: 'gridcode',
        31: 'FAULT_TYPE',
        38: 'AvgPrecip'
    }
    fd = open('data_dict.json')
    data_dict = json.load(fd)

    with open('data_table.csv', 'w') as f:
        fw = csv.writer(f)
        with open('all_attributes.csv', 'r') as g:
            gr = csv.reader(g)
            for row in gr:
                init_features = [0 for _ in range(len(data_dict))]
                if row[0] == 'target_id':
                    continue
                else:
                    for feature_num in features:
                        if feature_num < 9 or feature_num == 38:
                            if features[feature_num] in data_dict:
                                id_ = data_dict[features[feature_num]]
                                init_features[int(id_)] = row[feature_num] # a boolean value (0 or 1) or a real value (precip)
                        elif feature_num == 15 or feature_num==18:
                            for e in row[feature_num]:
                                if e in data_dict:
                                    init_features[int(data_dict[e])] = 1
                        else:
                            if row[feature_num] in data_dict:
                                id_ = data_dict[row[feature_num]]
                                init_features[int(id_)] = 1
                init_features.extend([float(row[-1])/100]) # the probability of the target (label)
                fw.writerow(init_features)
    fd.close()

def find_target(target_id):
    with open('s2s_target_scores_top10000.csv', 'r') as f:
        fr = csv.reader(f)
        for row in fr:
            if row[0] == target_id:
                return row[4]
        return -1
def merge_all_csv():
    with open('all_attributes.csv', 'w') as f:
        fw = csv.writer(f)
        with open('s2s_target_attributes_all.csv', 'r') as g:
            gr = csv.reader(g)
            for row in gr:
                if int(row[38]) == 0: #AvgPrecip is zero, ignore
                    print(row[0])
                    continue
                target = row[0]
                found = find_target(target)
                if found == -1:
                    continue
                else:
                    w_row = []
                    w_row.extend(row)
                    w_row.extend([found])                    
                    fw.writerow(w_row)