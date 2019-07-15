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

def find_mostfrequent(feature):
    (h, w) = feature.shape
    frequent = -1
    idx = -1
    for c in range(w):
        frq = np.sum(feature[:, c])
        if frequent < frq:
            frequent = frq
            idx = c
    return idx

def find_nodata(data_mat, num_features=9):
    feature_indices = [0, 4, 13, 35, 91, 116, 126, 133, 134]
    for r in range(data.shape[0]):
        sample = data_mat[r, :]
        for i in range(len(feature_indices)-1):
            feature = sample[feature_indices[i]:feature_indices[i+1]]
            if np.sum(feature) == 0: #it means that we have a no data point here
                idx = find_mostfrequent(feature)
                sample[feature_indices[i]:feature_indices[i+1]][idx] = 1
        feature = sample[feature_indices[-1]:]
        if np.sum(feature) == 0: #no-data point
            sample[feature_indices[-1]:][idx] = 1
        data_mat[r, :] = sample
    return data_mat

def create_dataset(path, shape, imputation=False):
    wr_path = '/'.join(path.split('/')[:-1])
    print(wr_path)
    f = h5py.File(wr_path+'/sea2sky.h5', 'a')
    (h, w) = shape
    idx = 0
    data_mat = np.zeros(shape)
    with open(path, 'r') as g:
        gr = csv.reader(g)
        for row in gr:
            assert len(row) == 137+1
            n_row = np.array([float(e) for e in row])
            # if n_row[133] == 0.: #ignore 0 precipitation
            #     continue
            data_mat[idx, :] = n_row
            idx += 1
    # print(idx, h) # idx is the new h
    # filtered_data_mat = data_mat[0:idx, :]
    if imputation:
        data_mat = find_nodata(data_mat)
    np.random.shuffle(data_mat)
    f.create_dataset('train/data', (h-h//5, w-2))
    f.create_dataset('train/gt', (h-h//5, 1))
    f.create_dataset('train/id', (h-h//5, 1))
    f.create_dataset('test/data', (h//5, w-2))
    f.create_dataset('test/gt', (h//5, 1))
    f.create_dataset('test/id', (h//5, 1))
    f['test']['data'][:, :] = data_mat[0:h//5, :-2]
    f['test']['gt'][:, :] = data_mat[0:h//5, -1].reshape(h//5, 1)
    f['test']['id'][:, :] = data_mat[0:h//5, -2].reshape(h//5, 1)
    f['train']['data'][:, :] = data_mat[h//5:, :-2]
    f['train']['gt'][:, :] = data_mat[h//5:, -1].reshape(h-h//5, 1)
    f['train']['id'][:, :] = data_mat[h//5:, -2].reshape(h-h//5, 1)
    # import ipdb; ipdb.set_trace()
    f = normalize(f, 133)
    f.close()

def create_csv_table(path):
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
    fd = open(path+'data_dict.json')
    data_dict = json.load(fd)

    with open('data_table.csv', 'w') as f:
        fw = csv.writer(f)
        with open(path+'all_attributes.csv', 'r') as g:
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
                init_features.extend([row[0]]) # the target id
                init_features.extend([float(row[-1])/100]) # the probability of the target (label)
                fw.writerow(init_features)
    fd.close()

def find_target(target_id, path):
    with open(path+'s2s_target_scores_top10000.csv', 'r') as f:
        fr = csv.reader(f)
        for row in fr:
            if row[0] == target_id:
                return row[4]
        return -1
def merge_all_csv(path):
    with open('all_attributes.csv', 'w') as f:
        fw = csv.writer(f)
        with open(path+'s2s_target_attributes_all.csv', 'r') as g:
            gr = csv.reader(g)
            for row in gr:
                if row[0] == 'target_id':
                    w_row = []
                    w_row.extend(row)
                    w_row.extend('score')
                    fw.writerow(w_row)
                else:
                    if int(row[38]) == 0: # AvgPrecip is zero, ignore
                        print(row[0])
                        continue
                    target = row[0]
                    found = find_target(target, path)
                    if found == -1:
                        continue
                    else:
                        w_row = []
                        w_row.extend(row)
                        w_row.extend([found])                    
                        fw.writerow(w_row)