import h5py
import numpy as np
import pdb

"""
Each hdf5 file is a dataset. 
It contains multiple videos and each video has multiple frames.
The feature of each frame is extracted from the penultimate layer (pool 5) of the GoogLeNet model (1024 dims).

Args:
    data_name: OVP, Youtube, TVSum, SumMe
    label_type: 1 - ; 2 - 
Returns:
    feature: an array, each entry (1024 * num of frames) corresponds to a video
    label: an array, each entry (num of frames * 1) corresponds to a video
    weight: 
""" 
def load_h5(data_dir, data_name, model_type):
    feature = []
    label = []  # if label_type = 1, use gt_1 as labels; if label_type = 2, use gt_2 as labels
    weight = [] # how to use it?
    filename = data_dir + 'Data_' + data_name + '_google_p5.h5'
    f = h5py.File(filename)
    # flatten video index into a 1D array, and sort it in ascending order
    video_ord = np.sort(np.array(f['ord']).astype('int32').flatten())   
    for i in video_ord: # for each video
        feature.append(np.matrix(f['fea_' + i.__str__()]).astype('float32'))
        label.append(np.array(f['gt_' + model_type.__str__() + '_' + i.__str__()]).astype('float32'))
        weight.append(np.array(model_type - 1.0).astype('float32'))
    f.close()

    return feature, label, weight

"""
Load both training, validation and testing data.
Args:
    test_data_name: default test data is TVSum
    model_type: 1 - ; 2 - 
"""
def load_data(data_dir, test_data_name = 'TVSum', model_type = 2):
    train_set = [[], [], [], []]
    
    # load OVP dataset (50 videos) for training
    [feature, label, weight] = load_h5(data_dir, 'OVP', model_type)
    # a list, each entry corresponds to a video
    # each entry is a np.array containing indices whose labels are nonzero 
    label_nonzero = [np.where(l)[0].astype('int32') for l in label]
    # feature is an array, each entry (1024 * num of frames) corresponds to a video
    train_set[0].extend(feature)
    # label is an array, each entry (num of frames * 1) corresponds to a video
    train_set[1].extend(label)
    train_set[2].extend(label_nonzero)
    train_set[3].extend(weight)
    
    # load Youtube (39 videos) for training
    [feature, label, weight] = load_h5(data_dir, 'Youtube', model_type)
    label_nonzero = [np.where(l)[0].astype('int32') for l in label]
    train_set[0].extend(feature)
    train_set[1].extend(label)
    train_set[2].extend(label_nonzero)
    train_set[3].extend(weight)

    valid_set = [[], [], [], []]
    valid_idx = []
    test_set = [[], [], [], []]
    test_idx = []

    # load TVSum (50 videos) dataset for testing
    # randomly select 16 SumMe videos for training, and other 9 videos for validation
    if test_data_name == 'TVSum':
        # load TVSum
        [feature, label, weight] = load_h5(data_dir, 'TVSum', model_type)
        label_nonzero = [np.where(l)[0].astype('int32') for l in label]
        test_set[0].extend(feature)
        test_set[1].extend(label)
        test_set[2].extend(label_nonzero)
        test_set[3].extend(weight)
        test_idx.extend(range(50))
        # load SumMe
        [feature, label, weight] = load_h5(data_dir, 'SumMe', model_type)
        label_nonzero = [np.where(l)[0].astype('int32') for l in label]
        rand_idx = np.random.permutation(25)
        for i in range(25):
            if i <= 15:     # 16 videos for training
                train_set[0].append(feature[rand_idx[i]])
                train_set[1].append(label[rand_idx[i]])
                train_set[2].append(label_nonzero[rand_idx[i]])
                train_set[3].append(weight[rand_idx[i]])
            else:   # 9 videos for validation
                valid_set[0].append(feature[rand_idx[i]])
                valid_set[1].append(label[rand_idx[i]])
                valid_set[2].append(label_nonzero[rand_idx[i]])
                valid_set[3].append(weight[rand_idx[i]])
                valid_idx.append(rand_idx[i])

    # load SumMe (25 videos) dataset for testing
    # randomly select 31 TVSum videos for training, and other 19 videos for validation
    elif test_data_name == 'SumMe':
        # load SumMe
        [feature, label, weight] = load_h5(data_dir, 'SumMe', model_type)
        label_nonzero = [np.where(l)[0].astype('int32') for l in label]
        test_set[0].extend(feature)
        test_set[1].extend(label)
        test_set[2].extend(label_nonzero)
        test_set[3].extend(weight)
        test_idx.extend(range(25))
        # load TVSum
        [feature, label, weight] = load_h5(data_dir, 'TVSum', model_type)
        label_nonzero = [np.where(l)[0].astype('int32') for l in label]
        rand_idx = np.random.permutation(50)
        for i in range(50):
            if i <= 30:     # 31 videos for training
                train_set[0].append(feature[rand_idx[i]])
                train_set[1].append(label[rand_idx[i]])
                train_set[2].append(label_nonzero[rand_idx[i]])
                train_set[3].append(weight[rand_idx[i]])
            else:   # 19 videos fro validation
                valid_set[0].append(feature[rand_idx[i]])
                valid_set[1].append(label[rand_idx[i]])
                valid_set[2].append(label_nonzero[rand_idx[i]])
                valid_set[3].append(weight[rand_idx[i]])
                valid_idx.append(rand_idx[i])

    # No such dataset for testing
    else:
        raise('No such dataset for testing!')

    """
    Notes:
    TVSum for testing: train_set[0] is an array of 105 entries, each entry corresponds to a video
    SumMe for testing: train_set[0] is an array of 120 entries, each entry corresponds to a video
    It is the same with train_set[1], train_set[2]
    """

    for i in range(len(train_set[0])):  # for each video
        train_set[0][i] = np.transpose(train_set[0][i]) # each row is a frame feature
        train_set[1][i] = train_set[1][i].flatten().astype('float32')   # 1D column vector
        train_set[2][i] = train_set[2][i].flatten().astype('int32') # 1D column vector
        train_set[3][i] = train_set[3][i]

    for i in range(len(valid_set[0])):
        valid_set[0][i] = np.transpose(valid_set[0][i])
        valid_set[1][i] = valid_set[1][i].flatten().astype('float32')
        valid_set[2][i] = valid_set[2][i].flatten().astype('int32')
        valid_set[3][i] = valid_set[3][i]

    for i in range(len(test_set[0])):
        test_set[0][i] = np.transpose(test_set[0][i])
        test_set[1][i] = test_set[1][i].flatten().astype('float32')
        test_set[2][i] = test_set[2][i].flatten().astype('int32')
        test_set[3][i] = test_set[3][i]
    
    pdb.set_trace()
    
    return train_set, valid_set, test_set, test_idx
