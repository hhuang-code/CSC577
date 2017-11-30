from tools.data_loader import load_data
import numpy as np
import pdb

if __name__ == '__main__':

    # load data
    data_dir = '../data/'
    test_data_name = 'SumMe'
    model_type = 2  # 1 for vsLSTM, 2 for dppLSTM
    print('...loading data')
    train_set, valid_set, test_set, test_idx = load_data(data_dir, test_data_name, model_type)

    items = []
    for i in range(len(train_set[0])):
        items.append([train_set[j][i] for j in range(len(train_set))])

    pdb.set_trace()
    print(len(items))    
