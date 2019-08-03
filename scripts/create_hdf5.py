import os
import tarfile
import numpy as np
import h5py
from tqdm import tqdm



def extract_from_tar(tar_file):
    tar = tarfile.open(tar_file)
    
    fmap = {'Training': 'train', 'Validation': 'val', 'Testing': 'test'}
    tmap = {'Images': 'x', 'Categories': 'y'}
    keys = {'x_train', 'y_train', 'x_val', 'y_val', 'x_test'}
    ext_data = dict((k, {}) for k in keys)
    
    for tarinfo in tqdm(tar.getmembers()): 
        path = tarinfo.name # file path
        splits = os.path.normpath(path).split(os.path.sep)
    
        if (len(splits) >= 4 and splits[1] in fmap and splits[-2] in tmap):
            
            content = tar.extractfile(tarinfo).read()
            key =  tmap[splits[-2]] + '_' + fmap[splits[1]]
            basename = os.path.splitext(splits[-1])[0] # without extension
        
            ext_data[key][basename] = content
    return ext_data


def binary2uint(data):
    return np.frombuffer(data, dtype='uint8')


def create_hdf5(tar_file, out_name):
    data_dict = extract_from_tar(tar_file)
    
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    with h5py.File(out_name, 'w', libver='latest', swmr=True) as out:
        out.create_dataset('x_train', (len(data_dict['x_train']),), dtype=dt)
        out.create_dataset('y_train', (len(data_dict['y_train']),), dtype=dt)   
        out.create_dataset('x_val', (len(data_dict['x_val']),), dtype=dt)
        out.create_dataset('y_val', (len(data_dict['y_val']),), dtype=dt)     
        out.create_dataset('x_test', (len(data_dict['x_test']),), dtype=dt)
    
    # Write data
    with h5py.File(out_name, 'a', libver='latest', swmr=True) as dset:
        for i, (k, v) in enumerate(tqdm(data_dict['x_train'].items(),
                                   'Processing train set..')):
            dset['x_train'][i] = binary2uint(v)
            dset['y_train'][i] = binary2uint(data_dict['y_train'][k])

        for i, (k, v) in enumerate(tqdm(data_dict['x_val'].items(),
                                   'Processing val set..')):
            dset['x_val'][i] = binary2uint(v)
            dset['y_val'][i] = binary2uint(data_dict['y_val'][k])

        for i, (k, v) in enumerate(tqdm(data_dict['x_test'].items(),
                                   'Processing test set..')):
            dset['x_test'][i] = binary2uint(v)