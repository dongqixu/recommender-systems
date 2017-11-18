import h5py
import numpy as np


def write_h5py(file_name='test.hdf5'):
    data = np.random.random(size=(100, 20))
    print(data)
    print('----------')
    with h5py.File(file_name, 'w') as file:
        file.create_dataset('numpy_array', data=data)


def read_h5py(file_name='test.hdf5'):
    with h5py.File(file_name, 'r') as file:
        data = file['numpy_array'][:]
        print(data)
        print(data.shape)


if __name__ == '__main__':
    write_h5py()
    read_h5py()
