import h5py
import numpy as np
import os
import torch
from dataset_io import read_file


def store_rating_list(group, step, user_num=480189, movie_num=17770):
    if not os.path.isdir(f'{group}_group'):
        os.mkdir(f'{group}_group')
    batch = 0
    num = 0
    count = None
    rating = list()
    user_rate_count = [0] * user_num
    movie_rate_count = [0] * movie_num
    with open(f'../nf_txt/{group}_group.txt') as rating_file:
        rating_line = read_file(rating_file, separator=' ')
        for line in rating_line:
            user_id, movie_id, user_movie_rating = line
            # index start from zero
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            # check before adding
            if group == 'user':
                count = user_id
            elif group == 'movie':
                count = movie_id
            # perform storing and new start -> zero always
            num += 1
            if count % step == 0 and count == (batch+1)*step:
                print(f'{num} Writing batch {batch}...')
                num = 0
                with h5py.File(f'{group}_group/{group}_{batch}.hdf5', 'w') as file:
                    file.create_dataset('numpy_array', data=np.array(rating, dtype=int))
                batch += 1
                rating = list()
            # update count
            user_rate_count[user_id] += 1
            movie_rate_count[movie_id] += 1
            rating.append((int(user_id), int(movie_id), int(user_movie_rating)))
    # final set
    print(f'Writing batch {batch} as the final batch.')
    with h5py.File(f'{group}_group/{group}_{batch}.hdf5', 'w') as file:
        file.create_dataset('numpy_array', data=np.array(rating, dtype=int))
    with h5py.File(f'rate_count.hdf5', 'w') as file:
        file.create_dataset('user_rate_count_numpy', data=np.array(user_rate_count, dtype=int))
        file.create_dataset('movie_rate_count_numpy', data=np.array(movie_rate_count, dtype=int))


def load_rating_list(batch, group):
    with h5py.File(f'{group}_group/{group}_{batch}.hdf5', 'r') as file:
        rating_batch = file['numpy_array'][:]
    # pytorch
    rating_batch = torch.from_numpy(rating_batch)
    if torch.cuda.is_available():
        rating_batch = rating_batch.cuda()
    return rating_batch


def load_rate_count_numpy():
    with h5py.File(f'rate_count.hdf5', 'r') as file:
        user_rate_count_numpy = file['user_rate_count_numpy'][:]
        movie_rate_count_numpy = file['movie_rate_count_numpy'][:]
    return user_rate_count_numpy, movie_rate_count_numpy


def write_h5py(file_name='test.hdf5'):
    data = np.random.random(size=(100, 20))
    print(data)
    print('----------')
    with h5py.File(file_name, 'w') as file:
        file.create_dataset('numpy_array', data=data)


def read_h5py(file_name='test.hdf5'):
    with h5py.File('rate_count.hdf5', 'r') as file:
        user_rate_count = file['user_rate_count_numpy'][:]
    with h5py.File('_rate_count.hdf5', 'r') as file:
        _user_rate_count = file['user_rate_count_numpy'][:]
    print(np.array_equal(user_rate_count, _user_rate_count))


if __name__ == '__main__':
    # write_h5py()
    # read_h5py()
    print('movie')
    store_rating_list('movie', 1000)
    print('user')
    store_rating_list('user', 30000)
    # read_h5py()