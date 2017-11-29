import codecs
import h5py
import numpy as np


def recommend_list(user_id, pq_matrix='PQ_matrix.hdf5'):
    with h5py.File(pq_matrix, 'r') as file:
        user_matrix = file['user_matrix'][:]
        movie_matrix = file['movie_matrix'][:]
    print('Success loading data')

    user_feature = user_matrix[user_id]
    movie_feature = np.transpose(movie_matrix)

    print('Shape of user', user_feature.shape)
    print('Shape of movie feature', movie_feature.shape)
    prediction = np.dot(movie_feature, user_feature)

    print('Prediction', prediction)
    print('Max and min', max(prediction), min(prediction))

    index_highest = np.flip(np.argsort(prediction), 0)[0:20]
    print('Index with 20 highest rating', index_highest)
    print('Rating of 20 highest movies:', prediction[index_highest])
    return index_highest


def get_movie_title(index_tuple):
    title_list = list()
    for index in index_tuple:
        # index start from 1
        index += 1
        with codecs.open('movie_titles_utf8.txt', 'r', 'utf-8') as movie_file:
            for line in movie_file:
                line = line.strip().split(',')
                if int(line[0]) == index:
                    title_list.append(line[2])
                    break
            else:
                print('Error loading')
    return title_list


if __name__ == '__main__':
    user_id = np.random.randint(480189)
    print('User:', user_id)
    movie_list = recommend_list(user_id)
    name = get_movie_title(movie_list)
    print('Recommend movies:', name)
