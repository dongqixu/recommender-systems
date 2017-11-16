import numpy as np
import sys
import time
import torch
from operator import itemgetter

'''Read dataset with pytorch'''


def read_file(file, separator='::'):
    for line in file:
        yield line.strip().split(separator)


# Format: MovieID::Title::Genres
def extract_movie(file='dataset/movies.dat'):
    movie = list()
    with open(file, 'r', encoding='ISO-8859-1') as movie_file:
        movie_line = read_file(movie_file)
        for line in movie_line:
            movie_id, title, genres = line
            movie.append((int(movie_id), title, genres))
    return movie


# UserID::Gender::Age::Occupation::Zip-code
def extract_user(file='dataset/users.dat'):
    user = list()
    with open(file, 'r', encoding='ISO-8859-1') as user_file:
        user_line = read_file(user_file)
        for line in user_line:
            user_id, gender, age, occupation, zip_code = line
            user.append((int(user_id), gender, int(age), occupation, zip_code))
    return user


# UserID::MovieID::Rating::Timestamp
def extract_rating_with_count(user_num, movie_num, file='dataset/ratings.dat', user_movie_order=True):
    rating = list()
    user_rate_count = [0] * user_num
    movie_rate_count = [0] * movie_num
    with open(file, 'r', encoding='ISO-8859-1') as rating_file:
        rating_line = read_file(rating_file)
        for line in rating_line:
            user_id, movie_id, user_movie_rating, _ = line
            # index start from zero
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            # update count
            user_rate_count[user_id] += 1
            movie_rate_count[movie_id] += 1
            # append list
            rating.append((int(user_id), int(movie_id), int(user_movie_rating)))  # remove int(timestamp)
    # TODO: sorted order -> user, movie
    if user_movie_order:
        rating = sorted(rating, key=itemgetter(0, 1))
    else:
        rating = sorted(rating, key=itemgetter(1, 0))
    # numpy
    rating = np.array(rating, dtype=int)
    user_rate_count = np.array(user_rate_count, dtype=int)
    movie_rate_count = np.array(movie_rate_count, dtype=int)
    # pytorch
    rating = torch.from_numpy(rating)
    user_rate_count = torch.from_numpy(user_rate_count)
    movie_rate_count = torch.from_numpy(movie_rate_count)
    # GPU
    if torch.cuda.is_available():
        rating = rating.cuda()
        user_rate_count = user_rate_count.cuda()
        movie_rate_count = movie_rate_count.cuda()
    #     print('GPU:\n', rating[0], user_rate_count[0:2], movie_rate_count[0:2])
    # else:
    #     print('CPU:\n', rating[0], user_rate_count[0:2], movie_rate_count[0:2])
    return rating, user_rate_count, movie_rate_count


def get_rating_index(user_num, movie_num, file='dataset/ratings.dat'):
    user_index = [[] for _ in range(user_num)]
    movie_index = [[] for _ in range(movie_num)]
    with open(file, 'r', encoding='ISO-8859-1') as rating_file:
        rating_line = read_file(rating_file)
        for line in rating_line:
            user_id, movie_id, _, _ = line
            # index start from zero
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            # update index list
            user_index[user_id].append(movie_id)
            movie_index[movie_id].append(user_id)

    # numpy
    cuda_enable = torch.cuda.is_available()
    for u in range(user_num):
        user_index[u].sort()
        user_index[u] = np.array(user_index[u], dtype=int)
        if len(user_index[u]) > 0:
            user_index[u] = torch.from_numpy(user_index[u])
            if cuda_enable:
                user_index[u] = user_index[u].cuda()
    for i in range(movie_num):
        movie_index[i].sort()
        movie_index[i] = np.array(movie_index[i], dtype=int)
        if len(movie_index[i]) > 0:
            movie_index[i] = torch.from_numpy(movie_index[i])
            if cuda_enable:
                movie_index[i] = movie_index[i].cuda()
    return user_index, movie_index


if __name__ == '__main__':
    # set default GPU
    if torch.cuda.is_available():
        cuda_device = 0
        if len(sys.argv) > 1:
            cuda_device = int(sys.argv[1])
        torch.cuda.set_device(cuda_device)

    start_time = time.time()
    movie_list = extract_movie()
    user_list = extract_user()
    rating_list, user_rate_count_list, movie_rate_count_list = extract_rating_with_count(6040, 3952)
    rating_list_movie_first, _, _ = extract_rating_with_count(6040, 3952, user_movie_order=False)
    # print(f'Statistics:\n'
    #       f' {len(movie_list)} movies\n'
    #       f' {len(user_list)} users')
    # print('Report:\n'
    #       ' UserIDs range between 1 and 6040\n'
    #       ' MovieIDs range between 1 and 3952')
    # print('Rating format:\n'
    #       ' UserID::MovieID::Rating::Timestamp')
    # print(f'Shape and Sample:\n'
    #       f' {rating_list.shape}\n'
    #       f' {rating_list[0:5]}')
    end_time = time.time()
    print(f'Time to extract rating list: {end_time - start_time}')

    user_index_list, movie_index_list = get_rating_index(6040, 3952)
    print(user_index_list[0][0:2])
    print(f'Time to extract index list: {time.time() - end_time}')

    start_time = time.time()
    count = 0
    for u in range(len(user_index_list)):
        if user_rate_count_list[u] != len(user_index_list[u]):
            print('Exit with rate count mismatch')
            exit(1)
        for i in range(user_rate_count_list[u]):
            if rating_list[count][0] != u or rating_list[count][1] != user_index_list[u][i]:
                print('Exit with rating list mismatch')
                exit(2)
            count += 1
    print('Rating list check passed.')
    end_time = time.time()
    print(f'Time to check rating list: {end_time - start_time}')

    start_time = time.time()
    count = 0
    for i in range(len(movie_index_list)):
        if movie_rate_count_list[i] != len(movie_index_list[i]):
            print('Exit with rate count mismatch')
            exit(3)
        for u in range(movie_rate_count_list[i]):
            if rating_list_movie_first[count][1] != i or \
                            rating_list_movie_first[count][0] != movie_index_list[i][u]:
                print('Exit with rating list mismatch')
                exit(4)
            count += 1
    print('Rating list of movie check passed.')
    end_time = time.time()
    print(f'Time to check rating list of movie: {end_time - start_time}')
