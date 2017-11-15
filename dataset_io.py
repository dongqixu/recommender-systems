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


# add the usage of numpy
# UserID::MovieID::Rating::Timestamp
def extract_rating(user_num, movie_num, file='dataset/ratings.dat'):
    rating = list()
    user_rate = [0] * user_num
    movie_rate = [0] * movie_num
    with open(file, 'r', encoding='ISO-8859-1') as rating_file:
        rating_line = read_file(rating_file)
        for line in rating_line:
            user_id, movie_id, user_movie_rating, _ = line
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            user_rate[user_id] += 1
            movie_rate[movie_id] += 1
            rating.append((int(user_id), int(movie_id), int(user_movie_rating)))  # remove int(timestamp)
    rating = sorted(rating, key=itemgetter(0, 1))
    # pytorch
    if torch.cuda.is_available():
        rating = torch.cuda.IntTensor(rating)
        user_rate = torch.cuda.FloatTensor(user_rate)
        movie_rate = torch.cuda.FloatTensor(movie_rate)
    else:
        rating = torch.IntTensor(rating)
        user_rate = torch.FloatTensor(user_rate)
        movie_rate = torch.FloatTensor(movie_rate)
    return rating, user_rate, movie_rate


def get_record_index(user_num, movie_num, file='dataset/ratings.dat'):
    user_index = [[] for _ in range(user_num)]
    movie_index = [[] for _ in range(movie_num)]
    with open(file, 'r', encoding='ISO-8859-1') as rating_file:
        rating_line = read_file(rating_file)
        for line in rating_line:
            user_id, movie_id, _, _ = line
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            user_index[user_id].append(movie_id)
            movie_index[movie_id].append(user_id)

    # pytorch
    if torch.cuda.is_available():
        for u in range(user_num):
            user_index[u].sort()
            user_index[u] = torch.cuda.IntTensor(user_index[u])
        for i in range(movie_num):
            movie_index[i].sort()
            movie_index[i] = torch.cuda.IntTensor(movie_index[i])
    else:
        for u in range(user_num):
            user_index[u].sort()
            user_index[u] = torch.IntTensor(user_index[u])
        for i in range(movie_num):
            movie_index[i].sort()
            movie_index[i] = torch.IntTensor(movie_index[i])
    return user_index, movie_index


if __name__ == '__main__':
    start_time = time.time()

    movie_list = extract_movie()
    user_list = extract_user()
    rating_list, user_rated_count, movie_rated_count = extract_rating(6040, 3952)
    print(f'Statistics:\n'
          f' {len(movie_list)} movies\n'
          f' {len(user_list)} users')
    print('Report:\n'
          ' UserIDs range between 1 and 6040\n'
          ' MovieIDs range between 1 and 3952')
    print('Rating format:\n'
          ' UserID::MovieID::Rating::Timestamp')
    print(f'Shape and Sample:\n'
          f' {rating_list.shape}\n'
          f' {rating_list[0:5]}')

    end_time = time.time()
    print(f'\nTime: {end_time - start_time}')

    user_index_list, movie_index_list = get_record_index(6040, 3952)
    print(user_index_list[0][0:10])
    print(f'\nTime: {time.time() - end_time}')
    count = 0
    for u in range(len(user_index_list)):
        for i in range(len(user_index_list[u])):
            if rating_list[count][0] != u or rating_list[count][1] != user_index_list[u][i]:
                print(u, i)
                exit(1)
            count += 1
