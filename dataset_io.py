import time
import torch

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
def extract_rating(file='dataset/ratings.dat'):
    rating = list()
    with open(file, 'r', encoding='ISO-8859-1') as rating_file:
        rating_line = read_file(rating_file)
        for line in rating_line:
            user_id, movie_id, user_movie_rating, timestamp = line
            rating.append((int(user_id), int(movie_id), int(user_movie_rating), int(timestamp)))
    # pytorch
    if torch.cuda.is_available():
        rating = torch.cuda.IntTensor(rating)
    else:
        rating = torch.IntTensor(rating)  # 32-bit integer (signed)
    return rating


if __name__ == '__main__':
    start_time = time.time()

    movie_list = extract_movie()
    user_list = extract_user()
    rating_list = extract_rating()
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
          f' {rating_list[-1]}')

    end_time = time.time()
    print(f'\nTime: {end_time - start_time}')
