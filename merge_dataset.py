import os
import time


# read all the file under the dir and return like the list
def read_dir(dir_path):
    all_files = []
    if os.path.isdir(dir_path):
        file_list = os.listdir(dir_path)
        # TODO: .DS_Store
        for f in file_list:
            f = dir_path + '/' + f
            if os.path.isdir(f):
                sub_files = read_dir(f)
                all_files = sub_files + all_files
            else:
                all_files.append(f)
        return all_files
    else:
        print('Error, not a dir')
        return None


# read all the file data
def read_file(file_write, file_path, movie_id):
    with open(file_path, 'r') as f:
        next(f)  # skip the first line -> that is movie_id
        lines = f.readlines()
        for line in lines:
            file_write.write(str(movie_id) + ',' + str(line))
            # f.flush()  # flush will be slow


def read_file_check_movie_id(file_write, file_path, movie_id):
    with open(file_path, 'r') as f:
        movie_id_checked = False
        for line in f:
            if not movie_id_checked:
                line = line.split(':')[0]
                if movie_id == int(line):
                    movie_id_checked = True
                else:
                    print('Movie ID mismatch')
                    exit(1)
            else:
                file_write.write(str(movie_id) + ',' + str(line))


def merge_dataset(dir_path, merge_file):
    start_time = time.time()
    file_list = read_dir(dir_path)
    file_list.sort()
    print(file_list)

    count = 0
    for file_path in file_list:
        file_name = file_path.strip().split('/')[-1]
        movie_id = int(file_name.split('.')[0].split('_')[1])
        read_file(merge_file, file_path, movie_id)

        count += 1
        if count % 1000 == 0:
            print(f'Reading {count} movies...')
    print(f'Total {count} movies read.')

    end_time = time.time()
    print(f'\nTime: {end_time - start_time}')


if __name__ == '__main__':
    dir_path = '../nf_prize_dataset/training_set'
    with open('training_dataset.txt', 'w') as merge_file:
        merge_dataset(dir_path, merge_file)
