import os
import time


# read all files under the dir
def list_dir(dir_path):
    all_files = []
    if os.path.isdir(dir_path):
        file_list = os.listdir(dir_path)
        # TODO: .DS_Store
        for f in file_list:
            f = dir_path + '/' + f
            if os.path.isdir(f):
                sub_files = list_dir(f)
                all_files += sub_files
            else:
                all_files.append(f)
        all_files.sort()
        return all_files
    else:
        print('Error, not a dir')
        return None


# read all the file data
def merge_train_file(file_write, file_read, movie_id):
    with open(file_read, 'r') as f:
        next(f)  # skip the first line -> that is movie_id
        lines = f.readlines()
        for line in lines:
            line = line.replace(',', ' ')
            file_write.write(str(movie_id) + ' ' + str(line))
            # f.flush()  # flush will be slow


def check_movie_id(file_read, movie_id):
    with open(file_read, 'r') as f:
        for line in f:
            line = line.split(':')[0]
            if movie_id == int(line):
                break
            else:
                print('Movie ID mismatch')
                exit(404)


def merge_dataset(dir_path, merge_file):
    start_time = time.time()
    file_list = list_dir(dir_path)
    # print(file_list)

    count = 0
    for file_read in file_list:
        file_name = file_read.strip().split('/')[-1]
        movie_id = int(file_name.split('.')[0].split('_')[1])
        check_movie_id(file_read, movie_id)
        merge_train_file(merge_file, file_read, movie_id)

        count += 1
        if count % 1000 == 0:
            print(f'Reading {count} movies...')
    print(f'Total {count} movies read.')

    end_time = time.time()
    print(f'\nTime: {end_time - start_time}')


if __name__ == '__main__':
    dir_path = '../nf_prize_dataset/training_set'
    with open('training_dataset_original.txt', 'w') as merge_file:
        merge_dataset(dir_path, merge_file)
