import pickle


def revise_user_id(user_table, revised, dataset='training_dataset_original.txt'):
    count = 0
    with open(revised, 'w') as file_write:
        with open(dataset, 'r') as file_read:
            for line in file_read:
                line = line.strip().split(' ')
                movie_id, user_id, rating, _ = line
                user_id = user_table[int(user_id)]
                file_write.write(f'{user_id} {movie_id} {rating}\n')
                count += 1
                if count % 10000000 == 0:
                    print(f'Reading {count} lines...')
        print(f'Finish reading {count} lines.')


if __name__ == '__main__':
    with open('user_table.pickle', 'rb') as table:
        user_table = pickle.load(table)
    revise_user_id(user_table, revised='dataset_no_sorting.txt')