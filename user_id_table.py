import pickle


def extract_user_id(dataset='training_dataset_original.txt'):
    user_id_set = set()
    count = 0
    with open(dataset, 'r') as f:
        for line in f:
            user_id = line.strip().split(' ')[1]
            user_id_set.add(int(user_id))
            count += 1
            if count % 10000000 == 0:
                print(f'Reading {count} lines...')
    print(f'Finish reading {count} lines.')
    return user_id_set


if __name__ == '__main__':
    user_list = list(extract_user_id())
    user_list.sort()
    print('Number of users: ', len(user_list))

    user_dict = dict()
    for i, e in enumerate(user_list):
        user_dict[e] = i+1

    with open('user_table.pickle', 'wb') as file_write:
        pickle.dump(user_dict, file_write, protocol=pickle.HIGHEST_PROTOCOL)
    with open('user_list.pickle', 'wb') as file_write:
        pickle.dump(user_list, file_write, protocol=pickle.HIGHEST_PROTOCOL)

    with open('user_table.pickle', 'rb') as file_read:
        user_table = pickle.load(file_read)
    with open('user_list.pickle', 'rb') as file_read:
        user_back_table = pickle.load(file_read)

    print(user_table == user_dict)
    print(user_back_table == user_list)
