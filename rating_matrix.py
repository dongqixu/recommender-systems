import numpy as np
import time
from dataset_io import extract_rating


class RatingMatrix(object):
    # report number different from statistics
    def __init__(self, feature_num, lambda_p, lambda_q):
        # dataset
        self.movie_num = 3952
        self.user_num = 6040
        self.rating_list = extract_rating()  # numpy array

        # parameter
        self.feature_num = feature_num
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q

        # matrix initialization
        self.user_matrix = np.random.random((self.user_num, self.feature_num)).astype(np.float64)
        self.movie_matrix = np.random.random((self.feature_num, self.movie_num)).astype(np.float64)
        # TODO
        # self.user_rated_count = np.zeros(self.user_num, dtype=np.float64)
        self.movie_rated_count = np.zeros(self.movie_num, dtype=np.float64)

        # transpose
        self.transpose_rating = np.transpose(np.copy(self.rating_list))  # problem with memory sharing?
        self.train_user_id, self.train_movie_id, self.train_rating = self.transpose_rating[0:3, :]
        self.train_user_id -= 1
        self.train_movie_id -= 1
        del self.transpose_rating  # memory

        # TODO: problem with the incorrect statistics
        user_rated_list, self.user_rated_count = np.unique(self.train_user_id, return_counts=True)
        for test_i in range(len(user_rated_list)):
            if user_rated_list[test_i] != test_i:
                print(test_i)
        # print('Test on User Rated List Passed.')
        # TODO: Check whether it is right
        movie_rated_list, movie_rated_count = np.unique(self.train_movie_id, return_counts=True)
        movie_rated_dict = dict(zip(movie_rated_list, movie_rated_count))
        for test_i in range(self.movie_num):
            count = movie_rated_dict.get(test_i)
            if count is not None:
                self.movie_rated_count[test_i] = count
        # print('Test on Movie Rated List Passed.')
        # print(np.sum(self.user_rated_count), np.sum(self.movie_rated_count), len(self.train_rating))

        self.user_rated_count = self.user_rated_count * self.lambda_p
        self.movie_rated_count = self.movie_rated_count * self.lambda_q
        # print(np.sum(self.user_rated_count), np.sum(self.movie_rated_count), len(self.train_rating))

        print(f'Features: {self.feature_num}\n'
              f'Lambda P: {self.lambda_p}\n'
              f'Lambda Q: {self.lambda_q}\n')

    def get_loss_numpy(self):
        # u == i
        # self.user_matrix[self.train_user_id, :] -> (u, k)
        # self.movie_matrix[:, self.train_movie_id]) -> (k, i)
        intermediate_rating = self.user_matrix[self.train_user_id, :] * np.transpose(
            self.movie_matrix[:, self.train_movie_id])
        predict_rating = np.sum(intermediate_rating, axis=1)
        del intermediate_rating  # memory
        predict_rating = self.train_rating - predict_rating
        euclidean_distance_loss = np.sum(predict_rating * predict_rating)
        return euclidean_distance_loss

    def update_numpy(self):
        user_up = np.zeros((self.user_num, self.feature_num), dtype=np.float64)
        user_down = np.zeros((self.user_num, self.feature_num), dtype=np.float64)
        item_up = np.zeros((self.feature_num, self.movie_num), dtype=np.float64)
        item_down = np.zeros((self.feature_num, self.movie_num), dtype=np.float64)

        # prediction from previous section -> (1000209,)
        intermediate_rating = self.user_matrix[self.train_user_id, :] * np.transpose(
            self.movie_matrix[:, self.train_movie_id])
        predict_rating = np.sum(intermediate_rating, axis=1)
        del intermediate_rating  # memory

        # TODO: (k, i) * (i,) why?
        # user_up
        user_up_add = np.transpose(self.movie_matrix[:, self.train_movie_id] * self.train_rating[:])
        np.add.at(user_up, self.train_user_id, user_up_add)
        del user_up_add

        # user_down
        user_down_add = np.transpose(self.movie_matrix[:, self.train_movie_id] * predict_rating[:])
        np.add.at(user_down, self.train_user_id, user_down_add)
        del user_down_add

        # item_up
        # TODO: (u, k) -> (k, u) * (u,)
        item_up_add = np.transpose(self.user_matrix[self.train_user_id, :]) * self.train_rating[:]
        item_up_add_transpose = np.transpose(item_up_add)
        item_up_transpose = np.transpose(item_up)
        np.add.at(item_up_transpose, self.train_movie_id, item_up_add_transpose)
        del item_up_add

        # item_down
        item_down_add = np.transpose(self.user_matrix[self.train_user_id, :]) * predict_rating[:]
        item_down_add_transpose = np.transpose(item_down_add)
        item_down_transpose = np.transpose(item_down)
        np.add.at(item_down_transpose, self.train_movie_id, item_down_add_transpose)
        del item_down_add

        # each user entry
        '''lambda_p: constant, I_u: (user_num,), p_uk: (user_num, k)'''
        '''create a I_u member variable, put and give shifting'''
        # print('Shape', self.user_matrix.shape, self.user_rated_count.shape)
        combine = np.transpose(np.transpose(self.user_matrix) * self.user_rated_count)
        user_down += 1e-5
        user_down += combine
        self.user_matrix *= (user_up / user_down)
        del combine

        # each movie entry
        # print('Shape', self.movie_matrix.shape, self.movie_rated_count.shape)
        combine = self.movie_matrix * self.movie_rated_count
        item_down += 1e-5
        item_down += combine
        self.movie_matrix *= (item_up / item_down)
        del combine


if __name__ == '__main__':
    # np.random.seed(0)

    # file operation
    line_buffer = 1
    log = open('loss.txt', 'w', buffering=line_buffer)

    # parameter setting
    lambda_pq_list = [0.02*(x+1) for x in range(10)]
    feature_list = [200*(x+1) for x in range(8)]
    train_epoch = 200  # 100?

    for feature_num in feature_list:
        for lambda_pq in lambda_pq_list:
            R = RatingMatrix(feature_num=feature_num, lambda_p=lambda_pq, lambda_q=lambda_pq)

            print(f'Parameters: ({lambda_pq} {feature_num})\n'
                  f'Initial Loss: '
                  f'{R.get_loss_numpy():{12}.{8}}')
            log.write(f'Parameters: {feature_num} {lambda_pq}\n'
                      f'0 {R.get_loss_numpy():{12}.{8}}\n')
            for epoch in range(train_epoch):
                start_time = time.time()

                # R.update_numpy()

                loss = R.get_loss_numpy()
                log_string = f'[Epoch] {epoch+1}, time: {time.time() - start_time:{5}.{4}}, loss {loss:{12}.{8}}'
                print(log_string)
                log.write(f'{epoch+1} {loss}\n')
            print('------------------------------------------------------------------------------------')
            log.write('------------------------------------------------------------------------------------\n')

    # file operation
    log.close()
