import numpy as np
import time
from numpy_implement.dataset_io import extract_rating


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
        self.mask = None
        self.ground_truth = None
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

        '''
        uu = np.zeros(self.user_num, dtype=np.int)
        for ii in range(len(self.train_user_id)):
            uu[self.train_user_id[ii]] += 1
        print('uu', np.array_equal(uu, self.user_rated_count))

        mm = np.zeros(self.movie_num, dtype=np.int)
        for ii in range(len(self.train_movie_id)):
            mm[self.train_movie_id[ii]] += 1
        print('mm', np.array_equal(mm, self.movie_rated_count))
        '''

        self.user_rated_count = self.user_rated_count * self.lambda_p
        self.movie_rated_count = self.movie_rated_count * self.lambda_q
        # print(np.sum(self.user_rated_count), np.sum(self.movie_rated_count), len(self.train_rating))

        # self.fill_ground_truth_numpy()

        # # deprecated: test component
        # self.fill_ground_truth()
        # temp_ground_truth = np.copy(self.ground_truth)
        # temp_mask = np.copy(self.mask)
        # self.ground_truth[:, :] = 0
        # self.mask[:, :] = 0
        # self.fill_ground_truth_numpy()
        # print(np.array_equal(temp_ground_truth, self.ground_truth))
        # print(np.array_equal(temp_mask, self.mask))

        # print(f'Matrix R: {self.ground_truth.shape}\n'
        #       f'Matrix P: {self.user_matrix.shape}\n'
        #       f'Matrix Q: {self.movie_matrix.shape}')
        print(f'Features: {self.feature_num}\n'
              f'Lambda P: {self.lambda_p}\n'
              f'Lambda Q: {self.lambda_q}\n')

    # deprecated
    def fill_ground_truth_loop(self):
        # only initialize with calling
        self.mask = np.zeros((self.user_num, self.movie_num), dtype=np.int)
        self.ground_truth = np.zeros((self.user_num, self.movie_num), dtype=np.int)

        # index start from zero
        for _, temp_rating in enumerate(self.rating_list):
            user_id, movie_id, user_movie_rating, _ = temp_rating
            user_id -= 1
            movie_id -= 1
            self.ground_truth[user_id, movie_id] = user_movie_rating
            self.mask[user_id, movie_id] = 1

    def fill_ground_truth_numpy(self):
        # only initialize with calling
        self.mask = np.zeros((self.user_num, self.movie_num), dtype=np.int)
        self.ground_truth = np.zeros((self.user_num, self.movie_num), dtype=np.int)

        self.ground_truth[self.train_user_id, self.train_movie_id] = self.train_rating
        self.mask[self.train_user_id, self.train_movie_id] = 1

    # deprecated
    # how to increase the speed: too many entry of (u, i)
    def get_loss_loop(self):
        euclidean_distance_loss = 0
        # index start from zero
        for index, temp_rating in enumerate(self.rating_list):
            # if (index + 1) % 50000 == 0:
            #     print(index+1)
            user_id, movie_id, _, _ = temp_rating
            u = user_id - 1
            i = movie_id - 1
            predict_rating_ui = 0
            for k in range(self.feature_num):
                predict_rating_ui += self.user_matrix[u][k] * self.movie_matrix[k][i]
            error_ui = self.ground_truth[u][i] - predict_rating_ui
            euclidean_distance_loss += error_ui * error_ui
        return euclidean_distance_loss

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

    # deprecated
    def get_loss_matrix(self):
        predict_r = np.dot(self.user_matrix, self.movie_matrix)
        error_matrix = self.ground_truth - predict_r
        error_matrix = error_matrix * error_matrix * self.mask
        euclidean_distance_loss = np.sum(error_matrix)
        return euclidean_distance_loss

    # deprecated
    def update_loop(self):
        user_up = np.zeros((self.user_num, self.feature_num), dtype=np.float64)
        user_down = np.zeros((self.user_num, self.feature_num), dtype=np.float64)
        item_up = np.zeros((self.feature_num, self.movie_num), dtype=np.float64)
        item_down = np.zeros((self.feature_num, self.movie_num), dtype=np.float64)

        # for each entry of training set
        for index, temp_rating in enumerate(self.rating_list):
            # too slow for single operation
            # if (index + 1) % 50000 == 0:
            #     print(index + 1)
            user_id, movie_id, _, _ = temp_rating
            u = user_id - 1
            i = movie_id - 1
            predict_rating_ui = 0
            for k in range(self.feature_num):
                predict_rating_ui += self.user_matrix[u][k] * self.movie_matrix[k][i]
            for k in range(self.feature_num):
                user_up[u][k] += self.movie_matrix[k][i] * self.ground_truth[u][i]
                user_down[u][k] += self.movie_matrix[k][i] * predict_rating_ui
                item_up[k][i] += self.user_matrix[u][k] * self.ground_truth[u][i]
                item_down[k][i] += self.user_matrix[u][k] * predict_rating_ui

        # each user entry
        for u in range(self.user_num):
            for k in range(self.feature_num):
                user_down[u][k] += 1e-5 + self.user_rated_count[u] * self.user_matrix[u][k]
                # update
                self.user_matrix[u][k] *= (user_up[u][k] / user_down[u][k])

        # each movie entry
        for i in range(self.movie_num):
            for k in range(self.feature_num):
                item_down[k][i] += 1e-5 + self.movie_rated_count[i] * self.movie_matrix[k][i]
                # update
                self.movie_matrix[k][i] *= (item_up[k][i] / item_down[k][i])

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

        # deprecated
        '''
        # for each entry of training set
        for index, temp_rating in enumerate(predict_rating):
            u = self.train_user_id[index]
            i = self.train_movie_id[index]
            user_up[u, :] += self.movie_matrix[:, i] * self.train_rating[index]
            user_down[u, :] += self.movie_matrix[:, i] * predict_rating[index]
            item_up[:, i] += self.user_matrix[u, :] * self.train_rating[index]
            item_down[:, i] += self.user_matrix[u, :] * predict_rating[index]
        '''

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

    # deprecated
    def update_matrix(self):
        predict_r = np.dot(self.user_matrix, self.movie_matrix) * self.mask

        # user feature update
        q_transpose = np.transpose(np.copy(self.movie_matrix))
        up = np.dot(self.ground_truth, q_transpose)
        down = np.dot(predict_r, q_transpose) + 1e-5 + np.transpose(
            np.transpose(self.user_matrix) * self.user_rated_count)
        p_update_multiply = up / down

        # movie characteristic update
        p_transpose = np.transpose(np.copy(self.user_matrix))
        up = np.dot(p_transpose, self.ground_truth)
        down = np.dot(p_transpose, predict_r) + 1e-5 + self.movie_matrix * self.movie_rated_count
        # debug
        q_update_multiply = up / down

        # update
        self.user_matrix = self.user_matrix * p_update_multiply
        self.movie_matrix = self.movie_matrix * q_update_multiply


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

            R.fill_ground_truth_numpy()  # use matrix operation

            print(f'Parameters: ({lambda_pq} {feature_num})\n'
                  f'Initial Loss: '
                  f'{R.get_loss_numpy():{12}.{8}}')
            log.write(f'Parameters: {feature_num} {lambda_pq}\n'
                      f'0 {R.get_loss_numpy():{12}.{8}}\n')
            for epoch in range(train_epoch):
                start_time = time.time()

                # R.update_numpy()
                R.update_matrix()  # user matrix operation

                loss = R.get_loss_numpy()
                log_string = f'[Epoch] {epoch+1}, time: {time.time() - start_time:{5}.{4}}, loss {loss:{12}.{8}}'
                print(log_string)
                log.write(f'{epoch+1} {loss}\n')
            print('------------------------------------------------------------------------------------')
            log.write('------------------------------------------------------------------------------------\n')

    # file operation
    log.close()
