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
        # self.mask = np.zeros((self.user_num, self.movie_num), dtype=np.int)
        # self.ground_truth = np.zeros((self.user_num, self.movie_num), dtype=np.int)
        self.user_matrix = np.random.random((self.user_num, self.feature_num)).astype(np.float64)
        self.movie_matrix = np.random.random((self.feature_num, self.movie_num)).astype(np.float64)

        # transpose
        self.transpose_rating = np.transpose(np.copy(self.rating_list))  # problem with memory sharing?
        self.train_user_id, self.train_movie_id, self.train_rating = self.transpose_rating[0:3, :]
        self.train_user_id -= 1
        self.train_movie_id -= 1

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
        # index start from zero
        for _, temp_rating in enumerate(self.rating_list):
            user_id, movie_id, user_movie_rating, _ = temp_rating
            user_id -= 1
            movie_id -= 1
            self.ground_truth[user_id, movie_id] = user_movie_rating
            self.mask[user_id, movie_id] = 1

    def fill_ground_truth_numpy(self):
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
        error = self.train_rating - predict_rating
        euclidean_distance_loss = np.sum(error * error)
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
                user_down[u][k] += 1e-5
                # update
                self.user_matrix[u][k] *= (user_up[u][k] / user_down[u][k])

        # each movie entry
        for i in range(self.movie_num):
            for k in range(self.feature_num):
                item_down[k][i] += 1e-5
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
        user_down += 1e-5
        self.user_matrix *= (user_up / user_down)

        # each movie entry
        item_down += 1e-5
        self.movie_matrix *= (item_up / item_down)

    # deprecated
    def update_matrix(self):
        predict_r = np.dot(self.user_matrix, self.movie_matrix) * self.mask

        # user feature update
        q_transpose = np.transpose(np.copy(self.movie_matrix))
        up = np.dot(self.ground_truth, q_transpose)
        down = np.dot(predict_r, q_transpose) + 1e-5
        p_update_multiply = up / down

        # movie characteristic update
        p_transpose = np.transpose(np.copy(self.user_matrix))
        up = np.dot(p_transpose, self.ground_truth)
        down = np.dot(p_transpose, predict_r) + 1e-5
        # debug
        q_update_multiply = up / down

        # update
        self.user_matrix = self.user_matrix * p_update_multiply
        self.movie_matrix = self.movie_matrix * q_update_multiply


if __name__ == '__main__':
    np.random.seed(0)
    start_time = time.time()

    R = RatingMatrix(feature_num=200, lambda_p=0.1, lambda_q=0.1)
    print(f'Loss: {R.get_loss_numpy()}')

    # R.update_loop_based()
    for i in range(5):
        R.update_numpy()
        print(f'Loss: {R.get_loss_numpy()}')
    # R.update_matrix_based()

    print(f'Loss: {R.get_loss_numpy()}')
    print(f'time: {time.time() - start_time:{5}.{4}}, loss')

    '''
    train_epoch = 50

    feature_list = [5000*(x+1) for x in range(3)]

    for feature_num in feature_list:
        R = RatingMatrix(feature_num=feature_num, lambda_p=0.1, lambda_q=0.1)
        R.print_parameters()
        print(f'Initial loss: {R.get_loss_matrix_based()}')
        for epoch in range(train_epoch):
            start_time = time.time()
            # R.update_loop_based()
            R.update_matrix_based()
            loss = R.get_loss_matrix_based()
            print(f'[Epoch] {epoch+1}, time: {time.time() - start_time:{5}.{4}}, loss {loss:{12}.{8}}')
        print('------------------------------------------------------------------------------------')
    '''
