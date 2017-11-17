import numpy as np
import sys
import time
import torch
from dataset_io import extract_rating_with_count, get_rating_index


class RatingMatrix(object):
    # report number different from statistics
    def __init__(self, feature_num, lambda_p, lambda_q):
        # parameter
        self.user_step = 6040
        self.movie_step = 3952
        self.feature_num = feature_num
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q

        # gpu acceleration
        self.cuda_enable = torch.cuda.is_available()
        # dataset
        self.user_num = 6040
        self.movie_num = 3952
        # rating list in two shape
        self.rating_list_user_first, self.user_rate_count, self.movie_rate_count = \
            extract_rating_with_count(self.user_num, self.movie_num)
        self.rating_list_movie_first, _, _ = \
            extract_rating_with_count(self.user_num, self.movie_num, user_movie_order=False)
        self.user_index, self.movie_index = get_rating_index(self.user_num, self.movie_num)
        '''double type for combination of lambda'''
        self.user_rate_count_double = self.user_rate_count.double()
        self.movie_rate_count_double = self.movie_rate_count.double()

        # matrix initialization -> numpy
        self.user_matrix = np.random.random((self.user_num, self.feature_num)).astype(float)
        self.movie_matrix = np.random.random((self.feature_num, self.movie_num)).astype(float)
        self.predict_rating_user_group = np.zeros(len(self.rating_list_user_first), dtype=float)
        self.predict_rating_movie_group = np.zeros(len(self.rating_list_movie_first), dtype=float)
        # pytorch
        self.user_matrix = torch.from_numpy(self.user_matrix)
        self.movie_matrix = torch.from_numpy(self.movie_matrix)
        self.predict_rating_user_group = torch.from_numpy(self.predict_rating_user_group)
        self.predict_rating_movie_group = torch.from_numpy(self.predict_rating_movie_group)
        if self.cuda_enable:
            print('Cuda is enabled.')
            self.user_matrix = self.user_matrix.cuda()
            self.movie_matrix = self.movie_matrix.cuda()
            self.predict_rating_user_group = self.predict_rating_user_group.cuda()
            self.predict_rating_movie_group = self.predict_rating_movie_group.cuda()

        # TODO: transpose -> order? copy problem?
        self.transpose_rating_user_first = torch.transpose(self.rating_list_user_first, dim0=0, dim1=1)
        self.train_user_id_user_group, self.train_movie_id_user_group, self.train_rating_user_group \
            = self.transpose_rating_user_first
        self.transpose_rating_movie_group = torch.transpose(self.rating_list_movie_first, dim0=0, dim1=1)
        self.train_user_id_movie_group, self.train_movie_id_movie_group, self.train_rating_movie_group \
            = self.transpose_rating_movie_group
        # convert train rating from long to double
        self.train_rating_user_group = self.train_rating_user_group.double()
        self.train_rating_movie_group = self.train_rating_movie_group.double()

        # TODO: combine lambda
        self.user_rate_count_double = torch.mul(self.user_rate_count_double, self.lambda_p)
        self.movie_rate_count_double = torch.mul(self.movie_rate_count_double, self.lambda_q)

        # print(f'Features: {self.feature_num}\n'
        #       f'Lambda P: {self.lambda_p}\n'
        #       f'Lambda Q: {self.lambda_q}\n')

        # temp variable allocation -> numpy
        self.user_up = np.zeros((self.user_num, self.feature_num), dtype=float)
        self.user_down = np.zeros((self.user_num, self.feature_num), dtype=float)
        self.item_up = np.zeros((self.feature_num, self.movie_num), dtype=float)
        self.item_down = np.zeros((self.feature_num, self.movie_num), dtype=float)
        # pytorch
        self.user_up = torch.from_numpy(self.user_up)
        self.user_down = torch.from_numpy(self.user_down)
        self.item_up = torch.from_numpy(self.item_up)
        self.item_down = torch.from_numpy(self.item_down)
        if self.cuda_enable:
            self.user_up = self.user_up.cuda()
            self.user_down = self.user_down.cuda()
            self.item_up = self.item_up.cuda()
            self.item_down = self.item_down.cuda()

    def compute_prediction(self):
        # zero init
        pointer = 0
        step = self.user_step
        self.predict_rating_user_group = self.predict_rating_user_group.fill_(0)
        for u in range(0, self.user_num, step):
            # size of (u, i) pair
            shift = torch.sum(self.user_rate_count[u:u+step])
            user_index = self.train_user_id_user_group[pointer:pointer+shift]  # (1000209,)
            user_feature = self.user_matrix[user_index, :]  # (1000209, 100)
            movie_index = self.train_movie_id_user_group[pointer:pointer+shift]  # (1000209,)
            movie_feature = self.movie_matrix[:, movie_index]  # (100, 1000209)
            # element wise operation -> transpose
            u_prediction = torch.mul(user_feature, torch.t(movie_feature))
            self.predict_rating_user_group[pointer:pointer + shift] = torch.sum(u_prediction, dim=1)
            pointer += shift
        # zero init
        pointer = 0
        step = self.movie_step
        self.predict_rating_movie_group = self.predict_rating_movie_group.fill_(0)
        for i in range(0, self.movie_num, step):
            # size of (u, i) pair
            shift = torch.sum(self.movie_rate_count[i:i+step])
            user_index = self.train_user_id_movie_group[pointer:pointer+shift]  # (1000209,)
            user_feature = self.user_matrix[user_index, :]  # (1000209, 100)
            movie_index = self.train_movie_id_movie_group[pointer:pointer+shift]  # (1000209,)
            movie_feature = self.movie_matrix[:, movie_index]  # (100, 1000209)
            # element wise operation -> transpose
            i_prediction = torch.mul(user_feature, torch.t(movie_feature))
            self.predict_rating_movie_group[pointer:pointer + shift] = torch.sum(i_prediction, dim=1)
            pointer += shift

    def get_loss(self):
        # compute prediction
        self.compute_prediction()

        self.predict_rating_user_group = self.train_rating_user_group - self.predict_rating_user_group
        euclidean_distance_loss = (self.predict_rating_user_group * self.predict_rating_user_group).sum()
        print(euclidean_distance_loss)
        self.predict_rating_movie_group = self.train_rating_movie_group - self.predict_rating_movie_group
        euclidean_distance_loss = (self.predict_rating_movie_group * self.predict_rating_movie_group).sum()
        print(euclidean_distance_loss)
        return euclidean_distance_loss

    def update(self):
        start_time = time.time()
        # zero init
        self.user_up = self.user_up.fill_(0)
        self.user_down = self.user_down.fill_(0)
        self.item_up = self.item_up.fill_(0)
        self.item_down = self.item_down.fill_(0)

        # # compute prediction
        self.compute_prediction()

        end_time = time.time()
        # print('init time', end_time - start_time)

        # user related
        # zero init
        pointer = 0
        step = self.user_step
        for u in range(0, self.user_num, step):
            # size of (u, i) pair
            shift = torch.sum(self.user_rate_count[u:u + step])
            movie_index = self.train_movie_id_user_group[pointer:pointer+shift]  # (1000209,)
            movie_feature = self.movie_matrix[:, movie_index]  # (100, 1000209)
            _k, _i = movie_feature.size()
            # print(_i, _k)
            u_true_rating = self.train_rating_user_group[pointer:pointer+shift]
            u_prediction = self.predict_rating_user_group[pointer:pointer + shift]
            user_up_add = torch.t(movie_feature) * u_true_rating.unsqueeze(1).expand(_i, _k)
            user_down_add = torch.t(movie_feature) * u_prediction.unsqueeze(1).expand(_i, _k)
            user_index = self.train_user_id_user_group[pointer:pointer + shift]  # (1000209,)
            self.user_up.index_add_(0, user_index, user_up_add)
            self.user_down.index_add_(0, user_index, user_down_add)
            # print(self.user_up.size(), self.user_down.size())
            # print(time.time()-end_time)
            pointer += shift

        end_time = time.time()
        # item related
        pointer = 0
        step = self.movie_step
        for i in range(0, self.movie_num, step):
            # size of (u, i) pair
            shift = torch.sum(self.movie_rate_count[i:i + step])
            user_index = self.train_user_id_movie_group[pointer:pointer + shift]  # (1000209,)
            user_feature = self.user_matrix[user_index, :]  # (1000209, 100)
            _u, _k = user_feature.size()
            # print(_k, _u)
            i_true_rating = self.train_rating_movie_group[pointer:pointer+shift]
            i_prediction = self.predict_rating_movie_group[pointer:pointer + shift]
            item_up_add = torch.t(user_feature) * i_true_rating.unsqueeze(0).expand(_k, _u)
            item_down_add = torch.t(user_feature) * i_prediction.unsqueeze(0).expand(_k, _u)
            movie_index = self.train_movie_id_movie_group[pointer:pointer + shift]  # (1000209,)
            # print(item_up_add.size())
            torch.t(self.item_up).index_add_(0, movie_index, torch.t(item_up_add))
            torch.t(self.item_down).index_add_(0, movie_index, torch.t(item_down_add))
            # print(self.item_up.size())  # , self.item_down.size()
            # print(time.time()-end_time)
            pointer += shift
        '------------------------------------------------------------------------------'

        # each user entry
        '''lambda_p: constant, I_u: (user_num,), p_uk: (user_num, k)'''
        '''create a I_u member variable, put and give shifting'''
        # print('Shape', self.user_matrix.shape, self.user_rate_count.shape)
        combine = torch.t(torch.t(self.user_matrix) * self.user_rate_count_double.unsqueeze(1).expand(
            self.user_num, self.feature_num))
        self.user_down += 1e-5
        self.user_down += combine
        self.user_matrix = self.user_matrix * (self.user_up / self.user_down)

        # each movie entry
        # print('Shape', self.movie_matrix.shape, self.movie_rate_count.shape)
        combine = self.movie_matrix * self.movie_rate_count_double.unsqueeze(0).expand(
            self.feature_num, self.movie_num)
        self.item_down += 1e-5
        self.item_down += combine
        self.movie_matrix = self.movie_matrix * (self.item_up / self.item_down)

        # print('Pass')


if __name__ == '__main__':
    if torch.cuda.is_available():
        cuda_device = 0
        if len(sys.argv) > 1:
            cuda_device = int(sys.argv[1])
        torch.cuda.set_device(cuda_device)

    np.random.seed(0)

    # file operation
    line_buffer = 1
    log = open('loss.txt', 'w', buffering=line_buffer)

    # parameter setting
    lambda_pq_list = [0.02*(x+1) for x in range(10)]
    feature_list = [200*(x+1) for x in range(8)]
    train_epoch = 200  # 100?

    R = RatingMatrix(feature_num=100, lambda_p=0.1, lambda_q=0.1)
    R.get_loss()
    for i in range(100):
        print('---------------------loss:', R.get_loss())
        R.update()

    '''
    for feature_num in feature_list:
        for lambda_pq in lambda_pq_list:
            R = RatingMatrix(feature_num=feature_num, lambda_p=lambda_pq, lambda_q=lambda_pq)

            print(f'Parameters: ({lambda_pq} {feature_num})\n'
                  f'Initial Loss: '
                  f'{R.get_loss():{12}.{8}}')
            log.write(f'Parameters: {feature_num} {lambda_pq}\n'
                      f'0 {R.get_loss():{12}.{8}}\n')
            for epoch in range(train_epoch):
                start_time = time.time()

                # R.update_numpy()

                loss = R.get_loss()
                log_string = f'[Epoch] {epoch+1}, time: {time.time() - start_time:{5}.{4}},' \
                             f' loss {loss:{12}.{8}}'
                print(log_string)
                log.write(f'{epoch+1} {loss}\n')
            print('------------------------------------------------------------------------------------')
            log.write('------------------------------------------------------------------------------------\n')
    '''
    # file operation
    log.close()
