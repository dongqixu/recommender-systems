import numpy as np
import sys
import time
import torch
from data_block import load_rating_list, load_rate_count_numpy


class RatingMatrix(object):
    # report number different from statistics
    def __init__(self, feature_num, lambda_p, lambda_q):
        # parameter
        coefficient = 10
        self.batch_user_step = 48000  # 480189 // coefficient  # large step for loading batch
        self.batch_movie_step = 1700  # 17770 // coefficient
        self.user_step = 480 * 1  # small step for each computation
        self.movie_step = 17 * 1
        # TODO: calculate 100480507
        self.loading_length_rating = 250000  # maximum of rating number
        self.loading_length_user = self.user_step  # maximum of user number
        self.loading_length_movie = self.movie_step  # maximum of movie number

        self.feature_num = feature_num
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
        # gpu acceleration
        self.cuda_enable = torch.cuda.is_available()
        # dataset
        self.user_num = 480189
        self.movie_num = 17770

        # TODO: batch loading, rating list in two shape
        self.rating_list_user_first = None
        self.transpose_rating_user_first = None
        self.train_user_id_user_group, self.train_movie_id_user_group, self.train_rating_user_group \
            = None, None, None
        self.train_rating_user_group = None
        self.rating_list_movie_first = None
        self.transpose_rating_movie_group = None
        self.train_user_id_movie_group, self.train_movie_id_movie_group, self.train_rating_movie_group \
            = None, None, None
        self.train_rating_movie_group = None
        # init load?
        # self.loading_batch(group='user', batch=0)
        # self.loading_batch(group='movie', batch=0)

        # full loading -> long
        self.user_rate_count_numpy, self.movie_rate_count_numpy = load_rate_count_numpy()
        '''double type for combination of lambda'''
        self.user_rate_count_double = torch.from_numpy(self.user_rate_count_numpy).double()
        self.movie_rate_count_double = torch.from_numpy(self.movie_rate_count_numpy).double()
        if self.cuda_enable:
            self.user_rate_count_double = self.user_rate_count_double.cuda()
            self.movie_rate_count_double = self.movie_rate_count_double.cuda()
        # TODO: combine lambda
        self.user_rate_count_double = torch.mul(self.user_rate_count_double, self.lambda_p)
        self.movie_rate_count_double = torch.mul(self.movie_rate_count_double, self.lambda_q)

        '''the key memory usage'''
        # full loading -> matrix initialization -> numpy
        self.user_matrix = np.random.random((self.user_num, self.feature_num)).astype(float)
        self.movie_matrix = np.random.random((self.feature_num, self.movie_num)).astype(float)
        # TODO: loading length
        self.predict_rating_user_group = np.zeros(self.loading_length_rating, dtype=float)
        self.predict_rating_movie_group = np.zeros(self.loading_length_rating, dtype=float)
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

        '''the key of memory reuse'''
        # TODO: part loading!
        self.user_up = np.zeros((self.loading_length_user, self.feature_num), dtype=float)
        self.user_down = np.zeros((self.loading_length_user, self.feature_num), dtype=float)
        self.item_up = np.zeros((self.feature_num, self.loading_length_movie), dtype=float)
        self.item_down = np.zeros((self.feature_num, self.loading_length_movie), dtype=float)
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

        # time init
        self.init_time = time.time()
        print('Init finished')

    def loading_batch(self, group, batch):
        if group == 'user':
            self.rating_list_user_first = load_rating_list(batch=batch, group=group)
            self.transpose_rating_user_first = torch.transpose(self.rating_list_user_first, dim0=0, dim1=1)
            self.train_user_id_user_group, self.train_movie_id_user_group, self.train_rating_user_group \
                = self.transpose_rating_user_first
            self.train_rating_user_group = self.train_rating_user_group.double()
        elif group == 'movie':
            self.rating_list_movie_first = load_rating_list(batch=batch, group=group)
            self.transpose_rating_movie_group = torch.transpose(self.rating_list_movie_first, dim0=0, dim1=1)
            self.train_user_id_movie_group, self.train_movie_id_movie_group, self.train_rating_movie_group \
                = self.transpose_rating_movie_group
            self.train_rating_movie_group = self.train_rating_movie_group.double()

    '''draft'''
    def compute_prediction(self):
        # zero init
        step = self.user_step
        for u_head in range(0, self.user_num, self.batch_user_step):
            pointer = 0
            self.loading_batch(group='user', batch=int(u_head/self.batch_user_step))
            for _u in range(0, self.batch_user_step, self.user_step):
                self.predict_rating_user_group = self.predict_rating_user_group.fill_(0)
                u = u_head + _u
                # size of (u, i) pair
                shift = np.sum(self.user_rate_count_numpy[u:u + step])
                print(shift)
                user_index = self.train_user_id_user_group[pointer:pointer+shift]  # (1000209,)
                print(user_index.size())
                user_feature = self.user_matrix[user_index, :]  # (1000209, 100)
                print(user_feature.size())
                movie_index = self.train_movie_id_user_group[pointer:pointer+shift]  # (1000209,)
                print(movie_index.size())
                movie_feature = self.movie_matrix[:, movie_index]  # (100, 1000209)
                print(movie_feature.size())
                # element wise operation -> transpose
                u_prediction = torch.mul(user_feature, torch.t(movie_feature))
                print(u_prediction.size())
                self.predict_rating_user_group[0:shift] = torch.sum(u_prediction, dim=1)
                '''no storage for each prediction, function to be added!'''
                pointer += shift
                print('succeed once')
                break
            break
        # zero init
        step = self.movie_step
        for i_head in range(0, self.movie_num, self.batch_movie_step):
            pointer = 0
            self.loading_batch(group='movie', batch=int(i_head/self.batch_movie_step))
            for _i in range(0, self.batch_movie_step, self.movie_step):
                self.predict_rating_movie_group = self.predict_rating_movie_group.fill_(0)
                i = i_head + _i
                # size of (u, i) pair
                shift = np.sum(self.movie_rate_count_numpy[i:i + step])
                print(shift)
                user_index = self.train_user_id_movie_group[pointer:pointer+shift]  # (1000209,)
                print(user_index.size())
                user_feature = self.user_matrix[user_index, :]  # (1000209, 100)
                print(user_feature.size())
                movie_index = self.train_movie_id_movie_group[pointer:pointer+shift]  # (1000209,)
                print(movie_index.size())
                movie_feature = self.movie_matrix[:, movie_index]  # (100, 1000209)
                print(movie_feature.size())
                # element wise operation -> transpose
                i_prediction = torch.mul(user_feature, torch.t(movie_feature))
                print(i_prediction.size(0))
                self.predict_rating_movie_group[0:shift] = torch.sum(i_prediction, dim=1)
                # TODO: hey
                '''no storage for each prediction, function to be added!'''
                pointer += shift
                print('succeed twice')
                exit(405)
                break

    '''not write'''
    # get loss does not compute prediction
    def get_loss(self):
        # compute prediction
        # self.compute_prediction()
        # init_time = time.time()
        self.predict_rating_user_group = self.train_rating_user_group - self.predict_rating_user_group
        euclidean_distance_loss = torch.sum(self.predict_rating_user_group * self.predict_rating_user_group)
        # print('loss computation time: ', time.time() - init_time)
        # print(euclidean_distance_loss)
        # self.predict_rating_movie_group = self.train_rating_movie_group - self.predict_rating_movie_group
        # euclidean_distance_loss = (self.predict_rating_movie_group * self.predict_rating_movie_group).sum()
        # print(euclidean_distance_loss)
        return euclidean_distance_loss

    '''not write'''
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
        print('init time, include prediction', end_time - start_time)

        # user related
        # zero init
        pointer = 0
        step = self.user_step
        for u in range(0, self.user_num, step):
            # size of (u, i) pair
            shift = np.sum(self.user_rate_count_numpy[u:u + step])
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
            pointer += shift
        print('user related compute', time.time() - end_time)
        end_time = time.time()

        # item related
        pointer = 0
        step = self.movie_step
        for i in range(0, self.movie_num, step):
            # size of (u, i) pair
            shift = np.sum(self.movie_rate_count_numpy[i:i + step])
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
            pointer += shift
        print('movie related compute', time.time() - end_time)
        end_time = time.time()

        # each user entry
        '''lambda_p: constant, I_u: (user_num,), p_uk: (user_num, k)'''
        '''create a I_u member variable, put and give shifting'''
        # print('Shape', self.user_matrix.shape, self.user_rate_count_numpy.shape)
        combine = torch.t(torch.t(self.user_matrix) * self.user_rate_count_double.unsqueeze(1).expand(
            self.user_num, self.feature_num))
        self.user_down += 1e-5
        self.user_down += combine
        self.user_matrix = self.user_matrix * (self.user_up / self.user_down)
        print('user entry: ', time.time() - end_time)
        end_time = time.time()

        # each movie entry
        # print('Shape', self.movie_matrix.shape, self.movie_rate_count_numpy.shape)
        combine = self.movie_matrix * self.movie_rate_count_double.unsqueeze(0).expand(
            self.feature_num, self.movie_num)
        self.item_down += 1e-5
        self.item_down += combine
        self.movie_matrix = self.movie_matrix * (self.item_up / self.item_down)
        print('movie entry: ', time.time() - end_time)

        # print('[Pass]')

    def get_time(self):
        current = time.time()
        t = current - self.init_time
        self.init_time = current
        return t

    def set_time(self):
        self.init_time = time.time()


if __name__ == '__main__':
    if torch.cuda.is_available():
        cuda_device = 0
        if len(sys.argv) > 1:
            cuda_device = int(sys.argv[1])
        torch.cuda.set_device(cuda_device)

    np.random.seed(0)

    # file operation
    # line_buffer = 1
    # log = open('loss.txt', 'w', buffering=line_buffer)
    log = open('loss.txt', 'w')

    R = RatingMatrix(feature_num=1000, lambda_p=0.02, lambda_q=0.02)
    R.compute_prediction()

    # parameter setting
    lambda_pq_list = [0.02*(x+1) for x in range(10)]
    feature_list = [200*(x+1) for x in range(8)]
    train_epoch = 100  # 100?

    # R = RatingMatrix(feature_num=100, lambda_p=0.1, lambda_q=0.1)
    # R.get_loss()
    # for i in range(100):
    #     print('---------------------loss:', R.get_loss())
    #     R.update()

    for feature_num in feature_list:
        for lambda_pq in lambda_pq_list:
            R = RatingMatrix(feature_num=feature_num, lambda_p=lambda_pq, lambda_q=lambda_pq)
            R.compute_prediction()
            init_loss = R.get_loss()
            print(f'Parameters: ({lambda_pq} {feature_num})\n'
                  f'Initial Loss: {init_loss:{12}.{8}}')
            log.write(f'Parameters: {feature_num} {lambda_pq}\n'
                      f'0 {init_loss:{12}.{8}}\n')
            loss = init_loss
            for epoch in range(train_epoch):
                start_time = time.time()

                R.update()

                if (epoch+1) % 10 == 0:
                    loss = R.get_loss()
                    print('[loss] ', loss)

                t = time.time() - start_time
                log_string = f'[Epoch] {epoch+1}, time: {init_time:{5}.{4}}'
                print(log_string)
                log.write(f'{epoch+1}\n')
            print('------------------------------------------------------------------------------------')
            log.write('------------------------------------------------------------------------------------\n')
    # file operation
    log.close()
