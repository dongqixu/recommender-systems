import numpy as np
import sys
import time
import torch
from dataset_io import extract_rating, get_record_index


class RatingMatrix(object):
    # report number different from statistics
    def __init__(self, feature_num, lambda_p, lambda_q):
        # gpu acceleration
        self.cuda_enable = torch.cuda.is_available()
        # dataset
        self.user_num = 6040
        self.movie_num = 3952
        self.rating_list, self.user_rate_count, self.movie_rate_count = \
            extract_rating(self.user_num, self.movie_num)  # pytorch tensor
        self.user_index_list, self.movie_index_list = get_record_index(self.user_num, self.movie_num)

        # parameter
        self.feature_num = feature_num
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q

        # matrix initialization
        if self.cuda_enable:
            print('Cuda enable...')
            self.user_matrix = np.random.random((self.user_num, self.feature_num)).astype(float)
            self.movie_matrix = np.random.random((self.feature_num, self.movie_num)).astype(float)
            self.predict_rating = np.zeros(len(self.rating_list), dtype=float)
            self.user_matrix = torch.from_numpy(self.user_matrix).cuda()
            self.movie_matrix = torch.from_numpy(self.movie_matrix).cuda()
            self.predict_rating = torch.from_numpy(self.predict_rating).cuda()
        else:
            self.user_matrix = torch.FloatTensor(self.user_num, self.feature_num).uniform_(0, 1)
            self.movie_matrix = torch.FloatTensor(self.feature_num, self.movie_num).uniform_(0, 1)
            self.predict_rating = torch.zeros(len(self.rating_list))

        # transpose
        self.transpose_rating = torch.transpose(self.rating_list, dim0=0, dim1=1)
        self.train_user_id, self.train_movie_id, self.train_rating = self.transpose_rating
        if self.cuda_enable:
            self.train_rating = self.train_rating.double()
        else:
            self.train_rating = self.train_rating.float()

        self.train_user_id = self.train_user_id.long()

        # TODO: combine lambda
        # self.user_rate_count = torch.mul(self.user_rate_count, self.lambda_p)
        # self.movie_rate_count = torch.mul(self.movie_rate_count, self.lambda_q)

        print(f'Features: {self.feature_num}\n'
              f'Lambda P: {self.lambda_p}\n'
              f'Lambda Q: {self.lambda_q}\n')

    def get_loss(self):
        # u == i
        # self.user_matrix[self.train_user_id, :] -> (u, k)
        # self.movie_matrix[:, self.train_movie_id]) -> (k, i)

        # zero init
        pointer = 0
        self.predict_rating = self.predict_rating.fill_(0)
        start_time = time.time()
        step = 6040
        for u in range(0, self.user_num, step):
            # further cut on self.user_index_list[u]
            shift = torch.sum(self.user_rate_count[u:u+step])
            user_index = self.train_user_id[pointer:pointer+shift]  # (29415,)
            user_feature = self.user_matrix[user_index, :]  # (29415x100)

            rate_index = torch.cat(self.user_index_list[u:u+step])  # (29415,) index concat
            movie_feature = self.movie_matrix[:, rate_index]  # (100x29415)
            # transpose
            u_prediction = torch.mul(user_feature, torch.t(movie_feature))  # element wise
            u_prediction = torch.sum(u_prediction, dim=1)
            # transpose -> matrix multiplication
            self.predict_rating[pointer:pointer+shift] = u_prediction
            pointer += shift
        print(time.time()-start_time)
        self.predict_rating = self.train_rating - self.predict_rating
        euclidean_distance_loss = (self.predict_rating * self.predict_rating).sum()
        return euclidean_distance_loss

    def update(self):
        start_time = time.time()
        if self.cuda_enable:
            user_up = np.zeros((self.user_num, self.feature_num), dtype=float)
            user_down = np.zeros((self.user_num, self.feature_num), dtype=float)
            item_up = np.zeros((self.feature_num, self.movie_num), dtype=float)
            item_down = np.zeros((self.feature_num, self.movie_num), dtype=float)
            user_up = torch.from_numpy(user_up).cuda()
            user_down = torch.from_numpy(user_down).cuda()
            item_up = torch.from_numpy(item_up).cuda()
            item_down = torch.from_numpy(item_down).cuda()
        else:
            user_up = torch.zeros(self.user_num, self.feature_num)
            user_down = torch.zeros(self.user_num, self.feature_num)
            item_up = torch.zeros(self.feature_num, self.movie_num)
            item_down = torch.zeros(self.feature_num, self.movie_num)

        # zero init
        pointer = 0
        self.predict_rating = self.predict_rating.fill_(0)
        for u in range(self.user_num):
            # further cut on self.user_index_list[u]
            user_feature = self.user_matrix[u, :]
            movie_feature = self.movie_matrix[:, self.user_index_list[u]]
            # transpose -> matrix multiplication
            u_prediction = torch.mm(torch.t(movie_feature), user_feature.unsqueeze(1))
            shift = len(self.user_index_list[u])
            self.predict_rating[pointer:pointer + shift] = u_prediction.view(-1)
            pointer += shift

        # TODO: (k, i) * (i,) why?
        # user_up
        pointer = 0
        for u in range(self.user_num):
            movie_feature = self.movie_matrix[:, self.user_index_list[u]]
            shift = len(self.user_index_list[u])
            user_rate = self.train_rating[pointer:pointer + shift]
            user_predict = self.predict_rating[pointer:pointer + shift]
            user_up[u] = user_up[u] + torch.mm(movie_feature, user_rate.unsqueeze(1)).view(-1)
            user_down[u] = user_down[u] + torch.mm(movie_feature, user_predict.unsqueeze(1)).view(-1)
            pointer += shift

        print(time.time() - start_time)
        exit(1)

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
        # print('Shape', self.user_matrix.shape, self.user_rate_count.shape)
        combine = np.transpose(np.transpose(self.user_matrix) * self.user_rate_count)
        user_down += 1e-5
        user_down += combine
        self.user_matrix *= (user_up / user_down)
        del combine

        # each movie entry
        # print('Shape', self.movie_matrix.shape, self.movie_rate_count.shape)
        combine = self.movie_matrix * self.movie_rate_count
        item_down += 1e-5
        item_down += combine
        self.movie_matrix *= (item_up / item_down)
        del combine


if __name__ == '__main__':
    if torch.cuda.is_available():
        cuda_device = 0
        if len(sys.argv) > 1:
            cuda_device = int(sys.argv[1])
        torch.cuda.set_device(cuda_device)

    # np.random.seed(0)

    # file operation
    line_buffer = 1
    log = open('loss.txt', 'w', buffering=line_buffer)

    # parameter setting
    lambda_pq_list = [0.02*(x+1) for x in range(10)]
    feature_list = [200*(x+1) for x in range(8)]
    train_epoch = 200  # 100?

    R = RatingMatrix(feature_num=100, lambda_p=0.1, lambda_q=0.1)
    print(R.get_loss())
    exit(5)
    R.update()

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

    # file operation
    log.close()
