import math
import matplotlib.pyplot as plt

total_count = 1000209


def get_file(filename):
    f = open(filename, 'r')
    epoch = []
    list_of_epoch = []
    for line in f:
        if line != '------------------------------------------------------------------------------------\n':
            epoch.append(line)
        else:
            list_of_epoch.append(epoch)
            epoch = []
    return list_of_epoch

# print(list_of_epoch)


def get_data(list_of_epoch):
    _rsme = []
    _rsme_list = []
    feature = []
    lambda_pq = []
    for each_epoch in list_of_epoch:
        for i in each_epoch:
            i = i.strip('\n')
            i = i.split()
            if i[0] != 'Parameters:':
                rsme = math.sqrt(float(i[1]) / total_count)
                # print(rsme)
                _rsme.append(rsme)
                # print(_rsme)
            else:
                # print(i)
                feature.append(i[1])
                lambda_pq.append(i[2])
                # label.append()
                _rsme_list.append(_rsme)
                _rsme = []
    _rsme_list.append(_rsme)
    del _rsme_list[0]
    return _rsme_list,feature,lambda_pq
# print(feature)
# print(RSME_list[0])


def plot(x, _rsme_list, feature, lambda_pq):
    for i in range(len(_rsme_list)):
        plt.plot(x, _rsme_list[i], label='feature:' + feature[i] + ';lambda:' + lambda_pq[i])
        # print('feature:'+feature[i]+';lambda:'+lambda_pq[i])
    plt.xlabel('epoch times')
    plt.ylabel('RSME')
    plt.legend()
    plt.show()


# x = epoch, y = RSME
def plot_fixed_lambda(x, _rsme_list, feature, lambda_pq):
    index = [index for index,value in enumerate(lambda_pq) if value == '0.02']  # located
    for i in index:
        plt.plot(x, _rsme_list[i], label ='feature:' + feature[i] + ';lambda:' + lambda_pq[i])
    plt.xlabel('epoch times')
    plt.ylabel('RSME')
    plt.legend()
    plt.show()


# x = lambda, y = RSME
def plot_lambda(lambda_pq, feature, _rsme_list):
    x = list(set(lambda_pq))
    x.sort(key=lambda_pq.index)
    rsme = []
    # slice_rsme = []
    for i in _rsme_list:
        rsme.append(i[-1])
    for i in range(0, len(rsme),10):
        slice_rsme = rsme[i:i+10]
        print(slice_rsme)
        plt.plot(x, slice_rsme, label='feature:'+feature[i])
        # slice_rsme = []
    plt.xlabel('lambda')
    plt.ylabel('RSME')
    plt.legend()
    plt.show()


list_of_epoch_1000 = get_file('loss_1000.txt')
list_of_epoch_200 = get_file('loss_200.txt')
RSME_list_200, feature_200, lambda_200 = get_data(list_of_epoch_200)
RSME_list_1000, feature_1000, lambda_1000 = get_data(list_of_epoch_1000)

print(feature_1000)

# print(len(lambda_200))
# print(feature_200)
# print(len(RSME_list_200))
x1 = range(201)
x2 = range(1001)
# print(lambda_200)
plot_lambda(lambda_200,feature_200,RSME_list_200)

# plot(x2,RSME_list_1000,feature_1000,lambda_1000)
# plot_fixed_lambda(x1,RSME_list_200,feature_200,lambda_200)
# plot(x1)
# plot_fixed_lambda(x2)
