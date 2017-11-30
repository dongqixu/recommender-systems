import math
import matplotlib.pyplot as plt

# total_count =  1000209
total_count = 100480507


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
    RSME = []
    RSME_list = []
    feature = []
    lambdapq = []
    for each_epoch in list_of_epoch:
        for i in each_epoch:
            i = i.strip('\n')
            i = i.split()
            if i[0] != 'Parameters:':
                rsme = math.sqrt(float(i[1]) / total_count)
                # print(rsme)
                RSME.append(rsme)
                # print(RSME)
            else:
                # print(i)
                feature.append(i[1])
                lambdapq.append(i[2])
                # label.append()
                RSME_list.append(RSME)
                RSME = []
    RSME_list.append(RSME)
    del RSME_list[0]
    return RSME_list,feature,lambdapq
# print(feature)
# print(RSME_list[0])


def plot(x, RSME_list, feature, lambdapq):
    for i in range(len(RSME_list)):
        plt.plot(x, RSME_list[i], label='feature:'+feature[i]+';lambda:'+lambdapq[i])
        # print('feature:'+feature[i]+';lambda:'+lambdapq[i])
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel('epoch times')
    plt.ylabel('RSME')
    plt.legend(loc=2, bbox_to_anchor=(0.95,1.0),borderaxespad = 0.)
    plt.show()


# x = epoch, y = RSME
def plot_fixed_lambda(x, RSME_list, feature, lambdapq):
    index = [index for index, value in enumerate(lambdapq) if value =='0.02']  # located
    for i in index:
        plt.plot(x, RSME_list[i], label='feature:'+feature[i]+';lambda:'+lambdapq[i])
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel('epoch times')
    plt.ylabel('RSME')
    plt.legend()
    plt.show()


# x = lambda, y = RSME
def plot_lambda(lambdapq, feature, RSME_list):
    x = list(set(lambdapq))
    x.sort(key=lambdapq.index)
    rsme = []
    slice_rsme = []
    for i in RSME_list:
        rsme.append(i[-1])
    print(rsme)
    # for i in range(0,len(rsme),10): # 200 times
    for i in range(0,len(rsme)-6,10): # 1000 times
        slice_rsme = rsme[i:i+10]
        # print(slice_rsme)
        plt.plot(x,slice_rsme,label = 'feature:'+feature[i])
        slice_rsme = []
    slice_rsme = rsme[len(rsme)-6:len(rsme)]  # 1000 times
    plt.plot(x[0:6], slice_rsme, label='feature:'+feature[-1] )  # 1000 times
    ax = plt.gca()
    # ax.set_yscale('log')
    plt.xlabel('lambda')
    plt.ylabel('RSME')
    plt.legend()
    plt.show()


def plot_features_RMSE(feature,RSME_list,lambdapq):
    print(feature)
    # print(len(feature))
    print(lambdapq)
    x = list(set(feature))
    x.sort(key=feature.index)
    x = [int(i) for i in x]
    print(x)
    rsme = []
    for i in RSME_list:
        rsme.append(i[-1])
    # print(len(rsme))
    for i in range(10):  # 200 times
    # for i in range(6):  # 1000 times
        # print(rsme[i::10])
        plt.plot(x, rsme[i::10], label='lambda:'+lambdapq[i])
    '''    
    # 1000 times
    for i in range(6,10):
        print(x[0:5])
        print(rsme[i::10])
        plt.plot(x[0:5],rsme[i::10],label = 'lambda:'+lambdapq[i])
    '''
    plt.xlabel('feature')
    plt.ylabel('RMSE')
    plt.legend(loc='upper right')
    plt.show()


def plot_file(filename1, filename2):
    f1 = open(filename1, 'r')
    f2 = open(filename2, 'r')
    y = []
    for line in f1:
        line = math.sqrt(float(line) / total_count)
        print(line)
        y.append(line)
    plt.plot(range(len(y)), y, label='dropout')
    y = []
    for line in f2:
        line = math.sqrt(float(line) / total_count)
        print(line)
        y.append(line)
    plt.plot(range(len(y)), y, label='no_dropout')
    # ax = plt.gca()
    # ax.set_yscale('log')
    plt.xlabel('epoch times')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()


plot_file('loss/loss_dropout.txt', 'loss/loss_no_dropout.txt')


list_of_epoch_1000 = get_file('loss/loss_1000.txt')
list_of_epoch_200 = get_file('loss/loss_200.txt')
RSME_list_200, feature_200, lambda_200 = get_data(list_of_epoch_200)
RSME_list_1000, feature_1000, lambda_1000 = get_data(list_of_epoch_1000)


# print(feature_1000)
# print(lambda_200)
# print(len(lambda_200))
# print(feature_200)
# print(len(RSME_list_200))
x1 = range(201)
x2 = range(1001)
# plot_features_RMSE(feature_1000,RSME_list_1000,lambda_1000)
# plot_features_RMSE(feature_200,RSME_list_200,lambda_200)
# print(lambda_200)
# plot_lambda(lambda_1000,feature_1000,RSME_list_1000)
# plot_lambda(lambda_200,feature_200,RSME_list_200)
# plot(x1,RSME_list_200,feature_200,lambda_200)
# plot(x2,RSME_list_1000,feature_1000,lambda_1000)
# plot_fixed_lambda(x2,RSME_list_1000,feature_1000,lambda_1000)
# plot_fixed_lambda(x1,RSME_list_200,feature_200,lambda_200)
# plot(x1)
# plot_fixed_lambda(x2)
