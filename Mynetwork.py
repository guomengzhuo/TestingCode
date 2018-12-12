from __future__ import absolute_import
import numpy as np
import torch
from torch.nn import Module, Parameter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#input: batchsize * feature * 1



class GMZ(torch.nn.Module):
    def __init__(self, input1,input2,**kwargs):  # input1: K*M input2 : num of criteria
        super(GMZ, self).__init__()
        self.layer1 = torch.nn.Conv1d(input1,input1,1,1,0,bias=0,groups=input1)
        # self.layer2 = torch.nn.Conv1d(input1,input2,1,1,0,bias=0,groups=input2)  # group = number of criteria
        self.layer2 = torch.nn.Conv1d(input2,input2,1,1,0,bias=0,groups=input2)
        # self.layer3_1 = torch.nn.Linear(input2,1)
        # self.layer3_1 = torch.nn.Sequential(
        #     torch.nn.Linear(input2,1,bias=False),
        #     torch.nn.Sigmoid(),
        # )
        self.layer3_2 = torch.nn.Sequential(
            torch.nn.Linear(input2, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16,32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32,1),
            torch.nn.Sigmoid(),
        )
        #self.layer4 = fn( out_feature=1)

        self.layer_weight = torch.nn.Sequential(
            torch.nn.Linear(input1, 1),
            torch.nn.Sigmoid()
        )
        #############Test with Softmax
        # self.layer_weight = torch.nn.Sequential(
        #     torch.nn.Linear(input1, 16),
        #     torch.nn.Linear(16, 32),
        #     torch.nn.Linear(32, 2),
        #     torch.nn.Softmax(dim=0)
        # )
        #self.layer_weight = torch.Tensor(torch.rand(1)).requires_grad_(True)
        self.layer_normalized_weight = torch.nn.Sequential(
            torch.nn.Linear(input2, input2, bias=True),
            torch.nn.Softmax(dim=0),
        )  # 归一化权重和为1 且大于等于0
        self.layer_sig = torch.nn.Sigmoid()


    def forward(self, x, mean, K, train_size, num_criteria):
        y1 = self.layer1(x)
        # print(y1.shape)
        temp = y1.detach().numpy().reshape((train_size, K*num_criteria))
        # print(temp.shape)
        ############# y1实际上相当于每K个项目加合到一起 转化后的y1才是有M（准则数）个边际效用值
        sum_y1 = []
        for item in temp:
            b = [item[i:i + K] for i in range(0, len(item), K)] # K is the polynomial degree
            b = np.array(b)
            b = np.sum(b, axis=1)
            sum_y1.append(b)
        sum_y1 = np.array(sum_y1)
        # print(sum_y1.shape)
        y1 = torch.from_numpy(sum_y1).reshape((train_size, num_criteria, 1 ))
        #############
        #############
        # Normalized weights就是和为1 的权重 y2只起到一个提供此权重并且根据y2去训练权重的作用
        y2 = self.layer2(y1)
        y2 = y2.view(y2.size(0), -1)
        temp2 = y2.detach().numpy()
        mean_temp2 = np.mean(temp2, axis=0)
        mean_temp2 = torch.from_numpy(mean_temp2)
        mean_temp2 = mean_temp2.type(torch.FloatTensor)
        normalized_weights = self.layer_normalized_weight(mean_temp2)
        # print(normalized_weights)
        # print(y2.shape)
        # print(normalized_weights.shape)
        ###############
        y1 = y1.view(y1.size(0), -1)
        y3_1 = torch.mm(y1, normalized_weights.view(normalized_weights.size(0), -1 ))
        # y3_1 = self.layer3_1(y2)
        # print(y3_1.shape)
        y3_1 = self.layer_sig(y3_1)
        #########
        #y3_1 和y3_2 输入应该是按类家和的y1

        #########
        y3_2 = self.layer3_2(y1)  # MLP的输入应该是各个没有加权重的边际效用值 也就是转化后的y1 而不是y2
        #y3 = torch.cat((y3_1, y3_2), dim=1)
        ################### Layer using Sigmoid()
        weight = self.layer_weight(mean)
        y3 = y3_1 * weight + y3_2 * (1 - weight)
        ###################
        # weight, weight1 = self.layer_weight(mean)
        # y3 = y3_1 * weight + y3_2 * weight1
        ##################
        # print("weight is:")
        # print(weight)
        #y4 = self.layer4(y3)
        return y3, weight, normalized_weights




# class fn(Module):
#
#     def __init__(self, out_feature):
#         super(fn, self).__init__()
#         self.layer = torch.nn.Sigmoid()
#         self.weight = Parameter(torch.Tensor(out_feature))
#         torch.nn.init.constant_(self.weight, 0.0)
#
#     def forward(self, input):
#         w_x = self.layer(self.weight)
#         w_y = 1 - w_x
#
#         return (input[0]*w_x + input[1]*w_y)

# model = fn()
# inputs = np.array([1., 2.], dtype=np.float32)
# inputs = torch.from_numpy(inputs)
# params = []
# for param in model.parameters():
#     params.append(param)
# out = model(inputs)
# print(params)
# print(out.data.numpy())
# K = 4 # 5 degree of polynomial
M = 50 # 3 number of criteria
#####################################
# Using real data
# X, Y = process_data.load_data('C:/Data/Rank_uni/qs_ranking.csv', degree= 1)
# K = int(X.shape[1] / M)
#####################################
# Using simulation data
X = X
K = int(X.shape[1] / M)
# print(K)
Y = Y
#####################################  Training data and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 10)
train_size = X_train.shape[0]
num_criteria = M
mean = np.mean(X_train, axis=0)
mean = torch.from_numpy(mean)
mean = mean.type(torch.FloatTensor)

X_train = torch.from_numpy(X_train).reshape((X_train.shape[0], K*M, 1))
X_train = X_train.type(torch.FloatTensor)
Y_train = torch.from_numpy(Y_train).reshape((Y_train.shape[0], 1))
Y_train = Y_train.type(torch.FloatTensor)

# a = np.array([[1,2,3,4,5,6,7,8],
#               [2,3,1,4,1,5,1,1],
#               [3,3,3,3,1,2,3,4]])
# a = torch.from_numpy(a).reshape((3,8,1))
# a = a.type(torch.FloatTensor)
# y = np.array([[0],
#               [1],
#               [1]])
# y = torch.from_numpy(y)
# y = y.type(torch.FloatTensor)

net = GMZ(K*M, M)
print(net)
# 优化网络
optimizer = torch.optim.SGD(net.parameters(),lr=0.01)  # 优化的是参数 net.parameter
# optimizer = torch.optim.Adam(net.parameters(),lr=0.2,betas=(0.9, 0.999) )
loss_func = torch.nn.MSELoss()

#################
# Plot the relation between alpha and Loss, set flag_plot = True
#################
flag_plot = True
if flag_plot == True:
    record_weight = []
    record_loss = []
    iteration = []
    for t in range(300):  # 训练的步数
        prediction, weight, normalized_weights = net(X_train, mean, K, train_size,
                                                     num_criteria)  # input x and predict based on x
        loss = loss_func(prediction, Y_train)  # 预测值在前 真实值在后 # must be (1. nn output, 2. target)
        optimizer.zero_grad()  # 参数梯度降为0
        loss.backward()  # 反向传递
        optimizer.step()  # 优化梯度

        if t % 20 == 0:
            # iteration.append(t)
            # plot and show learning process
            print('********************')
            print('Loss is: ')
            print(loss)
            record_loss.append(loss)
            print('*********************')
            print('weight is: ')
            print(weight)
            record_weight.append(weight)
            print('*********************')
            print('Criteria weight is: ')
            print(normalized_weights)
    torch.save(net, 'net_params.pkl')
    plt.plot(record_loss,record_weight)
    # plt.plot(iteration, record_loss, label=r'Loss')
    plt.xlabel(r'MSE')
    plt.ylabel(r'$\alpha$')
    # plt.legend(loc='best')
    plt.show()


if flag_plot == False:
    for t in range(500):  # 训练的步数
        prediction, weight, normalized_weights = net(X_train, mean, K, train_size,
                                                     num_criteria)  # input x and predict based on x
        loss = loss_func(prediction, Y_train)  # 预测值在前 真实值在后 # must be (1. nn output, 2. target)
        optimizer.zero_grad()  # 参数梯度降为0
        loss.backward()  # 反向传递
        optimizer.step()  # 优化梯度

        if t % 20 == 0:
            # plot and show learning process
            print('********************')
            print('Loss is: ')
            print(loss)
            print('*********************')
            print('weight is: ')
            print(weight)
            print('*********************')
            print('Criteria weight is: ')
            print(normalized_weights)
    torch.save(net, 'net_params.pkl')


params = []
for param in net.parameters():
    params.append(param.detach().numpy())
# print(np.array(params))
parameters = []
for item in np.array(params):
    item = np.ravel(item)
    parameters.append(item)
# parameters = np.array(parameters)
# print(parameters)
# np.savez('trained_para.csv', parameters, delimiter=',' )
f = open('parameters_trained.csv','w')
for item in parameters:
    for num in item:
        f.write(str(num) + ',')
    f.write('\r\n')
f.close()
# m = 4
# n = 2
# a = np.array([1,2,3,4,5,6,7,8])
# a = torch.from_numpy(a).reshape((1,8,1))
# a = a.type(torch.FloatTensor)
# #a = torch.randn((1,m*n,1))
# #input1:m, input2:k
# model = GMZ(m*n,n)
# params = []
# for param in model.parameters():
#     params.append(param)
# output1,output2,o3_1,o3_2,o3,o4 = model(a)
# print(params)


###############Testing
# prediction, weight, normalized_weights = net(X_train, mean, K, train_size, num_criteria)
net2 = torch.load('net_params.pkl')
test_size = X_test.shape[0]
num_criteria = M
# mean = np.mean(X_test, axis=0)
# mean = torch.from_numpy(mean)
# mean = mean.type(torch.FloatTensor)

X_test = torch.from_numpy(X_test).reshape((X_test.shape[0], K*M, 1))
X_test = X_test.type(torch.FloatTensor)
Y_test = torch.from_numpy(Y_test).reshape((Y_test.shape[0], 1))
Y_test = Y_test.type(torch.FloatTensor)
prediction, weight, normalized_weights  = net2(X_test, mean, K, test_size, num_criteria)
loss_func = torch.nn.MSELoss()
loss_test = loss_func(prediction, Y_test)
print(loss_test,weight, normalized_weights )
