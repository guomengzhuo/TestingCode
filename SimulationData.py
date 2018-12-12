import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

def generate_attribute_value(data_size, dimensional):  # data_size 是样本大小 dimensional 是有几个属性
    Data = np.multiply(np.random.random((data_size, dimensional)), np.random.uniform(0, 10, (1, dimensional)))
    # generate orginal data Data = [num of data size * dimensional]
    # print(np.random.randint(0,10,(1, dimensional)))
    # print(np.mean(Data,axis=0))
    # print(Data)
    return Data
# data_size = 1000
# dimensional = 5
# np.random.seed(1)
# Data is attribute values for sampled data

def marginalvalue(Data, degree):
    para = np.random.uniform(-1, 1, Data.shape[1] * degree) # 生成多项式函数的参数 一共有 dimension * degree 个
    New_Data = []
    for i in range(Data.shape[1]):  # i is from 0 to dimensions
        for j in range(1, degree + 1):    # 用低微数据拟合高维多项式  for j in range(1, degree - 2): degree should be greater than 3
            New_Data.append(np.array(Data[:, i] ** j))
    New_Data = (np.array(New_Data).T) # 按照degree取幂
    # print(Data[:,0] ** 3 == New_Data[:,2]) is true
    # Data[:,0] ** 2 == New_Data[:,1] is true
    # Data[:,1] ** 1 == New_Data[:,3]
    global_value = np.dot(New_Data, para)
    return New_Data, global_value.reshape(New_Data.shape[0], 1), para

def marginalvalue_with_nosie(Data, degree):
    para = np.random.uniform(-1, 1, Data.shape[1] * degree) # 生成多项式函数的参数 一共有 dimension * degree 个
    New_Data = []
    for i in range(Data.shape[1]):  # i is from 0 to dimensions
        for j in range(1, degree + 1): # 用低微数据拟合高维多项式  for j in range(1, degree - 2): degree should be greater than 3
            # New_Data.append(np.array(np.sin(Data[:, i]) ** j))
            New_Data.append(np.array(Data[:, i] ** j) )
                            # + np.random.randint(np.min(Data[:,i]), np.max(Data[:,i])) * np.sin(Data[:,i]))
    New_Data = (np.array(New_Data).T) # 按照degree取幂
    # print(Data[:,0] ** 3 == New_Data[:,2]) is true
    # Data[:,0] ** 2 == New_Data[:,1] is true
    # Data[:,1] ** 1 == New_Data[:,3]
    global_value = np.dot(New_Data, para)
    # global_value = global_value + np.random.randint(-100, 100, (global_value.shape))
    global_value = np.sin(global_value)
    # global_value = np.cos(global_value)+np.sin(global_value)
    # global_value = np.random.random(np.dot(New_Data, para).shape)
    return global_value.reshape(New_Data.shape[0], 1)

def interacted_marginal_values(Data):  #Data是最原始的数据
    New_Data = []
    for i in range(Data.shape[1] - 1):  # i is from 0 to dimensions
        for j in range(i+1, Data.shape[1]):
            New_Data.append(np.array(Data[:, i] * Data[:, j]))
    New_Data = (np.array(New_Data).T)  # 按照degree取幂
    para = np.random.uniform(-1, 1, New_Data.shape[1])
    global_value = np.dot(New_Data, para)
    return global_value.reshape(New_Data.shape[0], 1)


def plot_marginal_value_functions(para, degree):
    x = np.linspace(-1.0, 1.0, 100)
    b = [para[i:i + degree] for i in range(0, len(para), degree)]  # b 是每degree个为一组将para分开
    y = []
    for i in b:
        temp = []
        temp = 0
        for j in range(1, degree + 1):
            temp += i[j - 1] * (x ** j)
        y.append(temp)
    y = np.array(y)
    print(y.shape)
    for i in range(np.array(y).shape[0]):
        plt.plot(x, y[i, :])
    plt.show()

def compare_global_values(New_Data, global_value):
    # New_Data 是平方、三次方的项
    # global_value 是根据New_Data和para计算的全局效用 通过计算差值获得相对大小
    comp_global_value = []
    comp_New_Data = []
    for index1 in range(len(global_value)):  # index from 0 - 99
        for index2 in range(index1 + 1, len(global_value)):
            comp_New_Data.append(New_Data[index1, :] - New_Data[index2, :])
            if global_value[index1] - global_value[index2] >= 0:
                comp_global_value.append(1)
                # print('1')
            else:
                comp_global_value.append(0)
                # print('0')
            # print(global_value[index1], global_value[index2])
    comp_New_Data = np.array(comp_New_Data)
    comp_global_value = np.array(comp_global_value).reshape(comp_New_Data.shape[0], 1)
    New_result_matrix = np.hstack((comp_New_Data, comp_global_value))
    return New_result_matrix, comp_New_Data, comp_global_value

if __name__ == '__main__':
    K = 3
    Data = generate_attribute_value(data_size = 300, dimensional= 50)
    # np.savetxt('original_attribute.csv', Data, delimiter=',')
    New_Data, global_value, para = marginalvalue(Data=Data, degree= K)

    # np.savetxt('parameters.csv', para, delimiter = ',')
    # np.savetxt('polynomial_attribute.csv', np.hstack((New_Data, global_value)), delimiter=',')
    # np.savetxt('global_value.csv', global_value)
    # plot_marginal_value_functions(para= para, degree= K)
    ###############
    #加入噪音的global value
    global_value_with_nosie = marginalvalue_with_nosie(Data=Data, degree= K)  # 加入噪音影响
    global_value = global_value_with_nosie #加入噪音的global value
    ###############
    ###############
    # global_value = interacted_marginal_values(Data)   # 每两个属性交互产生新属性
    ###############
    New_result_matrix, X, Y = compare_global_values(New_Data, global_value) # X 是圣经网络的输入 Y是输出
    np.savetxt('comp_results.csv', New_result_matrix, delimiter=',')
    np.savetxt('X.csv', X, delimiter=',')
    np.savetxt('Y.csv', Y, delimiter=',')
    # print(Data, New_Data, global_value)