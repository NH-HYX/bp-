import numpy as np
from diagram import Diagram
from data import Dataset


class BPNN(object):
    """输入层：4个节点，用Iris的四个特征值，为一个150*3的矩阵
    权重：（4+1）*3 的矩阵+一个阈值bias
       隐藏层：3个，设定为150*3的矩阵
    权重：（3+1）*3 的矩阵+一个阈值bias
       输出层：3个节点，使用Iris的输出为三个种类，为一个150*4的矩阵
    """

    # bpn = BPNN(dataset, learning_rate=0.01, n_iter=20000) 均方差
    def __init__(self, dataset, learning_rate=0.01, n_iter=10000, momentum=0.9, shutdown_condition=0.01):
        self.n_iter = n_iter
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.x = dataset.train_x
        self.Y = dataset.train_Y
        self.shutdown_condition = shutdown_condition
        self.cost = []
        self.momentum = momentum
        self.setup()
        self.diagram = Diagram(self)

    def setup(self):
        self.set_nn_architecture()
        self.set_weight()

    # step1
    def set_nn_architecture(self):
        self.input_node = self.x.shape[1]  # 4
        self.output_node = self.Y.shape[1]  # 3 Y = [1,0,0]
        self.hidden_node = int((self.input_node + self.output_node) / 2)
        # self.hidden_node = int(np.sqrt(input_node+output_node) + 5)

        # bias
        self.h_b = np.random.random(self.hidden_node) * 0.3 + 0.1  # return hidden_node 个元素的list
        self.y_b = np.random.random(self.output_node) * 0.3 + 0.1

    # step2 权值初始化
    def set_weight(self):
        self.w1 = np.random.random((self.input_node, self.hidden_node))
        self.w2 = np.random.random((self.hidden_node, self.output_node))

    # step3 正向过程：对加权和阈值后的输出进行激活函数 返回self.h,self.y,self.acc
    def predict(self, x, Y):
        self.h = self.sigmoid((np.dot(x, self.w1) + self.h_b))  # dot means 两个矩阵点积
        self.y = self.sigmoid((np.dot(self.h, self.w2) + self.y_b))
        self.zy = np.where(self.y > 0.5, 1, 0)  # 根据条件返回1或0。zy是预测输出
        p_y = Y - self.zy  # p_y = [1,0,0]
        self.acc = 0
        for i in p_y:
            if (i.sum() == 0):
                self.acc += 1
        self.acc = self.acc / Y.shape[0] * 100.0  # return this acc 均方差
        return self

    # step4
    def backend(self):
        E = (self.Y - self.y)
        errors = np.sum(np.square(E)) / self.Y.shape[1] / self.Y.shape[0]
        #### 輸出層 delta 計算
        delta_y = E * self.y * (1 - self.y)
        ### 隱藏層 delta 計算
        delta_h = (1 - self.h) * self.h * np.dot(delta_y, self.w2.T)
        # self.w2 += self.learning_rate * self.h.T.dot(delta_y) + self.momentum * self.h.T.dot(delta_y)
        # self.w1 += self.learning_rate * self.x.T.dot(delta_h) + self.momentum * self.x.T.dot(delta_h)
        self.w2 += self.learning_rate * self.h.T.dot(delta_y)
        self.w1 += self.learning_rate * self.x.T.dot(delta_h)
        self.y_b = self.learning_rate * delta_y.sum()
        self.h_b = self.learning_rate * delta_h.sum()
        return errors

    def train(self):
        self.error = 0

        for _iter in range(0, self.n_iter):  # n_iter = 20000
            self.predict(self.x, self.Y)
            self.error = self.backend()
            self.cost.append(self.error)  # cost的数目是由backend()函数决定
            if (self.acc >= 98):  # abort train
                # 这里权值训练好了，不再改变
                print("111111111111111111")
                print(self.acc)
                return self
        print(self.acc)
        return self

    def test(self):
        self.predict(self.dataset.test_x, self.dataset.test_Y)
        # output = []
        # for i in self.zy:
        #     out = self.output_transform(i)
        #     output.append(out)
        # print(output)
        # print("\n")
        return self

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def draw(self, xlabel='', ylabel='', legend_loc='', title=''):
        self.diagram.draw(xlabel, ylabel, legend_loc, title)
    # def output_transform(self, i):
    #     if i[0] == 1:
    #         return 'Iris-setosa'
    #     elif i[1] == 1:
    #         return 'Iris-versicolor'
    #     elif i[2] == 1:
    #         return 'Iris-virginica'
