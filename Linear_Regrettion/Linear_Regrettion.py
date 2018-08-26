import matplotlib.pyplot as plt
import numpy as np
import math


class Linear_regrettion:
    
    def __init__(self, M, ramda):
        self.M = M
        self.ramda = ramda
        self.A = np.eye(M)
        self.m = np.zeros(M)
        
    def generate_sin_rand(self, rand_num):
        PLOT = rand_num
        sin_x = []
        sin_y = []
        for i in range(PLOT):
            rand = np.random.rand() * 3.14 * 3
            tmp = math.sin(rand)
            sin_x.append(rand)
            sin_y.append(tmp)
        return sin_x, sin_y
    
    def renew_A(self, xn_list, A):
        N = len(xn_list)
        for i in range(N):
            tmp = xn_list[i].reshape(-1, 1)
            A += self.ramda * np.dot(tmp, tmp.T)
        return A

    def make_input_x(self, sin_x, M):
        xn_list = []
        for i in sin_x:
            tmp = []
            for j in range(M):
                tmp.append(i ** j)
            xn_list.append(tmp)
        xn_ = np.array(xn_list)
        return xn_
    
        
    def plot_test_result(self, test_x, test_y):
        x_range = np.arange(0, 3.14 * 3, 0.1)
        plot_x_ = self.make_input_x(x_range, self.M)

        y_test = []
        y_sigma_up = []
        y_sigma_down = []
        for i in range(len(plot_x_)):
            mu_ = np.dot(self.m_, plot_x_[i])
            inv_A_ = np.linalg.inv(self.A_)
            sigma_ = 1 / self.ramda + np.dot(np.dot(plot_x_[i], inv_A_), plot_x_[i])
            y_test.append(mu_)
            y_sigma_up.append(mu_ + math.sqrt(sigma_))
            y_sigma_down.append(mu_ - math.sqrt(sigma_))

        plt.scatter(test_x, test_y)
        plt.plot(x_range, y_test)
        plt.plot(x_range, y_sigma_up, linestyle="dashdot", c="c", alpha=0.5)
        plt.plot(x_range, y_sigma_down, linestyle="dashdot", c="c", alpha=0.5)
        plt.grid(True)
        
        
    def plot_result(self):
        x_range = np.arange(0, 3.14 * 3, 0.1)
        plot_x_ = self.make_input_x(x_range, self.M)
        self.y = []
        for i in range(len(plot_x_)):
            self.y.append(np.dot(self.p_w, plot_x_[i]))
        plt.plot(x_range, self.y)
        plt.scatter(self.data_x, self.data_y)
        plt.grid(True)
    
    def fit(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        plot_x = self.make_input_x(self.data_x, self.M)
        self.A_ = self.renew_A(plot_x, np.eye(self.M))
        N = len(plot_x)
        tmp = np.dot(self.A, self.m)
        for i in range(N):
            tmp += self.ramda * self.data_y[i] * plot_x[i]
        self.m_ = np.dot(np.linalg.inv(self.A_), tmp)
        self.p_w = np.random.multivariate_normal(self.m_, np.linalg.inv(self.A_))

def plot_all_result():
    l = Linear_regrettion(1, 10)
    data_x, data_y = l.generate_sin_rand(50)
    test_x, test_y = l.generate_sin_rand(10)
    M_list = [1, 2, 3, 4, 5, 10]
    for i, m in enumerate(M_list):
        lr = Linear_regrettion(m, 10)
        lr.fit(data_x, data_y)
        plt.subplot(2, 3, i+1)
        lr.plot_test_result(test_x, test_y)
        plt.ylim([-4, 4])
        plt.text(0.1, 3, "M = {}".format(m))
    plt.show()   


if __name__ == "__main__":
    plot_all_result()
