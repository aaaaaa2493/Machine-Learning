import scipy.optimize
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from autograd import grad
import autograd.numpy as np
from datetime import datetime
from threading import Thread
from typing import List

from sklearn import datasets, metrics

start = datetime.now()

old_print = print
file_output = open(f'{str(datetime.now()).replace(":", "_")[:-7]}.txt', 'w')
np.random.seed(0)
index_global = 3


def print(*args, **kwargs):
    old_print(*args, **kwargs)
    old_print(*args, **kwargs, file=file_output)


n: int = 10  # Количество классов (0-9)
m: int = 784 + 1  # Количество признаков (28х28 картинка)

lambdas = [0] + [10**i for i in range(-5, 10)]
lam: float = lambdas[3]  # Параметр регуляризации

NT: int = 70000  # Общая выборка
cross_validation_parameter: int = 4
N: int = 400  # Обучающая выборка
T: int = NT - N  # Тестовая выборка

#Y: List[NT]  # Список ответов на общую выборку

#X: List[m, N]  # Список обучающих выборок в виде столбцов

#A: List[NT]  # 1 / sum( e ** ( np.dot( w[k].T , x[i])) for i in range(k))
#ksi_start: List[NT]  # [i] = -ln A[i]

#W: List[m, n]

IDs = {}

# 69.463454463


def F(W, X, Y, ksi, N):

    S = lam * np.linalg.norm(W) ** 2 / 2

    QQ = np.sum(np.exp(np.dot(X, W) - np.tile(ksi, (1, n))), axis=1)

    for i in range(N):
        S += ksi[i] - 1 - np.dot(W.T[Y[i]], X[i].T) + QQ[i]

    return S[0]


F1 = lambda w: lam * np.linalg.norm(w) ** 2 / 2 + \
               np.sum(np.exp(np.dot(X, w).reshape((N, 1)) - ksi)) + \
               np.sum(-np.dot(XXi, w).reshape((XXsize, 1)) + XXksi - XXones)


W_INITIAL = np.random.rand(m, n) * 0.001

mnist = fetch_mldata('MNIST original', data_home='./data')


def normalize_features(train, test):
    """Normalizes train set features to a standard normal distribution
    (zero mean and unit variance). The same procedure is then applied
    to the test set features.
    """
    train_mean = train.mean(axis=0)
    # +0.1 to avoid division by zero in this specific case
    train_std = train.std(axis=0) + 0.1

    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, test


answers = mnist.target[:, np.newaxis]

data_and_answers = np.hstack((np.ones_like(answers), mnist.data[:, :], answers))
np.random.shuffle(data_and_answers)

train_X = data_and_answers[:N, :-1]
train_y = data_and_answers[:N, -1].reshape((N, 1))


test_X = data_and_answers[N:, :-1]
test_y = data_and_answers[N:, -1].reshape((T, 1))

train_X, test_X = normalize_features(train_X, test_X)

calc_ksi = lambda W, X, N: (-np.log(np.power(np.sum(np.exp(np.dot(X, W)), axis=1), -1))).reshape((N, 1))

X_INITIAL = train_X
Y_INITIAL = train_y
Y_INITIAL = np.vectorize(lambda x: int(x))(Y_INITIAL)


def count_indicies(Y):

    ret = {}

    for index, one_answer in enumerate(Y):
        one_answer = int(one_answer[0])

        if one_answer in ret:
            ret[one_answer] += [index]
        else:
            ret[one_answer] = [index]

    return ret


def start_thread(w, index_global, XXi, XXksi, XXsize, XXones, N):

    w_local = w
    index_local = index_global
    Xi = XXi
    Xksi = XXksi
    Xsize = XXsize
    Xones = XXones
    X_local = X
    N_local = N
    m_local = m
    ksi_local = ksi

    F1_local = lambda w: lam * np.linalg.norm(w) ** 2 / 2 + \
                         np.sum(np.exp(np.dot(X_local, w).reshape((N_local, 1)) - ksi_local)) + \
                         np.sum(-np.dot(Xi, w).reshape((Xsize, 1)) + Xksi - Xones)

    _w = scipy.optimize.minimize(
        fun=F1_local,
        x0=w_local,
        args=(),
        method='L-BFGS-B',
        jac=grad(F1_local),
        options={'maxiter': 100, 'disp': False},
        callback=lambda x: ...,
    )

    w_local = _w.x.reshape((m_local, 1))

    Wt[index_local] = w_local


# сходимость градиента к нулю,

for curr_lambda in lambdas:

    print(f'CALCULATING FOR LAMBDA = {curr_lambda}')

    W = W_INITIAL
    Wbest = None
    Fbest = None

    lam = curr_lambda

    X = X_INITIAL[:300, :]
    Y = Y_INITIAL[:300]
    X_TEST = X_INITIAL[300:, :]
    Y_TEST = Y_INITIAL[300:]

    N = 300

    X, X_TEST = normalize_features(X, X_TEST)

    ksi_start = calc_ksi(W, X, N)
    ksi_test = calc_ksi(W, X_TEST, 100)

    ksi = ksi_start

    for j in range(15):

        Wt = [0] * n
        all_threads = []

        for i in range(n):

            index_global = i

            wi = W.T[index_global].T

            IDs = count_indicies(Y)

            XXindeces = IDs.get(index_global, [])

            XXi = np.array([X[i] for i in XXindeces])
            XXksi = np.array([ksi[i] for i in XXindeces])
            XXsize = len(XXindeces)
            XXones = np.ones_like(XXksi)

            thread = Thread(target=lambda a=wi,
                                          b=index_global,
                                          c=XXi,
                                          d=XXksi,
                                          e=XXsize,
                                          f=XXones,
                                          g=N:  start_thread(a, b, c, d, e, f, g))

            all_threads += [thread]
            thread.start()

        for th in all_threads:
            th.join()

        Wt = np.array(Wt).T.reshape((m, n))

        Fcurr = F(Wt, X_TEST, Y_TEST, ksi_test, 100)

        if Fbest is None or Fcurr < Fbest:
            Wbest = Wt
            Fbest = Fcurr

        #print(Wt.shape)
        #print(np.sum(Wt))  # 3.86054716045

        print(f'{j+1}) F(W) = {F(Wt, X, Y, ksi, N)} F(test) = {Fcurr}')

        #print(W.shape)
        #print(Wt.shape)

        ksi = calc_ksi(Wt, X, N)
        ksi_test = calc_ksi(Wt, X_TEST, 100)
        W = Wt

    count_success = 0

    for i in range(T):
        result = np.dot(Wbest.T, test_X[i])
        if np.argmax(result) == test_y[i]:
            count_success += 1

    print('Success count:', count_success / T)
    print()

    for i in Wbest:
        for j in i:
            print(j, end=' ')

        print()

    break

end = datetime.now()
print(end - start)

quit(0)



mnist = fetch_mldata('MNIST original', data_home='./data')

y_all = mnist.target[:, np.newaxis]

print(y_all[28508])


for i in range(28*28):
    print(f'{mnist.data[28508, i]:>3}', end=' ')
    if (i+1)%28 == 0:
        print()



#quit(0)

intercept = np.ones_like(y_all)


def normalize_features(train, test):
    """Normalizes train set features to a standard normal distribution
    (zero mean and unit variance). The same procedure is then applied
    to the test set features.
    """
    train_mean = train.mean(axis=0)
    # +0.1 to avoid division by zero in this specific case
    train_std = train.std(axis=0) + 0.1

    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, test

train_normalized, test_normalized = normalize_features(
    mnist.data[:60000, :],
    mnist.data[60000:, :],
)

train_all = np.hstack((
    intercept[:60000],
    train_normalized,
    y_all[:60000],
))

test_all = np.hstack((
    intercept[60000:],
    test_normalized,
    y_all[60000:],
))

np.random.shuffle(train_all)
np.random.shuffle(test_all)

train_X = train_all[:, :-1]
print(train_X[0])
print(len(train_X[0]))
train_y = train_all[:, -1]
print(len(train_y))

test_X = test_all[:, :-1]
print(test_X[0])
print(len(test_X[0]))
test_y = test_all[:, -1]
print(len(test_y))
quit(0)

m, n = train_X.shape
k = np.unique(train_y).size
theta = np.random.rand(n, k) * 0.001

indicator_mask = np.zeros((train_X.shape[0], theta.shape[1]), dtype=np.bool)
for i, idx in enumerate(train_y):
    indicator_mask[i][int(idx)] = True


def probs(theta, X, y):
    if theta.ndim == 1:
        theta = theta.reshape((theta.size // k, k))
    values = np.exp(X.dot(theta))
    sums = np.sum(values, axis=1)
    return (values.T / sums).T


def cost_function(theta, X, y):
    log_probs = np.log(probs(theta, X, y))
    cost = 0
    for i in range(m):
         cost -= log_probs[i][int(y[i])]
    return cost


def gradient(theta, X, y):
    gradient_matrix = -X.T.dot(indicator_mask - probs(theta, X, y))
    return gradient_matrix.flatten()


J_history = []

t0 = time.time()
res = scipy.optimize.minimize(
    fun=cost_function,
    x0=theta,
    args=(train_X, train_y),
    method='L-BFGS-B',
    jac=gradient,
    options={'maxiter': 100, 'disp': True},
    callback=lambda x: J_history.append(cost_function(x, train_X, train_y)),
)
t1 = time.time()

print('Optimization took {s} seconds'.format(s=t1 - t0))
optimal_theta = res.x.reshape((theta.size // k, k))

plt.plot(J_history, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')

plt.savefig('plot.png')

def accuracy(theta, X, y):
    correct = np.sum(np.argmax(probs(theta, X, y), axis=1) == y)
    return correct / y.size

print('Training accuracy: {acc}'.format(acc=accuracy(res.x, train_X, train_y)))
print('Test accuracy: {acc}'.format(acc=accuracy(res.x, test_X, test_y)))
