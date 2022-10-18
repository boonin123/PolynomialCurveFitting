import numpy as np
import csv
import argparse
#import random
import matplotlib.pyplot as plt

def readData(file):
    data = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            data.append(row)

    data = [[float(y) for y in x] for x in data]
    data = np.asarray(data)

    x = data[:, 0]
    y = data[:, 1]

    return x, y

def calcDesignMatrix(x, degree):
    N = len(x)
    X = np.c_[np.ones(N),x]
    for i in range(2,degree+1):
        X = np.c_[X, np.power(x,i)]
    return X

def RLS(x, y, gamma, degree):
    X = calcDesignMatrix(x, degree)
    w = (np.linalg.inv(gamma * np.identity(X.shape[1]) + X.T @ X) @ X.T) @ y
    yhat = np.dot(X, w)
    return yhat, w

def RMSE(a, p):
    diff = np.subtract(a, p)
    squared_differences = np.square(diff)
    return np.sqrt(squared_differences.mean())

def RSOS(a, p):
    diff = np.subtract(a, p)
    squared_differences = np.square(diff)
    return squared_differences.mean()

def variance(yhat, y, degree):
    sr = RSOS(y, yhat)
    return sr/(y.size-degree-1)

def const_normalize(list, scalar):
    max_value = max(list)
    min_value = min(list)
    cnorm_list = []
    for i in range(len(list)):
        cnorm_list.append(scalar*(list[i] - min_value) / (max_value - min_value))
    return cnorm_list

def autofit(x, y, gamma, maxdeg, folds = 10):
    #for random data subsetting
    #x, y = zip(*random.sample(list(zip(x, y)), 40))

    x = np.array(x)
    y = np.array(y)
    rmse = []
    var = []

    for i in range(1, maxdeg+1):
        yhat, w = RLS(x, y, gamma, i)
        rmse.append(RMSE(y, yhat))
        var.append(variance(yhat,y,i))

    #normalize error lists and mult by constant (heuristic)
    nrmse = const_normalize(rmse, 0.1)
    nvar = const_normalize(var, 1)

    errors = np.add(nrmse, nvar)

    best_degree = np.argmin(errors)+1
    best_weights = RLS(x, y, gamma, best_degree)[1]

    #plt.plot(range(1,len(var)+1),errors)
    #plt.show()

    print('Autofit order chosen: ' + str(best_degree))
    print('Weights found using RLS: ' + str(best_weights))
    print('-----------------------------------')
    print('RMSE: ' + str(round(rmse[best_degree-1],4)) + ', Variance: ' + str(round(var[best_degree-1],8)))
    print('Heuristic Error (minimized): ' + str(round(errors[best_degree-1],8)))

    return best_weights

def plot_fit(x, y, weights, path):
    p = np.poly1d(list(weights)[::-1])
    plt.plot(x, p(x), color='red', label='RLS Best Fit Line')
    plt.scatter(x,y)
    plt.title('Best Fit Line of Degree ' + str(weights.size-1)
              + ' Fit to Dataset ' + path[-1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def main():
    m = 1
    trainpath = ''
    output_path = ''

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--max', type=int, required=True)
    parser.add_argument('-g', '--gamma', type=float, required=False)
    parser.add_argument('-p', '--trainPath', type=str, required=True)
    parser.add_argument('-o', '--modelOutput', type=str, required=False)
    parser.add_argument('-s', '--showPlot', action='store_true')
    parser.add_argument('-a', '--autofit', action='store_true')
    parser.add_argument('-i', '--info', action='store_true')


    args = vars(parser.parse_args())

    if args['info']:
        print('Name: Noah Boonin')
        print('Student ID: 31570275')
        print('Email: nboonin@u.rochester.edu')
        print('------------------------------')

    m = args['max']
    gamma = args['gamma']
    if gamma is None:
        gamma = 0
    trainpath = args['trainPath']
    feature, output = readData(trainpath)

    if not args['autofit']:
        ypred, w = RLS(feature, output, gamma, m)
        print('Weights found for order ' + str(m) + ' polynomial: ' + str(w))
        print('RMSE = ' + str(round(RMSE(ypred, output), 4))
              + ', Variance: ' + str(round(variance(ypred, output, m),8)))
    else:
        w = autofit(feature, output, gamma, m)

    if args['showPlot']:
        plot_fit(feature, output, w, trainpath)

    if args['modelOutput'] is not None:
        output_path = args['modelOutput']
        np.savetxt(output_path + "modelFile", w, header="m = %d\ngamma = %f" % (w.size-1, gamma))

if __name__ == "__main__":
    main()