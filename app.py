#!/usr/bin/python

try:
    import numpy
except:
    print("This implementation requires the numpy module.")
    exit(0)


def matrix_factorization(A, X, Y, K, steps=5000, alpha=0.0002, beta=0.02):
    Y = Y.T

    for step in range(steps):

        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i][j] > 0:
                    eij = A[i][j] - numpy.dot(X[i, :], Y[:, j])
                    for k in range(K):
                        X[i][k] = X[i][k] + alpha * (2 * eij * Y[k][j] - beta * X[i][k])
                        Y[k][j] = Y[k][j] + alpha * (2 * eij * X[i][k] - beta * Y[k][j])

        eR = numpy.dot(X, Y)
        e = 0
        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i][j] > 0:
                    e = e + pow(A[i][j] - numpy.dot(X[i, :], Y[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(X[i][k], 2) + pow(Y[k][j], 2))

        if e < 0.001:
            break

    return X, Y.T


###############################################################################

if __name__ == "__main__":

# use this for the low-rank matrix example from the lecture
    R = [
         [1,2,3,5],
         [2,4,8,12],
         [3,6,7,13],
        ]

# use this for the high-rank matrix example from the lecture
    #R = [
    #      [5,8,0,0],
    #      [0,0,9,8],
    #      [6,0,0,0],
    #    ]

    A = numpy.array(R)

    print("\nA = ")
    print(A)

print("\nA = ")
print(A)

n = len(A)
m = len(A[0])

k = 2
beta = 0.02

print("\nk =", k)
print("beta =", beta)

X_rand = numpy.random.rand(n, k)
Y_rand = numpy.random.rand(m, k)

print("\nX_rand = ")
print(X_rand)

print("\nY_rand = ")
print(Y_rand)

X, Y = matrix_factorization(A, X_rand, Y_rand, k, beta=0.02)

print("\nX = ")
print(X)

print("\nY^T = ")
print(numpy.transpose(Y))

print("\nX * Y^T = ")
print(numpy.dot(X, numpy.transpose(Y)))

print("\nA - X * Y^T = ")
print(A - numpy.dot(X, numpy.transpose(Y)))

print("\n|A - X * Y^T|_2 = ")
print(numpy.linalg.norm(A - numpy.dot(X, numpy.transpose(Y)), ord=2))
