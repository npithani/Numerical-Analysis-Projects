import math
from math import factorial
from math import sqrt
from math import pow
from math import pi
from math import e
import numpy as np
import scipy
import scipy.linalg   # SciPy Linear Algebra Library

def condition_numbers(matrixN):
    A_1 = scipy.array(matrixN)
    P_1, L_1, U_1 = scipy.linalg.lu(A_1)


    A1 = np.array(matrixN)
    L = np.array(L_1)
    U = np.array(U_1)
    LT = L.transpose()
    UT = U.transpose()


    #c0 = 1, c0 = -1
    v00 = 1/UT.item(0)
    v01 = -1/UT.item(0)

    #c0 = 1, c1 = 1
    v11 = (1 + (-1 * (v00 * UT.item(3)))) / (UT.item(4))

    #c0 = 1, c1 = -1
    v12 = (-1 + (-1 * (v00 * UT.item(3)))) / (UT.item(4))

    #c0 = -1, c1 = 1
    v13 = (1 + (-1 * (v01 * UT.item(3)))) / (UT.item(4))

    #c0 = -1, c1 = -1
    v14 = (-1 + (-1 * (v01 * UT.item(3)))) / (UT.item(4))

    #c0 = 1, c1 = 1, c2 = 1
    v21 = (1 + (-1 * (v11 * UT.item(7))) + (-1 * (v00 * UT.item(6)))) / (UT.item(8))

    #c0 = 1, c1 = 1, c2 = -1
    v22 = (-1 + (-1 * (v11 * UT.item(7))) + (-1 * (v00 * UT.item(6)))) / (UT.item(8))

    #c0 = 1, c1 = -1, c2 = 1
    v23 = (1 + (-1 * (v12 * UT.item(7))) + (-1 * (v00 * UT.item(6)))) / (UT.item(8))

    #c0 = 1, c1 = -1, c2 = -1
    v24 = (-1 + (-1 * (v12 * UT.item(7))) + (-1 * (v00 * UT.item(6)))) / (UT.item(8))

    #c0 = -1, c1 = 1, c2 = 1
    v25 = (1 + (-1 * (v13 * UT.item(7))) + (-1 * (v01 * UT.item(6)))) / (UT.item(8))


    #c0 = -1, c1 = 1, c2 = -1
    v26 = (-1 + (-1 * (v13 * UT.item(7))) + (-1 * (v01 * UT.item(6)))) / (UT.item(8))


    #c0 = -1, c1 = -1, c2 = 1
    v27 = (1 + (-1 * (v14 * UT.item(7))) + (-1 * (v01 * UT.item(6)))) / (UT.item(8))


    #c0 = -1, c1 = -1, c2 = -1
    v28 = (-1 + (-1 * (v14 * UT.item(7))) + (-1 * (v01 * UT.item(6)))) / (UT.item(8))


    #cs are 1, 1, 1
    v1 = [v00, v11, v21]


    #cs are 1, 1, -1
    v2 = [v00, v11, v22]

    #cs are 1, -1, 1
    v3 = [v00, v12, v23]

    #cs are 1, -1, -1
    v4 = [v00, v12, v24]

    #cs are -1, 1, 1
    v5 = [v01, v13, v25]

    #cs are -1, 1, -1
    v6 = [v01, v13, v26]

    #cs are -1, -1, 1
    v7 = [v01, v14, v27]

    #cs are -1, -1, -1
    v8 = [v01, v14, v28]

    vector_array = [v1, v2, v3, v4, v5, v6, v7, v8]
    array0 = [v00, v01]
    array1 = [v11, v12, v13, v14]
    array2 = [v21, v22, v23, v24, v25, v26, v27, v28]
    # print array2

    chosen0 = max(array0)
    #print chosen0

    chosen1 = max(array1)
    #print chosen1

    chosen2 = max(array2)
    #print chosen2

    v = [chosen0, chosen1, chosen2]
    # print "v: "
    # print v

    yA = np.linalg.solve(LT,v)
    # print "yA: "
    # print yA

    zA = np.linalg.solve(A1, yA)
    # print "zA: "
    # print zA

    def matrix_norm(thematrix):
        col1_sum = abs(A1.item(0)) + abs(A1.item(3)) + abs(A1.item(6))

        col2_sum = abs(A1.item(1)) + abs(A1.item(4)) + abs(A1.item(7))

        col3_sum = abs(A1.item(2)) + abs(A1.item(5)) + abs(A1.item(8))

        sum1 = max(col1_sum, col2_sum, col3_sum)

        return sum1

    z1 = np.linalg.norm(zA, 1)


    y1 = np.linalg.norm(yA, 1)


    inverse_norm1 = z1/y1


    inh = np.linalg.norm((zA/yA), 1)


    norm_1 = matrix_norm(A1)


    conditioNumber1 = norm_1 * inh
    print "condition number from part a: "
    print np.float64(conditioNumber1)

    y1 = np.random.rand(3,1)

    y2 = np.random.rand(3,1)

    y3 = np.random.rand(3,1)

    y4 = np.random.rand(3,1)

    y5 = np.random.rand(3,1)

    y_array = [y1, y2, y3, y4, y5]

    def chooseValue(an_array):
        zB = np.linalg.solve(A1, y1)
        greatest = np.linalg.norm(zB, 1) / np.linalg.norm(y1, 1)
        for x in an_array:
            value = (np.linalg.norm(np.linalg.solve(A1, x), 1)) / (np.linalg.norm(x, 1))
            if value > greatest:
                greatest = value

        return greatest

    inverse_norm2 = chooseValue(y_array)


    norm_2 = matrix_norm(A1)


    conditioNumber2 = norm_2 * inverse_norm2
    print "condition number from part b: "
    print np.float64(conditioNumber2)

    conditioNumber3 = np.linalg.cond(A1,1)
    print "condition number using the built-in numpy function: "
    print np.float64(conditioNumber3)


def main():
    print "Matrix A1: "
    firstMatrix = [[-10, 7, 0], [5, -1, 5], [-3, 2, 6], ]
    condition_numbers(firstMatrix)
    print "----------------------------------------------------"
    print "Matrix A2: "
    secondMatrix = [[92, 66, 25], [-73, 78, 24], [-80, 37, 10]]
    condition_numbers(secondMatrix)


main()