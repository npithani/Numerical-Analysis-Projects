#
# Neha Srisatya Pithani
# Fall 2017 Semester
# This program computes absolute and relative errors in Stirling’s approximation for
# n!≈ (2πn)^(1/2) * (n/e)^(n) for n = 1, 2, ... 10
#

import math
import numpy as np
from math import factorial
from math import sqrt
from math import pow
from math import pi
from math import e


def single_precision():
    print "==== SINGLE PRECISION ===="

    for x in range(1, 11):


        #right side of equation
        root = np.float32(math.sqrt((2*x*pi)))

        quotient = np.float32((float(x)/e))
        square = np.float32(pow(quotient, x))
        product = np.float32(root*square)

        #left side of equation
        fact = np.float32(factorial(x))

        #absolute error = | exact - approximate |
        abs_error = np.float32(abs(fact - product))
        print  "absolute error: ", abs_error

        #relative error = absolute error / exact
        rel_error = np.float32(abs_error / fact)
        print "relative error: ", rel_error


def double_precision():
    print "==== DOUBLE PRECISION===="
    for x in range(1, 11):


        #right side of equation
        root = np.float64(math.sqrt((2*x*pi)))

        quotient = np.float64((float(x)/e))
        square = np.float64(pow(quotient, x))
        product = np.float64(root*square)

        #left side of equation
        fact = np.float64(factorial(x))

        #absolute error = | exact - approximate |
        abs_error = np.float64(abs(fact - product))
        print  "absolute error: ", abs_error

        #relative error = absolute error / exact
        rel_error = np.float64(abs_error / fact)
        print "relative error: ", rel_error




def main():
    single_precision()
    double_precision()



main()
