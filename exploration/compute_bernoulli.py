from math import pi
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import bernoulli, factorial

def di_dx_f (i, m):
    """i'th derivative of 1/x^2 evaluated x=m"""
    return (-1)**i * factorial(i+1) / m**(2+i)


def euler_maclaurin(N, l):
    """Precomputes the first N terms of $\sum_1^\infty 1/k^2$, and approximates 
    the tail with the Euler-Maclaurin formula up to order l"""
    ssum = 0
    for i in range(1, N+1):
        ssum += 1/i**2
    

    ssum += (1/(N+1)**2) / 2 # f(m) / 2
    ssum += 1/(N+1) # integral_{m}^\infty 1/x^2 dx

    bernoullis = bernoulli(2*l)
    for k in range(1, l+1):
        ssum -= bernoullis[2*k] / factorial(2*k) * di_dx_f(2*k - 1, N+1)
    return ssum


def compare():
    """Compares the naive summation with 10 terms to the Euler-Maclaurin formula 
    with 5 terms of normal summation and 5 terms of the Euler-Maclaurin formula"""
    # Normal summation with 10 terms
    ssum = 0
    for i in range(1, 11):
        ssum += 1/i**2
    print("Normal sum (10 terms):", ssum)
    print("Absolute error:", abs(pi**2/6 - ssum))

    # 5 terms of normal summation and 5 terms of euler-maclaurin
    lb = euler_maclaurin(5, 4)
    ub = euler_maclaurin(5, 5)
    print("Lower bound:", lb)
    print("Upper bound:", ub)
    print("Upper bound - Lower bound:", ub - lb)
    print("(True) Absolute error:", abs(pi**2/6 - ub))

    print("True value:", pi**2/6)


def create_table():
    """Creates a LaTeX table showing the error term for the Euler-Maclaurin,
    for N=0 and N=5 terms of normal summation."""
    l = [i for i in range(1, 11)]
    N1 = 0
    estimates_1 = [euler_maclaurin(N1, i) for i in l]
    N2 = 5
    estimates_2 = [euler_maclaurin(N2, i) for i in l]

    # print to latex table
    print("\\begin{tabular}{|c|c|c|}")
    print("\\hline")
    print("l & N = 0 & N = 5 \\\\")
    print("\\hline")
    for i in range(len(l)):
        print(f"{l[i]} & {pi**2/6 - estimates_1[i]:.2E} & {pi**2/6 - estimates_2[i]:.2E} \\\\")
    print("\\hline")
    print("\\end{tabular}")


print("Comparison of naive summation and Euler-Maclaurin:")
compare()
print()
print("LaTeX table showing error term:")
create_table()

