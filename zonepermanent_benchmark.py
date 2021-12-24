import numpy as np
from piquassoboost.sampling.Boson_Sampling_Utilities import ZOnePermanent
def dosign(parity, x): return -x if parity else x
def plusminus(parity, base, x): return base - x if parity else base + x
def prod(x, y): return x * y
def multiprod(l):
  import functools
  return functools.reduce(prod, l)
def permanent(mat):
  import itertools
  n = len(mat)
  if n == 0: return 1
  return sum(multiprod((mat[i][sigma[i]] for i in range(n))) for sigma in itertools.permutations(range(n)))
def graySet(i, n):
  return [j for j in range(n) if (i & (1 << j)) != 0]
def grayCode(n): return (graySet(i ^ i >> 1, n) for i in range(0, 1<<n))
#print(list(len(x) for x in grayCode(5)))
#https://www2.math.upenn.edu/~wilf/website/CombinatorialAlgorithms.pdf pages 222-223
def permanent_ryser(mat):
  n = len(mat)
  if n == 0: return 1
  invset = {i for i in range(n-1)}
  #this is not O(2^(n-1)*n) as it is still n^2 without the Gray code yielding indices to add and subtract making the inner summation O(1) instead of O(n)
  #return dosign((n & 1) != 0, sum(dosign((len(S) & 1) != 0, multiprod((sum(mat[i][j] for j in S) for i in range(n)))) for S in grayCode(n)))
  rowadj = [mat[i][n-1] - sum(mat[i][j] for j in range(n-1)) for i in range(n)]
  return dosign(((n-1) & 1) != 0, sum(dosign((len(S) & 1) != 0, multiprod((rowadj[i] + (sum(mat[i][j] for j in S)<<1) for i in range(n)))) for S in grayCode(n-1))) >> (n-1)
def lsbIndex(x): return ((1 + (x ^ (x-1))) >> 1).bit_length() #count of consecutive trailing zero bits
def nextGrayCode(gcode, i):
  idx = lsbIndex(i)-1
  gcode[idx] = 1 - gcode[idx]
  return idx
#print([lsbIndex(x) for x in range(16)])
def permanent_ryser_gray(mat): #optimal row-major order
  n = len(mat)
  if n == 0: return 1
  gcode, rowsums = [0 for _ in range(n)], [mat[i][n-1] - sum(mat[i][j] for j in range(n-1)) for i in range(n)] #[0 for _ in range(n)]
  tot = multiprod(rowsums)
  #additions: (1+n)(2^n-1) multiplications: (2^n-1)(n-1)
  for i in range(1, 1<<(n-1)):
    idx = nextGrayCode(gcode, i)
    if gcode[idx]:
      for j in range(n): rowsums[j] += (mat[j][idx] << 1)
    else:
      for j in range(n): rowsums[j] -= (mat[j][idx] << 1)
    tot = plusminus((i & 1) != 0, tot, multiprod(rowsums))
  return dosign(((n-1) & 1) != 0, tot)>>(n-1)
#def getDeltas(n): return ([1 if (i & (1 << j)) != 0 else -1 for j in range(n)] for i in range(1<<n))
def getDeltas(n): return ([1 if (i & (1 << j)) != 0 else 0 for j in range(n)] for i in range(1<<n))
def permanent_glynn(mat):
  #Gray code order of deltas would yield reduction from n^2 to n similar to Ryser
  n = len(mat)
  if n == 0: return 1
  #return sum(dosign((sum(x < 0 for x in delta) & 1) != 0, multiprod((sum(delta[i] * mat[i][j] for i in range(n)) for j in range(n)))) for delta in [[1] + x for x in getDeltas(n-1)]) >> (n-1)
  return sum(dosign((sum(delta) & 1) != 0, multiprod((mat[n-1][j] + sum(dosign(delta[i]!=0, mat[i][j]) for i in range(n-1)) for j in range(n)))) for delta in getDeltas(n-1)) >> (n-1)
def permanent_glynn_gray(mat): #optimal row-major order
  n = len(mat)
  if n == 0: return 1
  #additions: n(n-1)+2^(n-1)-1 multiplications: n-1
  delta, rowsums = [1 for _ in range(n)], [sum(mat[i]) for i in range(n)] #[0 for _ in range(n)]
  tot = multiprod(rowsums)
  for i in range(1, 1<<(n-1)):
    idx = nextGrayCode(delta, i)
    if delta[idx]:
      for j in range(n): rowsums[j] += (mat[j][idx] << 1) #mat[j][idx]
    else:
      for j in range(n): rowsums[j] -= (mat[j][idx] << 1) #mat[j][idx]
    tot = plusminus((i & 1) != 0, tot, multiprod(rowsums))
  return tot >> (n-1) #tot
#can extend matrix and preserve permanent by putting ones on the diagonal
#print(list(permanent([[1 if i == j or i < 3 and j < 3 else 0 for j in range(n)] for i in range(n)]) for n in range(3, 8)))
#print(list(permanent([[1 for _ in range(n)] for _ in range(n)]) for n in range(7)))
#print(list(permanent_ryser([[1 for _ in range(n)] for _ in range(n)]) for n in range(7)))
#print(list(permanent_ryser_gray([[1 for _ in range(n)] for _ in range(n)]) for n in range(7)))
#print(list(permanent_glynn([[1 for _ in range(n)] for _ in range(n)]) for n in range(7)))
#print(list(permanent_glynn_gray([[1 for _ in range(n)] for _ in range(n)]) for n in range(7)))
#http://oeis.org/A072831 Number of bits in n!.
def get_fact_bitsizes(n):
  import math
  return math.factorial(n).bit_length()
#print(list(get_fact_bitsizes(n) for n in range(64)))    

permanent_ZOne_calculator = ZOnePermanent()
def checkSim():
  import os
  return 'SLIC_CONF' in os.environ #'MAXELEROSDIR'
hasSim, hasDFE = checkSim(), False
def calculate(x): return permanent_ZOne_calculator.calculate(np.array(x, dtype=np.uint8))
def calculateGray(x): return permanent_ZOne_calculator.calculate(np.array(x, dtype=np.uint8), gray=True)
def calculateRows(x): return permanent_ZOne_calculator.calculate(np.array(x, dtype=np.uint8), rows=True)
def calculateRowsGray(x): return permanent_ZOne_calculator.calculate(np.array(x, dtype=np.uint8), rows=True, gray=True)
def calculateGlynn(x): return permanent_ZOne_calculator.calculate(np.array(x, dtype=np.uint8), glynn=True)
def calculateGlynnGray(x): return permanent_ZOne_calculator.calculate(np.array(x, dtype=np.uint8), gray=True, glynn=True)
def calculateDFE(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True)
def calculateDFEGray(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, gray=True)
def calculateDFERows(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, rows=True)
def calculateDFERowsGray(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, rows=True, gray=True)
def calculateDFEGlynn(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, glynn=True)
def calculateDFEGlynnGray(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, gray=True, glynn=True)
def calculateDFEDual(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, dual=True)
def calculateDFEGrayDual(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, gray=True, dual=True)
def calculateDFERowsDual(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, rows=True, dual=True)
def calculateDFERowsGrayDual(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, rows=True, gray=True, dual=True)
def calculateDFEGlynnDual(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, glynn=True, dual=True)
def calculateDFEGlynnGrayDual(x): return permanent_ZOne_calculator.calculateDFE(np.array(x, dtype=np.uint8), sim=True, gray=True, glynn=True, dual=True)
permfuncs = [
  #permanent, permanent_ryser, permanent_ryser_gray, permanent_glynn, permanent_glynn_gray,
  calculate, calculateGray, #calculateRows, calculateRowsGray,
  calculateGlynn, calculateGlynnGray] + ([] if not hasSim else [
  calculateDFE, calculateDFEGray, calculateDFERows, calculateDFERowsGray,
  #calculateDFEGlynn, calculateDFEGlynnGray,
  calculateDFEDual, #calculateDFEGrayDual, calculateDFERowsDual, calculateDFERowsGrayDual,
  #calculateDFEGlynnDual, calculateDFEGlynnGrayDual,
  ]) + ([] if not hasDFE else [
  ])

#empirical validation of correctness
def validate_permanent():
  import math
  nmax = 15
  alloneresult = list(math.factorial(n) for n in range(0, nmax))
  for n in range(nmax):
    Adiag = [[1 if i == j or i < 3 and j < 3 else 0 for j in range(n)] for i in range(n)]
    A = [[1 for _ in range(n)] for _ in range(n)]
    for func in permfuncs:
      print(func.__name__, n)
      res = func(A)
      assert res == alloneresult[n], (func.__name__, n, res, alloneresult[n])
      res = func(Adiag)
      assert res == alloneresult[min(3, n)], (func.__name__, n, res, alloneresult[min(3, n)]) 
def timing_permanent():
  import timeit
  nmax = 15
  xaxis = list(range(nmax))
  results = [[] for _ in permfuncs]
  for n in xaxis:
    A = [[1 for _ in range(n)] for _ in range(n)]
    for i, func in enumerate(permfuncs):
      results[i].append(timeit.timeit(lambda: func(A), number=2)) 
  import matplotlib.pyplot as plt
  from matplotlib.ticker import MaxNLocator
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  for i, resset in enumerate(results):
    ax1.plot(xaxis, resset, label=permfuncs[i].__name__)
  ax1.set_xlabel("Size (n)")
  ax1.set_ylabel("Time (s)")
  ax1.legend()
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax1.set_title("0-1 Permanent Computation of Square Matrix A (n=|A|)")
  fig.savefig("zonepermtime.svg", format="svg")

validate_permanent()
timing_permanent()








