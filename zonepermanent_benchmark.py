import numpy as np
from piquassoboost.sampling.Boson_Sampling_Utilities import ZOnePermanent

def permanent(mat):
  import functools, itertools
  n = len(mat)
  if n == 0: return 1
  return sum(functools.reduce(lambda x, y: x * y, (mat[i][sigma[i]] for i in range(n))) for sigma in itertools.permutations(range(n)))
def graySet(i, n):
  return [j for j in range(n) if (i & (1 << j)) != 0]
def grayCode(n): return (graySet(i ^ i >> 1, n) for i in range(1<<n))
#print(list(len(x) for x in grayCode(5)))
def permanent_ryser(mat):
  import functools
  n = len(mat)
  if n == 0: return 1
  #print(list((S, (1 if (len(graySet(S, n)) & 1) == 0 else -1) * functools.reduce(lambda x, y: x * y, (sum(mat[i][j] for j in graySet(S, n)) for i in range(n)))) for S in range(1<<n)))
  #this is not O(2^(n-1)*n) as it is still n^2 without the Gray code yielding indices to add and subtract making the inner summation O(1) instead of O(n)
  return (1 if (n & 1) == 0 else -1) * sum((1 if (len(S) & 1) == 0 else -1) * functools.reduce(lambda x, y: x * y, (sum(mat[i][j] for j in S) for i in range(n))) for S in grayCode(n))
def lsbIndex(x): return ((1 + (x ^ (x-1))) >> 1).bit_length() #count of consecutive trailing zero bits
def nextGrayCode(gcode, i):
  idx = lsbIndex(i)-1
  gcode[idx] = 1 - gcode[idx]
  return idx
#print([lsbIndex(x) for x in range(16)])
def permanent_ryser_gray(mat): #optimal row-major order
  import functools
  n = len(mat)
  if n == 0: return 1
  tot, gcode, rowsums = 0, [0 for _ in range(n)], [0 for _ in range(n)]
  for i in range(1, 1<<n):
    idx = nextGrayCode(gcode, i)
    if gcode[idx]:
      for j in range(n): rowsums[j] += mat[j][idx]
    else:
      for j in range(n): rowsums[j] -= mat[j][idx]
    tot += (1 if (sum(gcode) & 1) == 0 else -1) * functools.reduce(lambda x, y: x * y, rowsums)
  return (1 if (n & 1) == 0 else -1) * tot
def getDeltas(n): return ([1 if (i & (1 << j)) != 0 else -1 for j in range(n)] for i in range(1<<n))
def permanent_glynn(mat):
  import functools
  #Gray code order of deltas would yield reduction from n^2 to n similar to Ryser
  n = len(mat)
  if n == 0: return 1
  return sum((-1 if (sum(x < 0 for x in delta) & 1) != 0 else 1) * functools.reduce(lambda x, y: x * y, (sum(delta[i] * mat[i][j] for i in range(n)) for j in range(n))) for delta in [[1] + x for x in getDeltas(n-1)]) >> (n-1)
def nextOneGrayCode(gcode, i):
  idx = lsbIndex(i)-1
  gcode[idx] *= -1
  return idx
def permanent_glynn_gray(mat): #optimal row-major order
  import functools
  n = len(mat)
  if n == 0: return 1
  delta, rowsums = [1 for _ in range(n)], [sum(mat[i]) for i in range(n)]
  tot = functools.reduce(lambda x, y: x * y, rowsums)
  for i in range(1, 1<<(n-1)):
    idx = nextOneGrayCode(delta, i)
    if delta[idx] > 0:
      for j in range(n): rowsums[j] += mat[j][idx] * 2
    else:
      for j in range(n): rowsums[j] -= mat[j][idx] * 2
    tot += (-1 if (sum(x < 0 for x in delta) & 1) != 0 else 1) * functools.reduce(lambda x, y: x * y, rowsums)
  return tot >> (n-1)
#can extend matrix and preserve permanent by putting ones on the diagonal
#print(list(permanent([[1 if i == j or i < 3 and j < 3 else 0 for j in range(n)] for i in range(n)]) for n in range(3, 8)))
#print(list(permanent([[1 for _ in range(n)] for _ in range(n)]) for n in range(7)))
#print(list(permanent_ryser_gray([[1 for _ in range(n)] for _ in range(n)]) for n in range(7)))
#print(list(permanent_glynn_gray([[1 for _ in range(n)] for _ in range(n)]) for n in range(7)))
#http://oeis.org/A072831 Number of bits in n!.
def get_fact_bitsizes(n):
  import math
  return math.factorial(n).bit_length()
print(list(get_fact_bitsizes(n) for n in range(64)))    

print(list(permanent([[1 if i == j or i < 3 and j < 3 else 0 for j in range(n)] for i in range(n)]) for n in range(3, 8)))
#[permanent_ryser([[1 if i == j or i < 3 and j < 3 else 0 for j in range(n)] for i in range(n)]) for n in range(10, 11)]
permanent_ZOne_calculator = ZOnePermanent()
print(list(permanent_ZOne_calculator.calculate(np.array([[1 if i == j or i < 3 and j < 3 else 0 for j in range(n)] for i in range(n)], dtype=np.uint8)) for n in range(3, 8)))
#print(list(permanent_ZOne_calculator.calculateDFE(np.array([[1 for _ in range(n)] for _ in range(n)], dtype=np.uint8), sim=True) for n in range(0, 21)))
print(list(permanent_ZOne_calculator.calculateDFE(np.array([[1 for _ in range(n)] for _ in range(n)], dtype=np.uint8), sim=True, gray=True) for n in range(3, 8)))
#print(list(permanent_ZOne_calculator.calculateDFE(np.array([[1 for _ in range(n)] for _ in range(n)], dtype=np.uint8), sim=True, rows=True) for n in range(0, 21)))
print(list(permanent_ZOne_calculator.calculateDFE(np.array([[1 for _ in range(n)] for _ in range(n)], dtype=np.uint8), sim=True, rows=True, gray=True) for n in range(0, 21)))
print(list(permanent_ZOne_calculator.calculate(np.array([[1 for _ in range(n)] for _ in range(n)], dtype=np.uint8)) for n in range(0, 26)))
import math
print(list(math.factorial(n) for n in range(0, 26)))











