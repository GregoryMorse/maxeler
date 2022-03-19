import time
import itertools

import numpy as np

import piquasso as pq


def print_histogram():
    def func(samples):
        hist = dict()

        for sample in samples:
            key = tuple(sample)
            if key in hist.keys():
                hist[key] += 1
            else:
                hist[key] = 1

        for key in hist.keys():
            print(f"{key}: {hist[key]}")

    return func
    
def test_complex_sampling(print_histogram):
    """
    NOTE: Expected distribution probabilities:
    (0, 2, 0, 1, 1): 0.1875
    (0, 2, 1, 0, 1): 0.1875
    (1, 1, 0, 1, 1): 0.1250
    (1, 1, 1, 0, 1): 0.1250
    (2, 0, 0, 1, 1): 0.1875
    (2, 0, 1, 0, 1): 0.1875
    """

    for _ in itertools.repeat(None, 10):
        shots = 10000

        with pq.Program() as program:
            pq.Q(0, 1) | pq.Beamsplitter(np.pi / 3)
            pq.Q(2)    | pq.Fourier()
            pq.Q(2, 3) | pq.Beamsplitter(np.pi / 4)

            pq.Q() | pq.Sampling()

        state = pq.SamplingState(1, 1, 1, 0, 1)

        t0 = time.time()

        result = state.apply(program, shots=shots)

        print("C++ time elapsed:", time.time() - t0, "s")

        print_histogram(result.samples)

#test_complex_sampling(print_histogram())

DEPTH = 8

def dosign(parity, x): return -x if parity else x
def plusminus(parity, base, x): return base - x if parity else base + x
def prod(x, y): return x * y
def multiprod(l):
  import functools
  return functools.reduce(prod, l)
def getDeltas(n): return ([1 if (i & (1 << j)) != 0 else 0 for j in range(n)] for i in range(1<<n))
def lsbIndex(x): return ((1 + (x ^ (x-1))) >> 1).bit_length() #count of consecutive trailing zero bits
def nextGrayCode(gcode, i):
  idx = lsbIndex(i)-1
  gcode[idx] = 1 - gcode[idx]
  return idx
def permanent_rectangular(mat):
  import itertools
  m = mat.shape[0]
  if m == 0: return 1
  n = mat.shape[1]
  if n == 0: return 1
  assert m <= n
  return sum(multiprod((mat[i][sigma[i]] for i in range(m))) for sigma in itertools.permutations(range(n), m))
def multiplicities_to_mat(mat, inp, outp):
  mat = np.repeat(np.repeat(mat, inp, axis=1), outp, axis=0)
  return mat.transpose() if mat.shape[0] > mat.shape[1] else mat
def permanent_repeated(mat, inp, outp):
  return permanent_rectangular(multiplicities_to_mat(mat, inp, outp))
def mathcomb(n, k): #binomial coefficients
  import math #return math.comb(n, k)
  return math.factorial(n) // (math.factorial(k) * math.factorial(n-k)) 
def mat_mul_rows(matzones, mat, inpidx, mplicity):
  return np.array([[x[k] + sum(mat[y][k] * mplicity[j] for j, y in enumerate(inpidx)) for k in range(len(mat[0]))] if i == 0 else x for i, x in enumerate(matzones)])
def binomial_gcode(bc, parity, n, k):
  return bc*k//(n-k+1) if parity else bc*(n-k)/(k+1)
def permanent_square_repeated(mat, inp, outp): #hybrid single multiplicity/repeated-Chin Huh method without proper Gray code, but one multiplicities using normal permanent
  #the Gray code anchor must be on the first row of the rectangular computation or this algorithm will be incorrect!
  matoutp = np.repeat(mat, outp, axis=0).transpose()
  matzones = [matoutp[i] for i in range(len(inp)) if inp[i] == 1]
  inpidx = [i for i, x in enumerate(inp) if x > 1]
  curmp = [inp[x] for x in inpidx]
  if len(curmp) == 0: return 1 if len(matoutp) == 0 else permanent_rectangular(multiplicities_to_mat(matoutp, [1]*len(matoutp[0]), inp)) #all 0-1 multiplicities
  if len(matzones) == 0:
    idx = np.argmin(curmp); curmp[idx] -= 1 #anchor to lowest multiplicity row for greatest reduction
    matzones = [matoutp[inpidx[idx]]]
  inp = [x for x in curmp]
  tot, parity, gcodeidx = 0, False, 0
  #a=n!/(k!(n-k)!) and b=n!/((k-1)!(n-k+1)!) b/a=k/(n-k+1)
  #a=n!/(k!(n-k)!) and b=n!/((k+1)!(n-k-1)!) b/a=(n-k)/(k+1)
  cur_multiplicity = 1
  while True:
    tot = plusminus(parity, tot, cur_multiplicity * permanent_glynn_rectangular(mat_mul_rows(matzones, matoutp, inpidx, curmp)))
    parity = not parity
    for i in range(len(curmp)-1, -1, -1):
      curdir = (gcodeidx & (1 << i)) == 0
      if not curdir and curmp[i] != inp[i] or curdir and curmp[i] != -inp[i]: 
        cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[i], (curmp[i] + inp[i]) // 2)
        curmp[i] = plusminus(curdir, curmp[i], 2)
        #for j in range(i+1, len(curmp)): gcodeidx ^= (1 << j)
        gcodeidx ^= ((1 << len(curmp)) - (1 << (i+1)))
        break
    else: break 
  return tot / 2**(sum(inp)) #2**(sum(inp))==sum of all cur_multiplicity values

def permanent_glynn_repeated(mat): pass
def permanent_glynn(mat):
  #Gray code order of deltas would yield reduction from n^2 to n similar to Ryser
  n = len(mat)
  if n == 0: return 1
  #return sum(dosign((sum(x < 0 for x in delta) & 1) != 0, multiprod((sum(delta[i] * mat[i][j] for i in range(n)) for j in range(n)))) for delta in [[1] + x for x in getDeltas(n-1)]) >> (n-1)
  return sum(dosign((sum(delta) & 1) != 0, multiprod((mat[n-1][j] + sum(dosign(delta[i]!=0, mat[i][j]) for i in range(n-1)) for j in range(n)))) for delta in getDeltas(n-1)) / (1 << (n-1))
def permanent_glynn_rectangular(mat):
  #Gray code order of deltas would yield reduction from n^2 to n similar to Ryser
  n = len(mat); m = len(mat[0])
  if n == 0: return 1
  #return sum(dosign((sum(x < 0 for x in delta) & 1) != 0, multiprod((sum(delta[i] * mat[i][j] for i in range(n)) for j in range(n)))) for delta in [[1] + x for x in getDeltas(n-1)]) >> (n-1)
  return sum(dosign((sum(delta) & 1) != 0, multiprod((mat[0][j] + sum(dosign(delta[i-1]!=0, mat[i][j]) for i in range(1, n)) for j in range(m)))) for delta in getDeltas(n-1)) / (1 << (n-1))
def permanent_glynn_gray(mat): #optimal row-major order
  mat = np.copy(mat).transpose()
  n = len(mat)
  if n == 0: return 1
  #additions: n(n-1)+2^(n-1)-1 multiplications: n-1
  renormalize = [0 for _ in range(n)]
  for i in range(n):
    for j in range(n):
      value1, value2 = abs(renormalize[i]+mat[i][j]), abs(renormalize[i]-mat[i][j])
      renormalize[i] = value1 if abs(value1) > abs(value2) else value2
  renormalize = [abs(x) for x in renormalize]
  for i in range(n):
    for j in range(n):
      mat[i][j] = mat[i][j] / renormalize[i]
  delta, rowsums = [1 for _ in range(n)], [sum(mat[i]) for i in range(n)] #[0 for _ in range(n)]
  #print(mat, rowsums)
  tot = multiprod(rowsums)
  for i in range(1, 1<<(n-1)):
    idx = nextGrayCode(delta, i)
    if delta[idx]:
      for j in range(n): rowsums[j] += (mat[j][idx] * 2) #mat[j][idx]
    else:
      for j in range(n): rowsums[j] -= (mat[j][idx] * 2) #mat[j][idx]
    tot = plusminus((i & 1) != 0, tot, multiprod(rowsums))
  tot *= multiprod(renormalize)
  return tot / (1 << (n-1)) #tot
  
from scipy.stats import unitary_group
from piquassoboost.sampling.Boson_Sampling_Utilities import ChinHuhPermanentCalculator, GlynnPermanent
permanent_Glynn_calculator = GlynnPermanent( )

def permanent_Glynn_Cpp(Arep, input_state, output_state): return permanent_Glynn_calculator.calculate_repeated(Arep, input_state, output_state)
def permanent_ChinHuh_calculator(Arep, input_state, output_state):
  if len(Arep) == 0 or not input_state.any() or not output_state.any(): return 1+0j
  return ChinHuhPermanentCalculator( Arep, input_state, output_state ).calculate()
largePermFuncs = (permanent_square_repeated, permanent_Glynn_Cpp, permanent_ChinHuh_calculator)
def verify():
  ERRBOUND = 1e-10
  nmax = DEPTH
  # generate the random matrix
  for gen_test_data in (unitary_group.rvs, ):#generate_random_unitary):
    A = {dim:np.random.random((dim, dim))+np.random.random((dim, dim))*1j if dim <= 1 else gen_test_data(dim) for dim in range(nmax+1)}
    input_states = {dim:np.random.randint(5, size=dim) for dim in range(nmax+1)}
    output_states = {dim:np.array(input_states[dim]) for dim in range(nmax+1)} #np.ones(dim, dtype=np.int64)
    for x in output_states: np.random.shuffle(output_states[x])
    res = [[] for _ in largePermFuncs]
    for i, func in enumerate(largePermFuncs):
      #print("Verifying", func.__name__)
      for dim in range(nmax+1):
        res[i].append(func(A[dim], input_states[dim], output_states[dim]))
        #print(func(np.ones((dim, dim), dtype=np.complex128)*1j, np.ones(dim, dtype=np.int64)*2, np.ones(dim, dtype=np.int64)*2))
        print(dim, func.__name__, res[i][-1])
    for i in range(len(res[0])):
      assert all(abs(res[0][i] - x[i]) < ERRBOUND for x in res[1:])
verify()
