import numpy as np
from thewalrus.libwalrus import perm_complex, perm_real, perm_BBFG_real, perm_BBFG_complex
from piquassoboost.sampling.Boson_Sampling_Utilities import ChinHuhPermanentCalculator, GlynnPermanent
import piquasso as pq
import random
from scipy.stats import unitary_group

import time

def checkSim():
  import os
  return 'SLIC_CONF' in os.environ #'MAXELEROSDIR'
hasSim = checkSim(); hasDFE = not hasSim

DEPTH = 40

def pairwise(t):
    return zip(t[::2], t[1::2])

def generate_random_unitary( dim ):

    with pq.Program() as program:
        for column in range(DEPTH):
            for modes in pairwise(range(column % 2, dim)):
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0, 2 * np.pi)

                pq.Q(*modes) | pq.Beamsplitter(theta=theta, phi=phi)

        pq.Q() | pq.Sampling()

    state = pq.SamplingState(1, 1, 1, 1, 0, 0)
    state.apply(program, shots=1)


    return state.interferometer




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
def permanent_glynn(mat):
  #Gray code order of deltas would yield reduction from n^2 to n similar to Ryser
  n = len(mat)
  if n == 0: return 1
  #return sum(dosign((sum(x < 0 for x in delta) & 1) != 0, multiprod((sum(delta[i] * mat[i][j] for i in range(n)) for j in range(n)))) for delta in [[1] + x for x in getDeltas(n-1)]) >> (n-1)
  return sum(dosign((sum(delta) & 1) != 0, multiprod((mat[n-1][j] + sum(dosign(delta[i]!=0, mat[i][j]) for i in range(n-1)) for j in range(n)))) for delta in getDeltas(n-1)) / (1 << (n-1))
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


permanent_Glynn_calculator = GlynnPermanent( )
#https://github.com/XanaduAI/thewalrus/issues/319 - 0 case bugged in Ryser/BBFG
def permanent_walrus_quad_Ryser(Arep): return 1+0j if len(Arep) == 0 else perm_complex(Arep, quad=True)
def permanent_walrus_quad_BBFG(Arep): return 1+0j if len(Arep) == 0 else perm_BBFG_complex(Arep)
def permanent_Glynn_Cpp(Arep): return permanent_Glynn_calculator.calculate(Arep)
def permanent_Glynn_SIM(Arep): return permanent_Glynn_calculator.calculateDFE(Arep)
def permanent_Glynn_SIMDual(Arep): return permanent_Glynn_calculator.calculateDFE(Arep, dual=True)
def permanent_Glynn_DFE(Arep): return permanent_Glynn_calculator.calculateDFE(Arep)
def permanent_Glynn_DFEDual(Arep): return permanent_Glynn_calculator.calculateDFE(Arep, dual=True)
def permanent_ChinHuh_calculator(Arep):
  if len(Arep) == 0: return 1+0j
  input_state = np.ones(Arep.shape[0], np.int64)
  output_state = np.ones(Arep.shape[0], np.int64)
  return ChinHuhPermanentCalculator( Arep, input_state, output_state ).calculate()

permFuncs = (permanent_glynn, permanent_glynn_gray, permanent_walrus_quad_Ryser, permanent_walrus_quad_BBFG, permanent_ChinHuh_calculator, permanent_Glynn_Cpp) + ((permanent_Glynn_SIM, permanent_Glynn_SIMDual) if hasSim else (permanent_Glynn_DFE, permanent_Glynn_DFEDual))

#np.save("mtx", A )
#A = np.load("mtx.npy")
#Arep = A

#Arep = np.zeros((dim,dim), dtype=np.complex128)
#for idx in range(dim):
#    Arep[:,idx] = A[:,0]
        
# calculate the permanent using walrus library
"""
iter_loops = 2
time_walrus = 1000000000        
for idx in range(iter_loops):
    start = time.time()   
    permanent_walrus_quad_Ryser = perm_complex(Arep, quad=True)
    time_loc = time.time() - start
    start = time.time()   
       
    if time_walrus > time_loc:
        time_walrus = time_loc

    time_walrus_BBFG = 1000000        
    for idx in range(iter_loops):
        start = time.time()   
        permanent_walrus_quad_BBFG = 0#perm_BBFG_complex(Arep)
        time_loc = time.time() - start
        start = time.time()   
       
        if time_walrus_BBFG > time_loc:
            time_walrus_BBFG = time_loc

        
        # calculate the hafnian with the power trace method using the piquasso library
        input_state = np.ones(dim, np.int64)
#        input_state = np.zeros(dim, np.int64)
#        input_state[0] = dim
        output_state = np.ones(dim, np.int64)


permanent_ChinHuh_calculator = ChinHuhPermanentCalculator( A, input_state, output_state )
time_Cpp = 1000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_ChinHuh_Cpp = 0#permanent_ChinHuh_calculator.calculate()

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Cpp > time_loc:
        time_Cpp = time_loc





permanent_Glynn_calculator = GlynnPermanent( )
time_Glynn_Cpp = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_Cpp = 0#permanent_Glynn_calculator.calculate(Arep)

    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_Cpp > time_loc:
        time_Glynn_Cpp = time_loc



#permanent_Glynn_calculator = GlynnPermanent(  )
time_Glynn_DFE = 1000000000
for idx in range(iter_loops):
    start = time.time()   

    permanent_Glynn_DFE = 0#permanent_Glynn_calculator.calculateDFE(Arep)
    #print( permanent_Glynn_DFE )
    time_loc = time.time() - start
    start = time.time()   
       
    if time_Glynn_DFE > time_loc:
        time_Glynn_DFE = time_loc


        
print(' ')
print( permanent_walrus_quad_Ryser )
print( permanent_walrus_quad_BBFG )
print( permanent_ChinHuh_Cpp )
print( permanent_Glynn_Cpp )
print( permanent_Glynn_DFE )


print(' ')
print('*******************************************')
print('Time elapsed with walrus: ' + str(time_walrus))
print('Time elapsed with walrus BBFG : ' + str(time_walrus_BBFG))
print('Time elapsed with piquasso: ' + str(time_Cpp))
print('Time elapsed with piquasso Glynn: ' + str(time_Glynn_Cpp))
print('Time elapsed with DFE Glynn: ' + str(time_Glynn_DFE))
#print( "speedup: " + str(time_walrus/time_Cpp) )
#print( "speedup Glynn: " + str(time_walrus/time_Glynn_Cpp) )
print(' ')
print(' ')


#print( 'Relative difference between quad walrus and piquasso result: ' + str(abs(permanent_walrus_quad_Ryser-permanent_ChinHuh_Cpp)/abs(permanent_ChinHuh_Cpp)*100) + '%')
#print( 'Relative difference between quad walrus and piquasso Glynn result: ' + str(abs(permanent_walrus_quad_Ryser-permanent_Glynn_Cpp)/abs(permanent_Glynn_Cpp)*100) + '%')


"""

def verify():
  ERRBOUND = 1e-10
  nmax = 17
  # generate the random matrix
  for gen_test_data in (unitary_group.rvs, ):#generate_random_unitary):
    A = {dim:np.random.random((dim, dim))+np.random.random((dim, dim))*1j if dim <= 1 else gen_test_data(dim) for dim in range(nmax)}
    res = [[] for _ in permFuncs]
    for i, func in enumerate(permFuncs):
      #print(func.__name__)
      for dim in range(nmax):
        res[i].append(func(A[dim]))
        print(dim, func.__name__, func(A[dim]))
    for i in range(len(res[0])):
      assert all(abs(res[0][i] - x[i]) < ERRBOUND for x in res[1:])
def timing():
  import timeit
  nmax = 17
  xaxis = list(range(nmax))
  results = [[] for _ in permFuncs]
  for gen_test_data in (unitary_group.rvs, ):
    A = {dim:np.random.random((dim, dim))+np.random.random((dim, dim))*1j if dim <= 1 else gen_test_data(dim) for dim in range(nmax)}
    for i, func in enumerate(permFuncs):
      for dim in xaxis:
        results[i].append(timeit.timeit(lambda: func(A[dim]), number=2))
  import matplotlib.pyplot as plt
  from matplotlib.ticker import MaxNLocator
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  for i, resset in enumerate(results):
    ax1.plot(xaxis, resset, label=permFuncs[i].__name__)
  ax1.set_xlabel("Size")  
  ax1.set_yscale('log', base=2)
  ax1.set_ylabel("Time (log2 s)")
  ax1.legend()
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax1.set_title("Permanent Computation of Square Matrix A (n=|A|)")
  fig.savefig("glynnpermtime.svg", format="svg")
        
verify()
timing()
