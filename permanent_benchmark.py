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

DEPTH = 16 if hasSim else 40

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
def permanent_walrus_quad_Ryser(Arep): return 1+0j if len(Arep) == 0 else perm_complex(Arep, quad=True) #fastest
def permanent_walrus_quad_BBFG(Arep): return 1+0j if len(Arep) == 0 else perm_BBFG_complex(Arep) #ChinHuh, 2^4*Glynn_Cpp, 2^5*walrus_quad_Ryser
def permanent_Glynn_Cpp(Arep): return permanent_Glynn_calculator.calculate(Arep) #2*walrus_quad_Ryser
def permanent_Glynn_SIM(Arep): return permanent_Glynn_calculator.calculateDFE(Arep)
def permanent_Glynn_SIMDual(Arep): return permanent_Glynn_calculator.calculateDFE(Arep, dual=True)
def permanent_Glynn_DFE(Arep): return permanent_Glynn_calculator.calculateDFE(Arep)
def permanent_Glynn_DFEDual(Arep): return permanent_Glynn_calculator.calculateDFE(Arep, dual=True)
def permanent_ChinHuh_calculator(Arep): #walrus_quad_BBFG, 2^4*Glynn_Cpp, 2^5*walrus_quad_Ryser
  if len(Arep) == 0: return 1+0j
  input_state = np.ones(Arep.shape[0], np.int64)
  output_state = np.ones(Arep.shape[0], np.int64)
  return ChinHuhPermanentCalculator( Arep, input_state, output_state ).calculate()

dfePermFuncs = ((permanent_Glynn_SIM, permanent_Glynn_SIMDual) if hasSim else (permanent_Glynn_DFE, permanent_Glynn_DFEDual))
largePermFuncs = (permanent_Glynn_Cpp, permanent_walrus_quad_Ryser) + dfePermFuncs
permFuncs = (permanent_glynn, permanent_glynn_gray, permanent_walrus_quad_BBFG, permanent_ChinHuh_calculator) + largePermFuncs

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

def load_test_data():
  nmax = 40
  randfuncs = (unitary_group.rvs, )#generate_random_unitary):
  import os, pickle
  if os.path.isfile("matrices.bin"):
    with open("matrices.bin", "rb") as f: 
      gen_test_data = pickle.load(f)
  else:
    # generate the random matrix
    gen_test_data = {rf.__name__:{dim:np.random.random((dim, dim))+np.random.random((dim, dim))*1j if dim <= 1 else rf(dim) for dim in range(nmax+1)} for rf in randfuncs}
    with open("matrices.bin", "wb") as f:
      pickle.dump(gen_test_data, f)
  return gen_test_data
def verify():
  ERRBOUND = 1e-10
  nmax = DEPTH
  gen_test_data = load_test_data()
  import os, pickle
  if os.path.isfile("verifydata.bin"):
    with open("verifydata.bin", "rb") as f:
      res = pickle.load(f)
  else: res = {}
  for key in gen_test_data:
    A = gen_test_data[key]
    if not key in res: res[key] = {}
    for func in largePermFuncs:
      if not func.__name__ in res[key]: res[key][func.__name__] = []
      print("Verifying", func.__name__)
      for dim in range(nmax+1):
        if len(res[key][func.__name__]) <= dim or func in dfePermFuncs:
          r = func(A[dim])
          if len(res[key][func.__name__]) <= dim: res[key][func.__name__].append(r)
          else: res[key][func.__name__][dim] = r
          print(r)
          with open("verifydata.bin", "wb") as f:
            pickle.dump(res, f)
        if dim >= 24: print(dim, func.__name__, res[key][func.__name__][dim])
    with open("verifydata.csv", "w") as f:
      import csv
      writer = csv.writer(f, delimiter='\t')
      writer.writerow(["Absolute Error compared to " + largePermFuncs[0].__name__]) 
      writer.writerow(["Size (n)"] + [f.__name__ for f in largePermFuncs])
      writer.writerows([[i] + [abs(res[key][largePermFuncs[0].__name__][i] - res[key][x.__name__][i]) for x in largePermFuncs] for i in range(len(res[key][largePermFuncs[0].__name__]))])
      writer.writerow(["Relative Error compared to " + largePermFuncs[0].__name__]) 
      writer.writerow(["Size (n)"] + [f.__name__ for f in largePermFuncs])
      writer.writerows([[i] + [abs(res[key][largePermFuncs[0].__name__][i] - res[key][x.__name__][i]) / abs(res[key][largePermFuncs[0].__name__][i]) for x in largePermFuncs] for i in range(len(res[key][largePermFuncs[0].__name__]))])
      writer.writerow(["Permanent Computation Raw Results"])
      writer.writerow(["Size (n)"] + [f.__name__ for f in largePermFuncs])
      writer.writerows([[i] + [res[key][x.__name__][i] for x in largePermFuncs] for i in range(len(res[key][largePermFuncs[0].__name__]))])
      for dim in range(nmax+1):
        writer.writerow(["Random Unitary Test Matrix " + str(dim) + "x" + str(dim)])
        if dim != 0: writer.writerow([""] + [str(j) for j in range(dim)])
        for i in range(dim):
          writer.writerow([i] + [A[dim][i][j] for j in range(dim)])
    for i in range(len(res[key][largePermFuncs[0].__name__])):
      assert all(abs(res[key][largePermFuncs[0].__name__][i] - res[key][x][i]) < ERRBOUND for x in res[key] if x != largePermFuncs[0].__name__)
      assert all(abs((res[key][largePermFuncs[0].__name__][i] - res[key][x][i]) / res[key][largePermFuncs[0].__name__][i]) < ERRBOUND for x in res[key] if x != largePermFuncs[0].__name__)
  
def timing():
  import timeit
  nmax = DEPTH
  xaxis = list(range(nmax+1))
  results = [[] for _ in largePermFuncs]
  gen_test_data = load_test_data()
  import os, pickle
  if os.path.isfile("resultdata.bin"):
    with open("resultdata.bin", "rb") as f:
      results = pickle.load(f)
  else: results = {}  
  for key in gen_test_data:
    A = gen_test_data[key]
    if not key in results: results[key] = {}
    for func in largePermFuncs:
      if not func.__name__ in results[key]: results[key][func.__name__] = []
      print("Testing", func.__name__)
      for dim in xaxis:
        if len(results[key][func.__name__]) <= dim or func in dfePermFuncs:
          mplier = 5 if dim < 24 else 1
          r = timeit.timeit(lambda: func(A[dim]), number=mplier) / mplier
          if len(results[key][func.__name__]) <= dim: results[key][func.__name__].append(r)
          else: results[key][func.__name__][dim] = r
          if dim >= 24: print(dim, func.__name__, results[key][func.__name__][dim])
          with open("resultdata.bin", "wb") as f:
            pickle.dump(results, f)        
    with open("resultdata.csv", "w") as f:
      import csv
      writer = csv.writer(f, delimiter='\t')
      writer.writerow(["Size (n)"] + [f.__name__ for f in largePermFuncs])
      writer.writerows([[i] + [results[key][x.__name__][i] for x in largePermFuncs] for i in range(len(results[key][largePermFuncs[0].__name__]))])
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for f in largePermFuncs:
      ax1.plot(xaxis, results[key][f.__name__], label=f.__name__)
    ax1.set_xlabel("Size")  
    ax1.set_yscale('log', base=10)
    ax1.set_ylabel("Time (log10 s)")
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title("Permanent Computation of Square Matrix A (n=|A|)")
    fig.savefig("glynnpermtime.svg", format="svg")
        
verify()
timing()
