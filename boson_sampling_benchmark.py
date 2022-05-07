import time
import itertools

import numpy as np

import piquasso as pq
import piquassoboost as pqb
from piquassoboost.sampling.simulator import BoostedSamplingSimulator
from piquassoboost.sampling.BosonSamplingSimulator import BosonSamplingSimulator
from piquassoboost.sampling.simulation_strategies.GeneralizedCliffordsSimulationStrategy import GeneralizedCliffordsSimulationStrategy, GeneralizedCliffordsSimulationStrategyChinHuh, GeneralizedCliffordsSimulationStrategySingleDFE, GeneralizedCliffordsSimulationStrategyDualDFE, GeneralizedCliffordsSimulationStrategyMultiSingleDFE, GeneralizedCliffordsSimulationStrategyMultiDualDFE

def checkSim():
  import os
  return 'SLIC_CONF' in os.environ #'MAXELEROSDIR'
hasSim = checkSim(); hasDFE = not hasSim

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
    
"""
shots = 100

with pq.Program() as program:
    pq.Q() | pq.StateVector(np.ones(dim))
    pq.Q() | pq.Interferometer(U)

    pq.Q() | pq.Sampling()

simulator = pqb.BoostedSamplingSimulator(d=dim)
"""
    
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

#Hacker's Delight division via multiplication with magic numbers
#https://github.com/hcs0/Hackers-Delight/blob/master/magicgu.py.txt
def magicgu(nmax, d):
   nc = ((nmax + 1)//d)*d - 1
   nbits = len(bin(nmax)) - 2
   for p in range(0, 2*nbits + 1):
      if 2**p > nc*(d - 1 - (2**p - 1)%d):
         m = (2**p + d - 1 - (2**p - 1)%d)//d
         return (m, p)
def mathcomb(n, k): #binomial coefficients
  import math #return math.comb(n, k)
  return math.factorial(n) // (math.factorial(k) * math.factorial(n-k)) 
def get_bincoeff_magic():
    import random
    largestbincoeff = mathcomb(40, 20)*21
    assert largestbincoeff.bit_length() == 42 #38+4 bits is the largest, anything smaller e.g. math.comb(2, 1)**20==1048576 not a concern math.comb(40, 20)==137846528820
    magicmulshift = [magicgu(largestbincoeff, d) for d in range(1, 41+1)]
    for _ in range(100000):
        assert all(random.randint(0, largestbincoeff // (d+1)) * x[0] >> x[1] for d, x in enumerate(magicmulshift))
    return magicmulshift
#[(1, 0), (1, 1), (183251937963, 39), (1, 2), (54975581389, 38), (183251937963, 40), (157073089683, 40), (1, 3), (61083979321, 39), (54975581389, 39), (199911205051, 41), (183251937963, 41), (169155635043, 41), (157073089683, 41), (146601550371, 41), (1, 4), (129354309151, 41), (61083979321, 40), (57869033041, 40), (54975581389, 40), (52357696561, 40), (199911205051, 42), (191219413527, 42), (183251937963, 42), (175921860445, 42), (169155635043, 42), (162890611523, 42), (157073089683, 42), (75828388123, 41), (146601550371, 42), (141872468101, 42), (1, 5), (133274136701, 42), (129354309151, 42), (62829235873, 41), (61083979321, 41), (59433060961, 41), (57869033041, 41), (56385211681, 41), (54975581389, 41), (214538854201, 43)]
#print(get_bincoeff_magic())
bcm = get_bincoeff_magic()
print(", ".join(str(x[0]) + "L" for x in bcm), ", ".join(str(x[1]) for x in bcm))
#test_complex_sampling(print_histogram())

DEPTH = 40 if hasSim else 40
saveFolder = "resultsbs"


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
def mat_mul_rows(matzones, mat, inpidx, mplicity):
  return np.array([[x[k] + sum(mat[y][k] * mplicity[j] for j, y in enumerate(inpidx)) for k in range(len(mat[0]))] if i == 0 else x for i, x in enumerate(matzones)])
def binomial_gcode(bc, parity, n, k):
  return bc*k//(n-k+1) if parity else bc*(n-k)//(k+1)
#https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1067.6055&rep=rep1&type=pdf
def cartesianProductGcode(counterChain, l, k):
    if counterChain == [0] * len(l): return
    for j in range(0, len(l)):
        if all(counterChain[i] == 0 for i in range(j)):
            k[j] = k[j]+1 if k[j] != (l[j] << 1)-1 else 0 #(k[j]+1) modulo 2*l[j]
    #assert k == counterToGcode(counterChain, l), (k, counterToGcode(counterChain, l))
def counterChainMplicity(l):
    import math, functools
    counterChain, k, term = [0] * len(l), [0] * len(l), [x-1 for x in l]
    while True:
        cartesianProductGcode(counterChain, l, k)
        assert k == counterToGcode(counterChain, l), (counterToGcode(counterChain, l), k, counterChain, l)
        print(counterChain, k, [k[j] if k[j] < l[j] else l[j]*2-k[j]-1 for j in range(len(l))],
            functools.reduce(lambda a, b: a * b, [math.comb(l[j]-1, l[j]-1-k[j] if k[j] < l[j] else k[j]-l[j]) for j in range(len(l))]))
        if counterChain == term: break
        i = 0
        while True:
            if counterChain[i] == l[i]-1: counterChain[i] = 0; i+=1
            else: counterChain[i] += 1; break
def locationToCounter(l, loc):
    g = []
    for x in l:
        loc, y = divmod(loc, x)
        g.append(y)
    return g
def counterToGcode(counterChain, l):
    g, parity = [x for x in counterChain], False
    for j in range(len(l)-1, -1, -1):
        if parity: g[j] += l[j]
        parity = (g[j] & 1) != 0
    return g
def divideGcode(l, p):
    import functools
    total = functools.reduce(lambda a, b: a * b, l)
    segment, rem = divmod(total, p)
    distribution = [segment + (1 if i < rem else 0) for i in range(p)]
    cursum, locs, g = 0, [], []
    for x in distribution:
        locs.append(locationToCounter(l, cursum))
        g.append(counterToGcode(locs[-1], l))
        cursum += x
    print(locs, g)
#counterChainMplicity([2, 5, 4, 6, 7, 2])
#divideGcode([2, 5, 4, 6, 7, 2], 20)
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
  skipidx = (1 << len(curmp))-1
  print(inp)
  while True:
    tot = plusminus(parity, tot, cur_multiplicity * permanent_glynn_rectangular(mat_mul_rows(matzones, matoutp, inpidx, curmp)))
    parity = not parity
    print(gcodeidx, curmp)
    for i in range(len(curmp)):
      if (skipidx & (1 << i)) != 0:
        curdir = (gcodeidx & (1 << i)) == 0
        cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[i], (curmp[i] + inp[i]) // 2)
        curmp[i] = plusminus(curdir, curmp[i], 2)
        #for j in range(i+1, len(curmp)): gcodeidx ^= (1 << j)
        if not curdir and curmp[i] == inp[i] or curdir and curmp[i] == -inp[i]: skipidx ^= ((1 << (i+1)) - 1)
        else: skipidx ^= ((1 << i) - 1)
        gcodeidx ^= (1 << i) - 1
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
from piquassoboost.sampling.Boson_Sampling_Utilities import ChinHuhPermanentCalculator, GlynnRepeatedPermanentCalculator, GlynnRepeatedSingleDFEPermanentCalculator, GlynnRepeatedDualDFEPermanentCalculator, GlynnRepeatedMultiSingleDFEPermanentCalculator, GlynnRepeatedMultiDualDFEPermanentCalculator
calculators = [None, None, None, None, None, None]

def permanent_Glynn_Cpp(Arep, input_state, output_state):
  #if len(Arep) == 0 or not input_state.any() or not output_state.any(): return 1+0j
  if calculators[0] is None: calculators[0] = GlynnRepeatedPermanentCalculator(Arep, input_state, output_state)
  else: calculators[0].matrix, calculators[0].input_state, calculators[0].output_state = Arep, input_state, output_state  
  return calculators[0].calculate()
def permanent_ChinHuh_calculator(Arep, input_state, output_state):
  #if len(Arep) == 0 or not input_state.any() or not output_state.any(): return 1+0j
  if calculators[1] is None: calculators[1] = ChinHuhPermanentCalculator(Arep, input_state, output_state)
  else: calculators[1].matrix, calculators[1].input_state, calculators[1].output_state = Arep, input_state, output_state  
  return calculators[1].calculate()
def permanent_Glynn_DFE(Arep, input_state, output_state):
  if calculators[2] is None: calculators[2] = GlynnRepeatedSingleDFEPermanentCalculator(Arep, input_state, output_state)
  else: calculators[2].matrix, calculators[2].input_state, calculators[2].output_state = Arep, input_state, output_state  
  return calculators[2].calculate()
def permanent_Glynn_DFEDual(Arep, input_state, output_state):
  if calculators[3] is None: calculators[3] = GlynnRepeatedDualDFEPermanentCalculator(Arep, input_state, output_state)
  else: calculators[3].matrix, calculators[3].input_state, calculators[3].output_state = Arep, input_state, output_state  
  return calculators[3].calculate()
def permanent_Glynn_MultiDFE(Arep, input_state, output_state):
  if calculators[4] is None: calculators[4] = GlynnRepeatedMultiSingleDFEPermanentCalculator(Arep, input_state, output_state)
  else: calculators[4].matrix, calculators[4].input_state, calculators[4].output_state = Arep, input_state, output_state  
  return calculators[4].calculate()
def permanent_Glynn_MultiDFEDual(Arep, input_state, output_state):
  if calculators[5] is None: calculators[5] = GlynnRepeatedMultiDualDFEPermanentCalculator(Arep, input_state, output_state)
  else: calculators[5].matrix, calculators[5].input_state, calculators[5].output_state = Arep, input_state, output_state  
  return calculators[5].calculate()

samplers = [None, None, None, None, None, None]
seed = 0x123456789ABCDEF
def boson_sampling_Clifford_GlynnRep(Arep, input_state, shots):
  if samplers[0] is None: samplers[0] = BosonSamplingSimulator(GeneralizedCliffordsSimulationStrategy(Arep, seed))
  else: samplers[0].simulation_strategy.interferometer_matrix = Arep
  samplers[0].simulation_strategy.seed(seed)  
  return samplers[0].get_classical_simulation_results(input_state, shots)
def boson_sampling_Clifford_ChinHuh(Arep, input_state, shots):
  if samplers[1] is None: samplers[1] = BosonSamplingSimulator(GeneralizedCliffordsSimulationStrategyChinHuh(Arep, seed))
  else: samplers[1].simulation_strategy.interferometer_matrix = Arep
  samplers[1].simulation_strategy.seed(seed)  
  return samplers[1].get_classical_simulation_results(input_state, shots)
def boson_sampling_Clifford_GlynnRepSingleDFE(Arep, input_state, shots):
  if samplers[2] is None: samplers[2] = BosonSamplingSimulator(GeneralizedCliffordsSimulationStrategySingleDFE(Arep, seed))
  else: samplers[2].simulation_strategy.interferometer_matrix = Arep
  samplers[2].simulation_strategy.seed(seed)
  return samplers[2].get_classical_simulation_results(input_state, shots)
def boson_sampling_Clifford_GlynnRepDualDFE(Arep, input_state, shots):
  if samplers[3] is None: samplers[3] = BosonSamplingSimulator(GeneralizedCliffordsSimulationStrategyDualDFE(Arep, seed))
  else: samplers[3].simulation_strategy.interferometer_matrix = Arep
  samplers[3].simulation_strategy.seed(seed)
  return samplers[3].get_classical_simulation_results(input_state, shots)
def boson_sampling_Clifford_GlynnRepMultiSingleDFE(Arep, input_state, shots):
  if samplers[4] is None: samplers[4] = BosonSamplingSimulator(GeneralizedCliffordsSimulationStrategyMultiSingleDFE(Arep, seed))
  else: samplers[4].simulation_strategy.interferometer_matrix = Arep
  samplers[4].simulation_strategy.seed(seed)
  return samplers[4].get_classical_simulation_results(input_state, shots)
def boson_sampling_Clifford_GlynnRepMultiDualDFE(Arep, input_state, shots):
  if samplers[5] is None: samplers[5] = BosonSamplingSimulator(GeneralizedCliffordsSimulationStrategyMultiDualDFE(Arep, seed))
  else: samplers[5].simulation_strategy.interferometer_matrix = Arep
  samplers[5].simulation_strategy.seed(seed)
  return samplers[5].get_classical_simulation_results(input_state, shots)
  
testPermFuncs = (permanent_repeated, permanent_square_repeated)
dfePermFuncs = (permanent_Glynn_DFE, permanent_Glynn_DFEDual, permanent_Glynn_MultiDFE, permanent_Glynn_MultiDFEDual) if hasSim else (permanent_Glynn_MultiDFE, permanent_Glynn_MultiDFEDual)
largePermFuncs = (permanent_Glynn_Cpp, permanent_ChinHuh_calculator) + dfePermFuncs
testSamplingFuncs = ()
dfeSamplingFuncs = ((boson_sampling_Clifford_GlynnRepMultiSingleDFE, boson_sampling_Clifford_GlynnRepMultiDualDFE, boson_sampling_Clifford_GlynnRepSingleDFE, boson_sampling_Clifford_GlynnRepDualDFE) if hasSim else (boson_sampling_Clifford_GlynnRepMultiSingleDFE, boson_sampling_Clifford_GlynnRepMultiDualDFE))
samplingFuncs = (boson_sampling_Clifford_GlynnRep, boson_sampling_Clifford_ChinHuh) + dfeSamplingFuncs
def load_test_data():
  nmax = 40
  randfuncs = (unitary_group.rvs, )#generate_random_unitary):
  import os, pickle
  if not os.path.isdir(saveFolder): os.mkdir(saveFolder)
  if os.path.isfile(os.path.join(saveFolder, "repmatrices.bin")):
    with open(os.path.join(saveFolder, "repmatrices.bin"), "rb") as f: 
      gen_test_data = pickle.load(f)
  else:
    # generate the random matrix
    gen_test_data = {rf.__name__:{dim:(np.random.random((dim, dim))+np.random.random((dim, dim))*1j if dim <= 1 else rf(dim),
      {d:np.array([], dtype=np.int64) if dim == 0 else np.random.multinomial(d, [1/dim]*dim) for d in range(nmax+1)},
      {d:np.array([], dtype=np.int64) if dim == 0 else np.random.multinomial(d, [1/dim]*dim) for d in range(nmax+1)}
         ) for dim in range(nmax+1)} for rf in randfuncs}
    with open(os.path.join(saveFolder, "repmatrices.bin"), "wb") as f:
      pickle.dump(gen_test_data, f)
  return gen_test_data
def random_realistic_input_states(photons, dim):
    indexes = set(np.random.choice(range(dim), photons, False))
    return np.array([1 if i in indexes else 0 for i in range(dim)])
def stability(nmax, photons, times=1000):
    for dim in range(nmax+20, nmax*2+1):
        print("Stability testing", dim)
        for _ in range(times):
            arr = np.random.random((dim, dim))+np.random.random((dim, dim))*1j if dim <= 1 else unitary_group.rvs(dim)
            if dim >= photons: input_state = random_realistic_input_states(photons, dim)
            else: input_state = np.array([], dtype=np.int64) if dim == 0 else np.random.multinomial(photons, [1/dim]*dim)
            output_state = np.array([], dtype=np.int64) if dim == 0 else np.random.multinomial(photons, [1/dim]*dim)
            assert sum(input_state) == sum(output_state)
            print(input_state, output_state)        
            permanent_Glynn_MultiDFE(arr, input_state, output_state)
            permanent_Glynn_MultiDFE(arr, output_state, input_state)
def other_stability(nmax, photons, times=1):
    shots = 100
    for dim in range(nmax+20, nmax*2+1):
        print(dim)
        if dim >= photons: input_state = random_realistic_input_states(photons, dim)
        else: input_state = np.array([], dtype=np.int64) if dim == 0 else np.random.multinomial(photons, [1/dim]*dim)
        U = np.random.random((dim, dim))+np.random.random((dim, dim))*1j if dim <= 1 else unitary_group.rvs(dim)
        with pq.Program() as program:
            pq.Q() | pq.StateVector(input_state)
            pq.Q() | pq.Interferometer(U)
    
            pq.Q() | pq.Sampling()
        simulator = BoostedSamplingSimulator(d=dim)
        result = simulator.execute(program=program, shots=shots)
def verify_timing(nmax, photons, shots=10): #shots=None for repeated row/column testing
  ERRBOUND = 1e-8 #1e-10
  testFuncs = testPermFuncs if shots is None else testSamplingFuncs
  dfeFuncs = dfePermFuncs if shots is None else dfeSamplingFuncs
  largeFuncs = largePermFuncs if shots is None else samplingFuncs
  suffix = str(photons) + ("" if shots is None else ("-" + str(shots)))
  verdata = "repverifydata" + suffix + ".bin"
  resdata = "represultdata" + suffix + ".bin"
  xaxis = list(range(nmax+1))
  gen_test_data = load_test_data()
  import os, pickle, timeit
  if os.path.isfile(os.path.join(saveFolder, verdata)):
    with open(os.path.join(saveFolder, verdata), "rb") as f:
      res = pickle.load(f)
  else: res = {}
  if os.path.isfile(os.path.join(saveFolder, resdata)):
    with open(os.path.join(saveFolder, resdata), "rb") as f:
      results = pickle.load(f)
  else: results = {}
  for key in gen_test_data:
    A = gen_test_data[key]
    if not key in res: res[key] = {}
    if not key in results: results[key] = {}
    for func in testFuncs:
      if func.__name__ in res[key]: del res[key][func.__name__]
    for func in largeFuncs:
      if not func.__name__ in res[key]: res[key][func.__name__] = []
      if not func.__name__ in results[key]: results[key][func.__name__] = []
      print("Verifying and Testing", func.__name__, "Photons", photons)
      for dim in xaxis:
        if func in dfeFuncs and dim == 0 or dim == 1 and not func in dfeFuncs:
          print("Initialization time", func.__name__, timeit.timeit(lambda: func(A[dim][0], A[dim][1][photons], A[dim][2][photons]) if shots is None else func(A[dim][0], A[dim][1][photons], shots), number=1))
        if len(res[key][func.__name__]) <= dim or len(results[key][func.__name__]) <= dim or func in dfeFuncs or True:
          mplier = 1#5 if photons < 8 else 1
          v = [None]
          #if func in dfeFuncs: print(check_power())
          print(A[dim][1][photons], A[dim][2][photons])
          def save_result():
              #v[0] = func(A[dim][0], [A[dim][1][photons]], [[A[dim][2][photons] for _ in range(3)]])
              if shots is None: v[0] = func(A[dim][0], [[A[dim][1][photons] for _ in range(3)]], [A[dim][2][photons]]) #v[0] = func(A[dim][0], A[dim][1][photons], A[dim][2][photons])
              else: v[0] = func(A[dim][0], A[dim][1][photons], shots)
          r = timeit.timeit(save_result, number=mplier) / mplier #v[0] = func(A[dim])
          #if func in dfeFuncs: print(check_power())
          if len(res[key][func.__name__]) <= dim: res[key][func.__name__].append(v[0])
          else: res[key][func.__name__][dim] = v[0]
          if len(results[key][func.__name__]) <= dim: results[key][func.__name__].append(r)
          else: results[key][func.__name__][dim] = r
          if dim < 24: print(photons, dim, v[0], r, A[dim][1][photons], A[dim][2][photons] if shots is None else None)
          with open(os.path.join(saveFolder, verdata), "wb") as f:
            pickle.dump(res, f)
          with open(os.path.join(saveFolder, resdata), "wb") as f:
            pickle.dump(results, f)
        if dim >= 24: print(photons, dim, func.__name__, res[key][func.__name__][dim], results[key][func.__name__][dim], A[dim][1][photons], A[dim][2][photons] if shots is None else None)
    if shots is None:
      with open(os.path.join(saveFolder, "repverifydata.csv"), "w") as f:
          import csv
          writer = csv.writer(f, delimiter='\t')
          writer.writerow(["Absolute Error compared to " + largeFuncs[0].__name__]) 
          writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
          writer.writerows([[i] + [abs(res[key][largeFuncs[0].__name__][i] - res[key][x.__name__][i]) for x in largeFuncs] for i in xaxis])
          writer.writerow(["Relative Error compared to " + largeFuncs[0].__name__]) 
          writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
          writer.writerows([[i] + [abs(res[key][largeFuncs[0].__name__][i] - res[key][x.__name__][i]) / abs(res[key][largeFuncs[0].__name__][i]) for x in largeFuncs] for i in xaxis])
          writer.writerow(["Permanent Computation Raw Results"])
          writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
          writer.writerows([[i] + [res[key][x.__name__][i] for x in largeFuncs] for i in xaxis])
          for dim in range(nmax+1):
            writer.writerow(["Random Unitary Test Matrix " + str(dim) + "x" + str(dim)])
            if dim != 0: writer.writerow([""] + [str(j) for j in range(dim)])
            for i in range(dim):
              writer.writerow([i] + [A[dim][0][i][j] for j in range(dim)])
            writer.writerow(["Random multinomial distributed input states"])
            for i in range(dim): writer.writerow([i, *A[dim][1][i]])
            writer.writerow(["Random multinomial distributed output states"])
            for i in range(dim): writer.writerow([i, *A[dim][2][i]])
    with open(os.path.join(saveFolder, "represultdata" + suffix + ".csv"), "w") as f:
      import csv
      writer = csv.writer(f, delimiter='\t')
      writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
      writer.writerows([[i] + [results[key][x.__name__][i] for x in largeFuncs] for i in xaxis])

    for i in xaxis:
      #assert all(abs(res[key][largeFuncs[0].__name__][i] - res[key][x][i]) < ERRBOUND for x in res[key] if x != largeFuncs[0].__name__)
      failures = [(i, x, res[key][x][i], res[key][largeFuncs[0].__name__][i], abs((res[key][largeFuncs[0].__name__][i] - res[key][x][i]) / abs(res[key][largeFuncs[0].__name__][i]))) for x in (y.__name__ for y in largeFuncs) if x != largeFuncs[0].__name__ and abs((res[key][largeFuncs[0].__name__][i] - res[key][x][i]) / abs(res[key][largeFuncs[0].__name__][i])) > ERRBOUND]
      if len(failures) != 0: print("ACCURACY FAILURES: ", failures); assert False, failures
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    verinfo = None if not shots is None else ([(f, [abs(res[key][largeFuncs[0].__name__][i] - res[key][f.__name__][i]) / abs(res[key][largeFuncs[0].__name__][i]) for i in xaxis]) for f in largeFuncs[1:]], "repglynnpermacc", "Accuracy relative to " + largeFuncs[0].__name__ + " (log10)")
    timeinfo = ([(f, [results[key][f.__name__][i] for i in xaxis]) for f in largeFuncs], "repglynnpermtime" + suffix, "Time (log10 s)")
    for vals, fname, ylbl in ((verinfo, timeinfo) if shots is None else (timeinfo,)):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        markers = ['o', '*', 'x', '+', 's', 'p', '1', '2', '3', '4']
        lines = []        
        for i, val in enumerate(vals):
          lines.append(ax1.plot(xaxis, val[1], label=val[0].__name__, marker=markers[i], linestyle=' '))
        ax1.set_xlabel("Size (n=|A|)")  
        ax1.set_yscale('log', base=10)
        ax1.set_ylabel(ylbl)
        ax1.legend()
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_title(("Repeated Permanent of Square Matrix A" if shots is None else "Boson Sampling of Interferometer Matrix A") + " Photons=" + str(photons) + ("" if shots is None else (" Shots=" + str(shots))))
        fig.savefig(os.path.join(saveFolder, fname + ".svg"), format="svg")
        import tikzplotlib #pip install tikzplotlib
        #python3 -c "import tikzplotlib; print(tikzplotlib.Flavors.latex.preamble())"
        for line in lines:
            for z in line: z.set_label(z.get_label().replace("_", "\\_")) #fix bug with underscore in tikzplotlib legend label escaping
        ax1.legend()
        tikzplotlib.save(os.path.join(saveFolder, fname + ".tex"))
        plt.close(fig)
#other_stability(40, 30)
#stability(40, 30)
for i in range(19 if hasSim else 0, 40+1):
    verify_timing(DEPTH, i, None)
#verify_timing(DEPTH, 10, None)
#verify_timing(30, 10, 10)
#verify_timing(DEPTH, 20, 10)
