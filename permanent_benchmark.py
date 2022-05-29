#pip install numpy, scipy, piquasso, thewalrus, matplotlib, tikzplotlib
import numpy as np
#from thewalrus.libwalrus import perm_complex, perm_real, perm_BBFG_real, perm_BBFG_complex
import thewalrus
if thewalrus.version() in ("0.7.0", "0.8.0", "0.10.0", "0.11.0", "0.12.0", "0.13.0rc1", "0.13.0", "0.14.0", "0.15.0", "0.15.1", "0.16.0", "0.16.1", "0.16.2", "0.17.0"):
    from thewalrus import perm_complex, perm_real, perm_BBFG_real, perm_BBFG_complex
else:
    from thewalrus import perm
    def perm_complex(A, quad): return perm(A, "ryser")
    def perm_BBFG_complex(A): return perm(A, "bbfg")
from piquassoboost.sampling.Boson_Sampling_Utilities import GlynnPermanent, GlynnPermanentInf, GlynnPermanentSingleDFE, GlynnPermanentDualDFE, GlynnPermanentSingleDFEF, GlynnPermanentDualDFEF, GlynnPermanentDoubleCPU, BBFGPermanentDouble, BBFGPermanentLongDouble
import piquasso as pq
import random
from scipy.stats import unitary_group

import time

def checkSim():
  import os
  return 'SLIC_CONF' in os.environ #'MAXELEROSDIR'
hasSim = checkSim(); hasDFE = not hasSim

DEPTH = 12 if hasSim else 40
saveFolder = "results"

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

def check_power():
  import subprocess, re
  result = subprocess.run(['maxtop', '-v'], stdout=subprocess.PIPE)
  return re.findall(r'^\tPower usage: (.*)$', result.stdout.decode('utf-8'), re.MULTILINE)   

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
def permanent_glynn_rectangular(mat):
  #Gray code order of deltas would yield reduction from n^2 to n similar to Ryser
  n = len(mat); m = len(mat[0])
  if n == 0: return 1
  #return sum(dosign((sum(x < 0 for x in delta) & 1) != 0, multiprod((sum(delta[i] * mat[i][j] for i in range(n)) for j in range(n)))) for delta in [[1] + x for x in getDeltas(n-1)]) >> (n-1)
  return sum(dosign((sum(delta) & 1) != 0, multiprod((mat[0][j] + sum(dosign(delta[i-1]!=0, mat[i][j]) for i in range(1, n)) for j in range(m)))) for delta in getDeltas(n-1)) / (1 << (n-1))
"""
  (64, -62) * (64, -62) = (128, -124)
"""
def permanent_glynn_gray_fixpt(mat): #optimal row-major order
  mat = np.copy(mat).transpose()
  n = len(mat)
  if n == 0: return 1
  #additions: n(n-1)+2^(n-1)-1 multiplications: n-1
  #FixOpModeDefault is bitSizeLargest and offsetLargestMsb for all of ADD, ADD_SUB, ALL, DIV, MUL, MUL_DIV, NEG, SUB 
  #RoundingMode default is TONEAR of (TRUNCATE, TONEAR, TONEAREVEN) == in decimal (ROUND_FLOOR, ROUND_HALF_CEILING, ROUND_HALF_EVEN)
  #IEEE754 default rounding mode is Round to nearest, ties to even FE_TONEAREST in GCC, ROUND_HALF_EVEN in decimal
  def pairAdd(x, y): return (x[0] + y[0], x[1] + y[1])
  def pairSub(x, y): return (x[0] - y[0], x[1] - y[1])
  def roundToNear(x, prec):
    return (x + (1 << (prec-1))) >> prec
  def pairProd(x, y, prec): return (roundToNear(x[0] * y[0], prec) - roundToNear(x[1] * y[1], prec), roundToNear(x[0] * y[1], prec) + roundToNear(x[1] * y[0], prec))
  def pairProdTree(l):
    #l = [pairScalarMul(x, 2**62) for x in l]
    l = [x for x in l]
    limit, lastprec, extraprec = len(l), 0, 2
    while limit > 1:
      limitNew = limit // 2
      for idx in range(limitNew):
        l[idx] = pairProd(l[2*idx], l[2*idx+1], lastprec*2+62-extraprec) #62+extraprec accuracy while current accuracy after multiply is lastprec*2
      if 2 * limitNew < limit:
        limitNew += 1
        l[limitNew-1] = pairScalarMul(l[limit-1], 2**(extraprec - lastprec))
      limit = limitNew
      lastprec = extraprec
      extraprec *= 2
      if extraprec > len(l): extraprec = len(l)
    return pairScalarMul(l[0], 2**(62-lastprec)) #l[0]
  def pairScalarMul(x, y): return (x[0] * y, x[1] * y)
  def pairScalarDiv(x, y): return (x[0] / y, x[1] / y)
  def pairPlusMinus(parity, base, x): return pairSub(base, x) if parity else pairAdd(base, x)
  def pairToCplxFloat(x): return float(x[0]) + float(x[1]) * 1j
  #longdouble is just a padded 80-bit hardware long double
  renormalize = [np.clongdouble(0) for _ in range(n)]
  for i in range(n):
    for j in range(n):
      value1, value2 = renormalize[i]+np.clongdouble(mat[i][j]), renormalize[i]-np.clongdouble(mat[i][j])
      renormalize[i] = value1 if abs(value1) > abs(value2) else value2
  renormalize = [np.abs(x) for x in renormalize] #np.sqrt(x.real*x.real+x.imag*x.imag)
  #print([x.tobytes().hex() for x in renormalize])
  mat = [[(round(np.longdouble(mat[i][j].real)*np.longdouble(2**62) / renormalize[i]), round(np.longdouble(mat[i][j].imag)*np.longdouble(2**62) / renormalize[i])) for i in range(n)] for j in range(n)]
  #print([[(hex(mat[i][j][0]), hex(mat[i][j][1])) for i in range(n)] for j in range(n)])  
  delta, rowsums = [1 for _ in range(n)], [reduc(mat[i], pairAdd) for i in range(n)] #[0 for _ in range(n)]
  #print(mat, rowsums)
  tot = pairProdTree(rowsums)
  for i in range(1, 1<<(n-1)):
    idx = nextGrayCode(delta, i)
    if delta[idx]:
      for j in range(n): rowsums[j] = pairAdd(rowsums[j], pairScalarMul(mat[j][idx], 2)) #mat[j][idx]
    else:
      for j in range(n): rowsums[j] = pairSub(rowsums[j], pairScalarMul(mat[j][idx], 2)) #mat[j][idx]
    tot = pairPlusMinus((i & 1) != 0, tot, pairProdTree(rowsums))
  tot = (np.longdouble(tot[0]), np.longdouble(tot[1]))
  tot = pairScalarDiv(pairScalarDiv(tot, np.longdouble(2**62)), np.longdouble(2**62))
  tot = pairScalarDiv(tot, (1 << (n-1)))
  for x in renormalize:
    tot = pairScalarMul(tot, x)
  return pairToCplxFloat(tot) #tot
def reduc(l, f):
  import functools
  return functools.reduce(f, l)
def permanent_glynn_gray_exact(mat): #optimal row-major order
  n = len(mat)
  from decimal import Decimal, Context, ROUND_DOWN
  ctxt = Context(prec=256, rounding=ROUND_DOWN)
  def decPairAdd(x, y): return (ctxt.add(x[0], y[0]), ctxt.add(x[1], y[1]))
  def decPairSub(x, y): return (ctxt.subtract(x[0], y[0]), ctxt.subtract(x[1], y[1]))
  def decPairProd(x, y): return (ctxt.subtract(ctxt.multiply(x[0], y[0]), ctxt.multiply(x[1], y[1])), ctxt.add(ctxt.multiply(x[0], y[1]), ctxt.multiply(x[1], y[0])))
  def decPairScalarMul(x, y): return (ctxt.multiply(x[0], y), ctxt.multiply(x[1], y))
  def decPairScalarDiv(x, y): return (ctxt.divide(x[0], y), ctxt.divide(x[1], y))
  def decPairPlusMinus(parity, base, x): return decPairSub(base, x) if parity else decPairAdd(base, x)
  def decPairToCplxFloat(x): return float(x[0]) + float(x[1]) * 1j
  mat = [[(ctxt.create_decimal_from_float(mat[i][j].real), ctxt.create_decimal_from_float(mat[i][j].imag)) for i in range(n)] for j in range(n)]  
  if n == 0: return 1
  #additions: n(n-1)+2^(n-1)-1 multiplications: n-1
  delta, rowsums = [1 for _ in range(n)], [reduc(mat[i], decPairAdd) for i in range(n)] #[0 for _ in range(n)]
  #print(mat, rowsums)
  tot = reduc(rowsums, decPairProd)
  for i in range(1, 1<<(n-1)):
    idx = nextGrayCode(delta, i)
    if delta[idx]:
      for j in range(n): rowsums[j] = decPairAdd(rowsums[j], decPairScalarMul(mat[j][idx], 2)) #mat[j][idx]
    else:
      for j in range(n): rowsums[j] = decPairSub(rowsums[j], decPairScalarMul(mat[j][idx], 2)) #mat[j][idx]
    tot = decPairPlusMinus((i & 1) != 0, tot, reduc(rowsums, decPairProd))
  return decPairToCplxFloat(decPairScalarDiv(tot, (1 << (n-1)))) #tot


calculators = [None, None, None, None, None, None, None, None, None]
def batch_adapter(Arep, f):
    if isinstance(Arep, list): return [f(A) for A in Arep]
    else: return f(Arep)

#https://github.com/XanaduAI/thewalrus/issues/319 - 0 case bugged in Ryser/BBFG
def permanent_walrus_quad_Ryser(Arep):
    def f(Arep):
        return 1+0j if len(Arep) == 0 else perm_complex(Arep, quad=True) #2*permanent_Glynn_Cpp
    return batch_adapter(Arep, f)
def permanent_walrus_quad_BBFG(Arep):
    def f(Arep):
        return 1+0j if len(Arep) == 0 else perm_BBFG_complex(Arep) #ChinHuh, 2^6*Glynn_Cpp, 2^5*walrus_quad_Ryser
    return batch_adapter(Arep, f)
def permanent_Glynn_Cpp(Arep):
    if calculators[0] is None: calculators[0] = GlynnPermanent(Arep)
    else: calculators[0].matrix = Arep
    return calculators[0].calculate()
def permanent_Glynn_Cpp_Inf(Arep):
    if calculators[1] is None: calculators[1] = GlynnPermanentInf(Arep)
    else: calculators[1].matrix = Arep
    return calculators[1].calculate()
def permanent_Glynn_SIM(Arep): return permanent_Glynn_DFE(Arep)
def permanent_Glynn_SIMDual(Arep): return permanent_Glynn_DFEDual(Arep)
def permanent_Glynn_SIMF(Arep): return permanent_Glynn_DFEF(Arep)
def permanent_Glynn_SIMFDual(Arep): return permanent_Glynn_DFEFDual(Arep)
def permanent_Glynn_DFE(Arep):
    if calculators[2] is None: calculators[2] = GlynnPermanentSingleDFE(Arep)
    else: calculators[2].matrix = Arep
    return calculators[2].calculate()
def permanent_Glynn_DFEDual(Arep):
    if calculators[3] is None: calculators[3] = GlynnPermanentDualDFE(Arep)
    else: calculators[3].matrix = Arep
    return calculators[3].calculate()
def permanent_Glynn_DFEF(Arep):
    if calculators[4] is None: calculators[4] = GlynnPermanentSingleDFEF(Arep)
    else: calculators[4].matrix = Arep
    return calculators[4].calculate()
def permanent_Glynn_DFEFDual(Arep):
    if calculators[5] is None: calculators[5] = GlynnPermanentDualDFEF(Arep)
    else: calculators[5].matrix = Arep
    return calculators[5].calculate()
def permanent_Glynn_Cpp_Double(Arep):
    if calculators[6] is None: calculators[6] = GlynnPermanentDoubleCPU(Arep)
    else: calculators[6].matrix = Arep
    return calculators[6].calculate()
def permanent_BBFG_Double(Arep):
    if calculators[7] is None: calculators[7] = BBFGPermanentDouble(Arep)
    else: calculators[7].matrix = Arep
    return calculators[7].calculate()
def permanent_BBFG_LongDouble(Arep):
    if calculators[8] is None: calculators[8] = BBFGPermanentLongDouble(Arep)
    else: calculators[8].matrix = Arep
    return calculators[8].calculate()

dfePermFuncs = ((permanent_Glynn_SIMF, permanent_Glynn_SIMFDual, permanent_Glynn_SIM, permanent_Glynn_SIMDual) if hasSim else (permanent_Glynn_DFE, permanent_Glynn_DFEDual))
largePermFuncs = (permanent_Glynn_Cpp_Inf, permanent_BBFG_Double, permanent_BBFG_LongDouble, permanent_Glynn_Cpp, permanent_Glynn_Cpp_Double, permanent_walrus_quad_Ryser) + dfePermFuncs
testPermFuncs = (permanent_glynn, permanent_glynn_gray_fixpt, permanent_glynn_gray_exact, permanent_walrus_quad_BBFG)
permFuncs = testPermFuncs + largePermFuncs

def load_test_data():
  nmax = 40
  randfuncs = (unitary_group.rvs, )#generate_random_unitary):
  import os, pickle
  if not os.path.isdir(saveFolder): os.mkdir(saveFolder)
  if os.path.isfile(os.path.join(saveFolder, "matrices.bin")):
    with open(os.path.join(saveFolder, "matrices.bin"), "rb") as f: 
      gen_test_data = pickle.load(f)
  else:
    # generate the random matrix
    gen_test_data = {rf.__name__:{dim:np.random.random((dim, dim))+np.random.random((dim, dim))*1j if dim <= 1 else rf(dim) for dim in range(nmax+1)} for rf in randfuncs}
    with open(os.path.join(saveFolder, "matrices.bin"), "wb") as f:
      pickle.dump(gen_test_data, f)
  return gen_test_data
def verify_timing(nmax, batchsize=1):
  ERRBOUND = 1e-6
  suffix = "" if batchsize == 1 else str(batchsize)
  verdata = "verifydata.bin"
  resdata = "resultdata" + suffix + ".bin"
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
    for func in testPermFuncs:
      if func.__name__ in res[key]: del res[key][func.__name__]
    for func in largePermFuncs:
      if not func.__name__ in res[key]: res[key][func.__name__] = []
      if not func.__name__ in results[key]: results[key][func.__name__] = []
      print("Verifying and Testing", func.__name__)
      for dim in xaxis:
        #if func in dfePermFuncs and dim == 0 or dim == 1 and not func in dfePermFuncs:
        #  print("Initialization time", func.__name__, timeit.timeit(lambda: func(A[dim]), number=1))
        if len(res[key][func.__name__]) <= dim or len(results[key][func.__name__]) <= dim or func in dfePermFuncs or func == permanent_BBFG_LongDouble:
          mplier = 5 if dim < 24 else 1
          v = [None]
          #if func in dfePermFuncs: print(check_power())
          #def save_result():
          #    v[0] = func(A[dim])
          def save_result():
              if batchsize == 1: v[0] = func(A[dim])
              else: v[0] = func([A[dim]] * batchsize)
          r = timeit.timeit(save_result, number=mplier) / mplier #v[0] = func(A[dim])
          #if func in dfePermFuncs: print(check_power())
          if batchsize != 1:
            assert all(abs(res[key][func.__name__][dim] - x) < ERRBOUND for x in v[0])
            v[0] = v[0][0]
          else:
            if len(res[key][func.__name__]) <= dim: res[key][func.__name__].append(v[0])
            else: res[key][func.__name__][dim] = v[0]
          if len(results[key][func.__name__]) <= dim: results[key][func.__name__].append(r)
          else: results[key][func.__name__][dim] = r
          if dim < 24: print(dim, v[0], r)
          if batchsize == 1:
            with open(os.path.join(saveFolder, verdata), "wb") as f:
              pickle.dump(res, f)
          with open(os.path.join(saveFolder, resdata), "wb") as f:
            pickle.dump(results, f)        
        if dim >= 24: print(dim, func.__name__, res[key][func.__name__][dim], results[key][func.__name__][dim])
    if batchsize == 1:
      with open(os.path.join(saveFolder, "verifydata.csv"), "w") as f:
          import csv
          writer = csv.writer(f, delimiter='\t')
          writer.writerow(["Absolute Error compared to " + largePermFuncs[0].__name__]) 
          writer.writerow(["Size (n)"] + [f.__name__ for f in largePermFuncs])
          writer.writerows([[i] + [abs(res[key][largePermFuncs[0].__name__][i] - res[key][x.__name__][i]) for x in largePermFuncs] for i in xaxis])
          writer.writerow(["Relative Error compared to " + largePermFuncs[0].__name__]) 
          writer.writerow(["Size (n)"] + [f.__name__ for f in largePermFuncs])
          writer.writerows([[i] + [abs(res[key][largePermFuncs[0].__name__][i] - res[key][x.__name__][i]) / abs(res[key][largePermFuncs[0].__name__][i]) for x in largePermFuncs] for i in xaxis])
          writer.writerow(["Permanent Computation Raw Results"])
          writer.writerow(["Size (n)"] + [f.__name__ for f in largePermFuncs])
          writer.writerows([[i] + [res[key][x.__name__][i] for x in largePermFuncs] for i in xaxis])
          for dim in range(nmax+1):
            writer.writerow(["Random Unitary Test Matrix " + str(dim) + "x" + str(dim)])
            if dim != 0: writer.writerow([""] + [str(j) for j in range(dim)])
            for i in range(dim):
              writer.writerow([i] + [A[dim][i][j] for j in range(dim)])
    with open(os.path.join(saveFolder, "resultdata" + suffix + ".csv"), "w") as f:
      import csv
      writer = csv.writer(f, delimiter='\t')
      writer.writerow(["Size (n)"] + [f.__name__ for f in largePermFuncs])
      writer.writerows([[i] + [results[key][x.__name__][i] for x in largePermFuncs] for i in xaxis])

    for i in xaxis:
      #assert all(abs(res[key][largePermFuncs[0].__name__][i] - res[key][x][i]) < ERRBOUND for x in res[key] if x != largePermFuncs[0].__name__)
      failures = [(i, x, res[key][x][i], res[key][largePermFuncs[0].__name__][i], abs((res[key][largePermFuncs[0].__name__][i] - res[key][x][i]) / abs(res[key][largePermFuncs[0].__name__][i]))) for x in (y.__name__ for y in largePermFuncs) if x != largePermFuncs[0].__name__ and abs((res[key][largePermFuncs[0].__name__][i] - res[key][x][i]) / abs(res[key][largePermFuncs[0].__name__][i])) > ERRBOUND]
      if len(failures) != 0: print("ACCURACY FAILURES: ", failures); assert False, failures

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    verinfo = ([(f, [abs(res[key][largePermFuncs[0].__name__][i] - res[key][f.__name__][i]) / abs(res[key][largePermFuncs[0].__name__][i]) for i in xaxis]) for f in largePermFuncs[1:]], "glynnpermacc", "Accuracy relative to " + largePermFuncs[0].__name__ + " (log10)")
    timeinfo = ([(f, [results[key][f.__name__][i] for i in xaxis]) for f in largePermFuncs], "glynnpermtime" + suffix, "Time (log10 s)")
    for vals, fname, ylbl in ((verinfo, timeinfo) if batchsize == 1 else (timeinfo,)):
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
        ax1.set_title("Permanent Computation of Square Matrix A" + ("" if batchsize==1 else (" Batch=" + str(batchsize))))
        fig.savefig(os.path.join(saveFolder, fname + ".svg"), format="svg")
        import tikzplotlib #pip install tikzplotlib
        #python3 -c "import tikzplotlib; print(tikzplotlib.Flavors.latex.preamble())"
        for line in lines:
            for z in line: z.set_label(z.get_label().replace("_", "\\_")) #fix bug with underscore in tikzplotlib legend label escaping
        ax1.legend()
        tikzplotlib.save(os.path.join(saveFolder, fname + ".tex"))
        plt.close(fig)
# (a b) (c d) (e f) (g h) = (a+c+e+g)(b+d+f+h)-(a+c+e-g)(b+d+f-h)+(a+c-e-g)(b+d-f-h)-(a-c-e-g)(b-d-f-h)+(a-c-e+g)(b-d-f+h)-(a-c+e+g)(b-d+f+h)+(a-c+e-g)(b-d+f-h)-(a+c-e+g)(b+d-f+h)==0 according to WolframAlpha
#nXm where m in [2..n-2] always is 0 due to cancellation of terms, however floating/fixed point cannot be relied upon in such cases
def verify_rectangular(nmax):
    for x in range(0, nmax+1):
        for y in range(0, nmax+1):
            #A = [np.random.random((x, y))+np.random.random((x, y))*1j for _ in range(10000)]
            A = np.random.random((x, y))+np.random.random((x, y))*1j
            r = permanent_Glynn_Cpp_Inf(A)
            res1, res2 = permanent_Glynn_Cpp(A), permanent_BBFG_LongDouble(A)# permanent_Glynn_DFE(A)
            if x == 0 or y == 0: assert res1 == 1 and res2 == 1 and r == 1, (res1, res2, r)
            elif x >= y + 2: assert res1 == 0 and res2 == 0 and r == 0, (res1, res2, r)
            #assert all((abs(r1-r2) / abs(r1)) <= 1e-10 for r1, r2 in zip(res1, res2)), (x, y, r1, r2)
            else: assert (abs(res1-res2) / abs(res1)) <= 1e-10, (x, y, res1, res2)
            #print(x, y, r1, r2, "None" if abs(r1)==0 else (abs(r1-r2) / abs(r1)) <= 1e-10)
            #if y>=2 and y <= x-2:
#verify_rectangular(20)
verify_timing(DEPTH, 1)
#for batch_size in ((2,) if hasSim else (2, 3, 4, 5, 10, 20, 25, 50, 100)):
#    verify_timing(DEPTH if hasSim else 22, batch_size)
