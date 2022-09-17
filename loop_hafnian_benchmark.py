
import numpy as np

from piquassoboost.sampling.Boson_Sampling_Utilities import PowerTraceHafnian, PowerTraceHafnianRecursive, PowerTraceLoopHafnian, PowerTraceLoopHafnianRecursive, PowerTraceHafnianDouble, PowerTraceHafnianLongDouble, PowerTraceHafnianInf, PowerTraceLoopHafnianDouble, PowerTraceLoopHafnianLongDouble, PowerTraceLoopHafnianInf, PowerTraceHafnianRecursiveDouble, PowerTraceHafnianRecursiveLongDouble, PowerTraceHafnianRecursiveInf, PowerTraceLoopHafnianRecursiveDouble, PowerTraceLoopHafnianRecursiveLongDouble, PowerTraceLoopHafnianRecursiveInf
from thewalrus import hafnian, hafnian_repeated
from scipy.stats import unitary_group

DEPTH=52
saveFolder = "resultslh"
def make_symmetric(A): return A + A.T
def load_test_data():
  nmax = 80
  randfuncs = (unitary_group.rvs, )#generate_random_unitary):
  import os, pickle
  if not os.path.isdir(saveFolder): os.mkdir(saveFolder)
  if os.path.isfile(os.path.join(saveFolder, "matrices.bin")):
    with open(os.path.join(saveFolder, "matrices.bin"), "rb") as f: 
      gen_test_data = pickle.load(f)
  else:
    # generate the random matrix
    gen_test_data = {rf.__name__:{dim:make_symmetric(np.random.random((dim, dim))+np.random.random((dim, dim))*1j if dim <= 1 else rf(dim)) for dim in range(nmax+1)} for rf in randfuncs}
    with open(os.path.join(saveFolder, "matrices.bin"), "wb") as f:
      pickle.dump(gen_test_data, f)
  return gen_test_data
  
#functools.reduce(lambda x, y: x*y//math.gcd(x, y), range(1, 80)).bit_length()
#functools.reduce(lambda x, y: x*y//math.gcd(x, y), [math.factorial(x) for x in range(1, 40)]).bit_length()

hcalculators = [None, None, None, None, None, None, None, None]
calculators = [None, None, None, None, None, None, None, None]

def to_repeated(f):
    return f(Arep, np.array([1]*(len(Arep)//2), dtype=np.int64))
def hafnian_walrus(Arep):
    return hafnian(Arep, loop=False)
def hafnian_repeated_walrus(Arep, rep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    return hafnian_repeated(Arep, np.hstack((rep[:,np.newaxis], rep[:,np.newaxis])).reshape((len(rep)*2,)), loop=False)
def hafnian_powertrace(Arep):
    if hcalculators[0] is None: hcalculators[0] = PowerTraceHafnian(Arep)
    else: hcalculators[0].matrix = Arep
    return hcalculators[0].calculate()
def hafnian_powertrace_recursive(Arep, rep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    #if calculators[1] is None:
    hcalculators[1] = PowerTraceHafnianRecursive(Arep, rep)
    #else: hcalculators[1].matrix = Arep; hcalculators[1].occupancy = np.array([1]*(len(Arep)//2), dtype=np.int64)
    return hcalculators[1].calculate()
def hafnian_powertrace_double(Arep):
    if hcalculators[2] is None: hcalculators[2] = PowerTraceHafnianDouble(Arep)
    else: hcalculators[2].matrix = Arep
    return hcalculators[2].calculate()
def hafnian_powertrace_longdouble(Arep):
    if hcalculators[3] is None: hcalculators[3] = PowerTraceHafnianLongDouble(Arep)
    else: hcalculators[3].matrix = Arep
    return hcalculators[3].calculate()
def hafnian_powertrace_inf(Arep):
    if hcalculators[4] is None: hcalculators[4] = PowerTraceHafnianInf(Arep)
    else: hcalculators[4].matrix = Arep
    return hcalculators[4].calculate()
def hafnian_powertrace_recursive_double(Arep, rep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    hcalculators[5] = PowerTraceHafnianRecursiveDouble(Arep, rep)
    return hcalculators[5].calculate()
def hafnian_powertrace_recursive_longdouble(Arep, rep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    hcalculators[6] = PowerTraceHafnianRecursiveLongDouble(Arep, rep)
    return hcalculators[6].calculate()
def hafnian_powertrace_recursive_inf(Arep, rep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    hcalculators[7] = PowerTraceHafnianRecursiveInf(Arep, rep)
    return hcalculators[7].calculate()
    
def lhafnian_walrus(Arep):
    return hafnian(Arep, loop=True)
def lhafnian_repeated_walrus(Arep, rep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    return hafnian_repeated(Arep, np.hstack((rep[:,np.newaxis], rep[:,np.newaxis])).reshape((len(rep)*2,)), loop=True)
def lhafnian_powertrace(Arep):
    if calculators[0] is None: calculators[0] = PowerTraceLoopHafnian(Arep)
    else: calculators[0].matrix = Arep
    return calculators[0].calculate()
def lhafnian_powertrace_recursive(Arep, rep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    #if calculators[1] is None:
    calculators[1] = PowerTraceLoopHafnianRecursive(Arep, rep)
    #else: calculators[1].matrix = Arep; calculators[1].occupancy = np.array([1]*(len(Arep)//2), dtype=np.int64)
    return calculators[1].calculate()
def lhafnian_powertrace_double(Arep):
    if calculators[2] is None: calculators[2] = PowerTraceLoopHafnianDouble(Arep)
    else: calculators[2].matrix = Arep
    return calculators[2].calculate()
def lhafnian_powertrace_longdouble(Arep):
    if calculators[3] is None: calculators[3] = PowerTraceLoopHafnianLongDouble(Arep)
    else: calculators[3].matrix = Arep
    return calculators[3].calculate()
def lhafnian_powertrace_inf(Arep):
    if calculators[4] is None: calculators[4] = PowerTraceLoopHafnianInf(Arep)
    else: calculators[4].matrix = Arep
    return calculators[4].calculate()
def lhafnian_powertrace_recursive_double(Arep, rep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    calculators[5] = PowerTraceLoopHafnianRecursiveDouble(Arep, rep)
    return calculators[5].calculate()
def lhafnian_powertrace_recursive_longdouble(Arep, rep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    calculators[6] = PowerTraceLoopHafnianRecursiveLongDouble(Arep, rep)
    return calculators[6].calculate()
def lhafnian_powertrace_recursive_inf(Arep, rep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    calculators[7] = PowerTraceLoopHafnianRecursiveInf(Arep, rep)
    return calculators[7].calculate()

largeLoopHafnianRepFuncs = (lhafnian_powertrace_recursive_inf, lhafnian_powertrace_recursive, lhafnian_powertrace_recursive_double, lhafnian_powertrace_recursive_longdouble, lhafnian_repeated_walrus)
largeLoopHafnianFuncs = (lhafnian_powertrace_inf, lhafnian_powertrace, lhafnian_powertrace_double, lhafnian_powertrace_longdouble, lhafnian_walrus)
largeHafnianRepFuncs = (hafnian_powertrace_recursive_inf, hafnian_powertrace_recursive, hafnian_powertrace_recursive_double, hafnian_powertrace_recursive_longdouble, hafnian_repeated_walrus) 
largeHafnianFuncs = (hafnian_powertrace, hafnian_powertrace_double, hafnian_powertrace_longdouble, hafnian_walrus) #(hafnian_powertrace_inf, hafnian_powertrace, hafnian_powertrace_double, hafnian_powertrace_longdouble, hafnian_walrus) 
testLoopHafnianFuncs = ()
testHafnianFuncs = ()

paperNames = {lhafnian_powertrace: "PiquassoBoost", lhafnian_powertrace_double: "PiquassoBoost Double", lhafnian_powertrace_longdouble: "PiquassoBoost LongDouble",
              lhafnian_powertrace_recursive: "PiquassoBoost Recursive Hybrid", lhafnian_powertrace_recursive_double: "PiquassoBoost Recursive Double", lhafnian_powertrace_recursive_longdouble: "PiquassoBoost Recursive LongDouble",
              lhafnian_powertrace_inf: "MPFR Inf", lhafnian_powertrace_recursive_inf: "MPFR Inf Recursive",
              lhafnian_walrus: "thewalrus", lhafnian_repeated_walrus: "thewalrus repeated", 
              hafnian_powertrace: "PiquassoBoost Hybrid", hafnian_powertrace_double: "PiquassoBoost Double", hafnian_powertrace_longdouble: "PiquassoBoost LongDouble", 
              hafnian_powertrace_recursive: "PiquassoBoost Recursive Hybrid", hafnian_powertrace_recursive_double: "PiquassoBoost Recursive Double", hafnian_powertrace_recursive_longdouble: "PiquassoBoost Recursive LongDouble",
              hafnian_walrus: "thewalrus", hafnian_repeated_walrus: "thewalrus repeated", hafnian_powertrace_inf: "MPFR Inf", hafnian_powertrace_recursive_inf: "MPFR Inf Recursive"}

def verify_timing(nmax, batchsize=1, loop=True, repeated=False):
  ERRBOUND = 1e-6
  largeFuncs = (largeLoopHafnianRepFuncs if loop else largeHafnianRepFuncs) if repeated else (largeLoopHafnianFuncs if loop else largeHafnianFuncs)
  suffix = "" if batchsize == 1 else str(batchsize)
  prefix = ("loophaf" if loop else "haf") + ("rep" if repeated else "")
  verdata = prefix + "verifydata.bin"
  resdata = prefix + "resultdata" + suffix + ".bin"
  xaxis = list(range(0, nmax+1, 1)) #list(range(0, nmax+1, 2))
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
    for func in (testLoopHafnianFuncs if loop else testHafnianFuncs):
      if func.__name__ in res[key]: del res[key][func.__name__]
    for func in largeFuncs:
      if not func.__name__ in res[key]: res[key][func.__name__] = []
      if not func.__name__ in results[key]: results[key][func.__name__] = []
      print("Verifying and Testing", func.__name__)
      for dim in xaxis:
        if len(res[key][func.__name__]) <= dim or len(results[key][func.__name__]) <= dim or True:
          mplier = 5 if dim < 24 and not func in (hafnian_powertrace_inf, lhafnian_powertrace_inf, hafnian_powertrace_recursive_inf, lhafnian_powertrace_recursive_inf) else 1
          v = [None]
          rep = np.array([2]*(dim//2), dtype=np.int64)
          def save_result():
              if batchsize == 1: v[0] = func(A[dim], rep) if repeated else func(A[dim])
              else: v[0] = func([A[dim]] * dim, rep) if repeated else func([A[dim]] * dim) #batchsize)
          r = timeit.timeit(save_result, number=mplier) / mplier #v[0] = func(A[dim])
          if batchsize != 1:
            assert all(abs(res[key][func.__name__][dim] - x) < ERRBOUND for x in v[0])
            if len(v[0]) != 0: v[0] = v[0][0]
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
      with open(os.path.join(saveFolder, prefix + "verifydata.csv"), "w") as f:
          import csv
          writer = csv.writer(f, delimiter='\t')
          writer.writerow(["Absolute Error compared to " + largeFuncs[0].__name__]) 
          writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
          writer.writerows([[i] + [abs(res[key][largeFuncs[0].__name__][i] - res[key][x.__name__][i]) for x in largeFuncs] for i in xaxis])
          writer.writerow(["Relative Error compared to " + largeFuncs[0].__name__]) 
          writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
          writer.writerows([[i] + [abs(res[key][largeFuncs[0].__name__][i] - res[key][x.__name__][i]) / abs(res[key][largeFuncs[0].__name__][i]) for x in largeFuncs] for i in xaxis if (i & 1) == 0])
          writer.writerow([("Loop " if loop else "") + "Hafnian Computation Raw Results"])
          writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
          writer.writerows([[i] + [res[key][x.__name__][i] for x in largeFuncs] for i in xaxis])
          for dim in range(nmax+1):
            writer.writerow(["Random Unitary Test Matrix " + str(dim) + "x" + str(dim)])
            if dim != 0: writer.writerow([""] + [str(j) for j in range(dim)])
            for i in range(dim):
              writer.writerow([i] + [A[dim][i][j] for j in range(dim)])
    with open(os.path.join(saveFolder, prefix + "resultdata" + suffix + ".csv"), "w") as f:
      import csv
      writer = csv.writer(f, delimiter='\t')
      writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
      writer.writerows([[i] + [results[key][x.__name__][i] for x in largeFuncs] for i in xaxis])

    for i in xaxis:
      if (i & 1) != 0: continue
      #assert all(abs(res[key][largeFuncs[0].__name__][i] - res[key][x][i]) < ERRBOUND for x in res[key] if x != largeFuncs[0].__name__)
      failures = [(i, x, res[key][x][i], res[key][largeFuncs[0].__name__][i], abs((res[key][largeFuncs[0].__name__][i] - res[key][x][i]) / abs(res[key][largeFuncs[0].__name__][i]))) for x in (y.__name__ for y in largeFuncs) if x != largeFuncs[0].__name__ and abs((res[key][largeFuncs[0].__name__][i] - res[key][x][i]) / abs(res[key][largeFuncs[0].__name__][i])) > ERRBOUND]
      if len(failures) != 0: print("ACCURACY FAILURES: ", failures); #assert False, failures

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import math
    plt.rcParams['text.usetex'] = True
    from matplotlib.ticker import MaxNLocator
    verinfo = ([(f, [abs(res[key][largeFuncs[0].__name__][i] - res[key][f.__name__][i]) / abs(res[key][largeFuncs[0].__name__][i]) for i in xaxis if (i & 1) == 0]) for f in largeFuncs[1:]], prefix + "acc", "Accuracy relative to " + paperNames[largeFuncs[0]] + " ($\\log_{10}$)")
    timeinfo = ([(f, [results[key][f.__name__][i] for i in xaxis if (i & 1) == 0]) for f in largeFuncs], prefix + "time" + suffix, "Time ($\\log_{10}$ s)")
    for vals, fname, ylbl in ((verinfo, timeinfo) if batchsize == 1 else (timeinfo,)):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        markers = ['o', '*', 'x', '+', 's', 'p', '1', '2', '3', '4', '8', 'P', 'h']
        lines = []
        idxdict = {}
        for i, val in enumerate(vals):
          idxdict[val[0]] = i
          lines.append(ax1.plot([i for i in xaxis if (i & 1) == 0], val[1], label=paperNames[val[0]], marker=markers[i], linestyle=' '))
        ax1.set_xlabel("Size ($n$)")  
        ax1.set_yscale('log')
        ax1.set_ylabel(ylbl)
        ax1.legend(loc="upper left")
        if (vals, fname, ylbl) == timeinfo: pass      
        else: ax1.axhline(y=1e-8, color='gray', linestyle='-'); ax1.axhline(y=1e-10, color='gray', linestyle='-')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_title(("Loop " if loop else "") + "Hafnian of $n\\times n$ Matrix" + ("" if batchsize==1 else (" Batch=$n$"))) #str(batchsize)
        fig.savefig(os.path.join(saveFolder, fname + ".svg"), format="svg")
        fig.savefig(os.path.join(saveFolder, fname + ".pgf"), format="pgf")
        #import tikzplotlib #pip install tikzplotlib
        #python3 -c "import tikzplotlib; print(tikzplotlib.Flavors.latex.preamble())"
        #for line in lines:
        #    for z in line: z.set_label(z.get_label().replace("_", "\\_")) #fix bug with underscore in tikzplotlib legend label escaping
        #ax1.legend()
        #tikzplotlib.save(os.path.join(saveFolder, fname + ".tex"))
        plt.close(fig)
#verify_timing(DEPTH, 1, True, False)
verify_timing(DEPTH, 1, False, False)
#verify_timing(DEPTH, 1, True, True)
#verify_timing(DEPTH, 1, False, True)
