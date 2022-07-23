
import numpy as np

from piquassoboost.sampling.Boson_Sampling_Utilities import PowerTraceHafnian, PowerTraceHafnianRecursive, PowerTraceLoopHafnian, PowerTraceLoopHafnianRecursive
from thewalrus import hafnian, hafnian_repeated
from scipy.stats import unitary_group

DEPTH=26
saveFolder = "resultslh"
def make_symmetric(A): return A + A.T
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
    gen_test_data = {rf.__name__:{dim:make_symmetric(np.random.random((dim, dim))+np.random.random((dim, dim))*1j if dim <= 1 else rf(dim)) for dim in range(nmax+1)} for rf in randfuncs}
    with open(os.path.join(saveFolder, "matrices.bin"), "wb") as f:
      pickle.dump(gen_test_data, f)
  return gen_test_data

calculators = [None, None]

def lhafnian_walrus(Arep):
    return hafnian_repeated(Arep, np.array([1]*(len(Arep)), dtype=np.int64), loop=True)
    #return hafnian(Arep, loop=True)
def lhafnian_powertrace(Arep):
    if calculators[0] is None: calculators[0] = PowerTraceLoopHafnian(Arep)
    else: calculators[0].matrix = Arep
    return calculators[0].calculate()
def lhafnian_powertrace_recursive(Arep):
    if (len(Arep) & 1) != 0: return 1+0j if Arep.shape == (0, 0) else 0
    #if calculators[1] is None:
    calculators[1] = PowerTraceLoopHafnianRecursive(Arep, np.array([1]*(len(Arep)//2), dtype=np.int64))
    #else: calculators[1].matrix = Arep; calculators[1].occupancy = np.array([1]*(len(Arep)//2), dtype=np.int64)
    return calculators[1].calculate()
    
largeLoopHafnianFuncs = (lhafnian_powertrace, lhafnian_powertrace_recursive, lhafnian_walrus)
testLoopHafnianFuncs = ()

paperNames = {lhafnian_powertrace: "PiquassoBoost", lhafnian_powertrace_recursive: "PiquassoBoost Recursive",
              lhafnian_walrus: "thewalrus"}

def verify_timing(nmax, batchsize=1):
  ERRBOUND = 1e-6
  largeFuncs = largeLoopHafnianFuncs
  suffix = "" if batchsize == 1 else str(batchsize)
  verdata = "verifydata.bin"
  resdata = "resultdata" + suffix + ".bin"
  xaxis = list(range(0, nmax+1, 2))
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
    for func in testLoopHafnianFuncs:
      if func.__name__ in res[key]: del res[key][func.__name__]
    for func in largeFuncs:
      if not func.__name__ in res[key]: res[key][func.__name__] = []
      if not func.__name__ in results[key]: results[key][func.__name__] = []
      print("Verifying and Testing", func.__name__)
      for dim in xaxis:
        if len(res[key][func.__name__]) <= dim or len(results[key][func.__name__]) <= dim or True:          
          mplier = 5 if dim < 24 else 1
          v = [None]
          def save_result():
              if batchsize == 1: v[0] = func(A[dim])
              else: v[0] = func([A[dim]] * dim) #batchsize)
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
      with open(os.path.join(saveFolder, "verifydata.csv"), "w") as f:
          import csv
          writer = csv.writer(f, delimiter='\t')
          writer.writerow(["Absolute Error compared to " + largeFuncs[0].__name__]) 
          writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
          writer.writerows([[i] + [abs(res[key][largeFuncs[0].__name__][i] - res[key][x.__name__][i]) for x in largeFuncs] for i in xaxis])
          writer.writerow(["Relative Error compared to " + largeFuncs[0].__name__]) 
          writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
          writer.writerows([[i] + [abs(res[key][largeFuncs[0].__name__][i] - res[key][x.__name__][i]) / abs(res[key][largeFuncs[0].__name__][i]) for x in largeFuncs] for i in xaxis])
          writer.writerow(["Loop Hafnian Computation Raw Results"])
          writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
          writer.writerows([[i] + [res[key][x.__name__][i] for x in largeFuncs] for i in xaxis])
          for dim in range(nmax+1):
            writer.writerow(["Random Unitary Test Matrix " + str(dim) + "x" + str(dim)])
            if dim != 0: writer.writerow([""] + [str(j) for j in range(dim)])
            for i in range(dim):
              writer.writerow([i] + [A[dim][i][j] for j in range(dim)])
    with open(os.path.join(saveFolder, "resultdata" + suffix + ".csv"), "w") as f:
      import csv
      writer = csv.writer(f, delimiter='\t')
      writer.writerow(["Size (n)"] + [f.__name__ for f in largeFuncs])
      writer.writerows([[i] + [results[key][x.__name__][i] for x in largeFuncs] for i in xaxis])

    for i in xaxis:
      #assert all(abs(res[key][largeFuncs[0].__name__][i] - res[key][x][i]) < ERRBOUND for x in res[key] if x != largeFuncs[0].__name__)
      failures = [(i, x, res[key][x][i], res[key][largeFuncs[0].__name__][i], abs((res[key][largeFuncs[0].__name__][i] - res[key][x][i]) / abs(res[key][largeFuncs[0].__name__][i]))) for x in (y.__name__ for y in largeFuncs) if x != largeFuncs[0].__name__ and abs((res[key][largeFuncs[0].__name__][i] - res[key][x][i]) / abs(res[key][largeFuncs[0].__name__][i])) > ERRBOUND]
      if len(failures) != 0: print("ACCURACY FAILURES: ", failures); #assert False, failures

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import math
    plt.rcParams['text.usetex'] = True
    from matplotlib.ticker import MaxNLocator
    verinfo = ([(f, [abs(res[key][largeFuncs[0].__name__][i] - res[key][f.__name__][i]) / abs(res[key][largeFuncs[0].__name__][i]) for i in xaxis]) for f in largeFuncs[1:]], "loophafacc", "Accuracy relative to " + paperNames[largeFuncs[0]] + " ($\\log_{10}$)")
    timeinfo = ([(f, [results[key][f.__name__][i] for i in xaxis]) for f in largeFuncs], "loophaftime" + suffix, "Time ($\\log_{10}$ s)")
    for vals, fname, ylbl in ((verinfo, timeinfo) if batchsize == 1 else (timeinfo,)):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        markers = ['o', '*', 'x', '+', 's', 'p', '1', '2', '3', '4', '8', 'P', 'h']
        lines = []
        idxdict = {}
        for i, val in enumerate(vals):
          idxdict[val[0]] = i
          lines.append(ax1.plot(xaxis, val[1], label=paperNames[val[0]], marker=markers[i], linestyle=' '))
        ax1.set_xlabel("Size ($n$)")  
        ax1.set_yscale('log', base=10)
        ax1.set_ylabel(ylbl)
        ax1.legend(loc="upper left")
        if (vals, fname, ylbl) == timeinfo: pass      
        else: ax1.axhline(y=1e-8, color='gray', linestyle='-'); ax1.axhline(y=1e-10, color='gray', linestyle='-')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_title("Loop Hafnian of $n\\times n$ Matrix" + ("" if batchsize==1 else (" Batch=$n$"))) #str(batchsize)
        fig.savefig(os.path.join(saveFolder, fname + ".svg"), format="svg")
        import tikzplotlib #pip install tikzplotlib
        #python3 -c "import tikzplotlib; print(tikzplotlib.Flavors.latex.preamble())"
        for line in lines:
            for z in line: z.set_label(z.get_label().replace("_", "\\_")) #fix bug with underscore in tikzplotlib legend label escaping
        ax1.legend()
        tikzplotlib.save(os.path.join(saveFolder, fname + ".tex"))
        plt.close(fig)
verify_timing(DEPTH, 1)
