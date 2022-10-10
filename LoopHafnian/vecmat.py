
#export PATH=$PATH:/nix/store/myclxgzxiqrlhgw5b6h4mjmvyks2c5lz-Groq.View.server.groqview-streams/bin/
#/nix/store/myclxgzxiqrlhgw5b6h4mjmvyks2c5lz-Groq.View.server.groqview-streams/bin/groqview-streams

#/nix/store/8vd44g013blr4a4qymjwhgx7mmxgpaf2-GroqAPI.lib/

import sys
MANT_DIG = sys.float_info.mant_dig
import numpy as np
from scipy.stats import unitary_group
import groq.api as g
import groq.api.instruction as inst
import groq.api.nn as nn
from groq.common import print_utils

#https://www.researchgate.net/publication/235134418_A_Simplified_Fraction-Free_Integer_Gauss_Elimination_Algorithm
def gaussianElimInteger(x): #division-free Gaussian elimination
  h = 0 #Initialization of pivot row
  k = 0 #Initialization of pivot column
  oddswaps, mulFactor = False, 1
  m = x.shape[0]
  n = x.shape[1]
  while (h < m and k < n):
    #Find the k-th pivot
    res = [(abs(x[i, k]), i) for i in range(h, m) if x[i, k] != 0]
    i_max = -1 if len(res) == 0 else min(res)[1] #index of maximum which is first one with a bit set, also should consider absolute value but here its not applicable since no negative values though zero still possible
    if (i_max < h or x[i_max, k] == 0): #No pivot in this column, pass to next column
      k += 1
    else:
      #swap rows h and i_max
      if (h != i_max): x[[h, i_max]] = x[[i_max, h]]; oddswaps = not oddswaps
      #Do for all rows below pivot
      for i in range(h+1, m):
        #Do for all remaining elements in current row
        mulFactor *= x[h, k]
        for j in range(k+1, n): x[i, j] = x[h, k] * x[i, j] - x[i, k] * x[h, j]
        x[i, k] = 0
      h += 1
      k += 1 #Increase pivot row and column
  return x, oddswaps, mulFactor #ret
def backSubstitution(x):
  h = 0 #Initialization of pivot row
  k = 0 #Initialization of pivot column
  m = x.shape[0]
  n = x.shape[1]
  while (h < m and k < n):
    #Find the k-th pivot
    res = [(abs(x[i, k]), i) for i in range(h, m) if x[i, k] != 0]
    i_max = -1 if len(res) == 0 else min(res)[1] #index of maximum which is first one with a bit set, also should consider absolute value but here its not applicable since no negative values though zero still possible
    if (i_max < h or x[i_max, k] == 0): #No pivot in this column, pass to next column
      k += 1
    else:
      #swap rows h and i_max
      if (h != i_max): x[[h, i_max]] = x[[i_max, h]]
      #Do for all rows below pivot
      #reduced row echelon form (RREF) is obtained without a back substitution step by starting from 0 and skipping h      
      for j in range(n-1, k-1, -1): x[h, j] //= x[h, k]
      for i in range(m):
        if (h == i): continue
        f = x[i, k] // x[h, k]
        x[i, k] = 0
        #Do for all remaining elements in current row
        for j in range(k+1, n):
          x[i, j] -= (x[h, j] * f)
      h += 1
      k += 1 #Increase pivot row and column
  return x #ret
def gaussianElimIntegerPolynomial(x): #division-free Gaussian elimination
  h = 0 #Initialization of pivot row
  k = 0 #Initialization of pivot column
  oddswaps, mulFactor = False, [1]
  m = len(x)
  n = len(x[0])
  while (h < m and k < n):
    #Find the k-th pivot
    res = [(abs(x[i][k][0]), i) for i in range(h, m) if x[i][k] != [] and x[i][k] != [0]]
    i_max = -1 if len(res) == 0 else min(res)[1] #index of maximum which is first one with a bit set, also should consider absolute value but here its not applicable since no negative values though zero still possible
    if (i_max < h or x[i_max][k] == 0): #No pivot in this column, pass to next column
      k += 1
    else:
      #swap rows h and i_max
      if (h != i_max): x[h], x[i_max] = x[i_max], x[h]; oddswaps = not oddswaps
      #Do for all rows below pivot
      for i in range(h+1, m):
        mulFactor = mulPolyR(mulFactor, x[h][k], None)
        for j in range(k+1, n): x[i][j] = addPoly(mulPolyR(x[h][k], x[i][j], None), [-x for x in mulPolyR(x[i][k], x[h][j], None)])
        x[i][k] = []
      h += 1
      k += 1 #Increase pivot row and column
  return x, oddswaps, mulFactor #ret
def dosign(parity, x): return -x if parity else x
def prod(x, y): return x * y
def multiprod(l):
  import functools
  return functools.reduce(prod, l)
def powerset(s): #empty set handled as special case
  import itertools
  for i in range(1, len(s)+1):
    yield from itertools.combinations(s, i)
def matix(mat, ix): return [[mat[i][j] for j in ix] for i in ix]
 
#http://oeis.org/A072831 Number of bits in n!.
def get_fact_bitsizes(n):
  import math
  return math.factorial(n).bit_length()
#print(list(get_fact_bitsizes(n) for n in range(64)))
#https://en.wikipedia.org/wiki/Hafnian
#https://the-walrus.readthedocs.io/en/latest/hafnian.html
def hafnian(mat): #symmetric matrix
  if (len(mat) & 1) != 0: return 0
  n = len(mat) // 2
  if n == 0: return 1
  import itertools, math
  return sum(multiprod((mat[sigma[2*i]][sigma[2*i+1]] for i in range(n))) for sigma in itertools.permutations(range(2*n))) // (math.factorial(n) * (1 << n))
def complete_graph_perf_match(iterable): #O(n^2) and recursive...
  pool = tuple(iterable)
  n = len(pool) // 2
  if n == 0: return
  elif n == 1: yield [pool]; return
  for i in range(1, n*2):
    yield from [[(pool[0], pool[i]), *x] for x in complete_graph_perf_match(pool[1:i] + pool[i+1:])]
#https://oeis.org/A000085
def num_single_pair_match(n):
  import math
  fact = math.factorial(n)
  return sum(fact // ((math.factorial(n-2*k)*math.factorial(k)) << k) for k in range(n // 2 + 1))
def complete_graph_single_pair_match(iterable):
  pool = tuple(iterable)
  n = len(pool)
  if n == 0: return
  if n == 1: yield [(pool[0], pool[0])]
  elif n == 2: yield from [[pool], [(pool[0], pool[0]), (pool[1], pool[1])]]; return
  for i in range(1, n):
    yield from [[(pool[0], pool[i]), *x] for x in complete_graph_single_pair_match(pool[1:i] + pool[i+1:])]
  yield from [[(pool[0], pool[0]), *x] for x in complete_graph_single_pair_match(pool[1:])]
#print(list(complete_graph_single_pair_match(range(4))))
#print([num_single_pair_match(x) for x in range(10)])
#assert all(len(list(complete_graph_single_pair_match(range(n)))) == num_single_pair_match(n) for n in range(1, 10))
#http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.3057&rep=rep1&type=pdf
def linear_extension_perf_match(iterable):
  #partial ordered set (poset) relations since groups are all size 2, are the even elements are ordered, and each pair of elements are ordered
  pool = tuple(iterable); n = len(pool)
  le, li = list(range(n)), list(range(n))
  a, b = [le[i] for i in range(1, n-1, 2)], [le[i] for i in range(2, n-1, 2)]  
  MaxPair = n // 2 - 1
  IsPlus = [True]
  #MaxPair, n, le, li, a, b = 2, 4, [0, 1, 2, 3], [0, 1, 2, 3], [0, 2], [1, 3] #a1,b1,a2,b2 = 0, 1, 2, 3
  def toSet(): return [(iterable[li[i*2]], iterable[li[i*2+1]]) for i in range(n // 2)]
  yield toSet()
  def transpose(x, y):
    le[li[x]], le[li[y]], li[x], li[y] = le[li[y]], le[li[x]], li[y], li[x]
  def Switch(i):
    if i == -1: IsPlus[0] = not IsPlus[0]
    else:
      transpose(a[i], b[i]); a[i], b[i] = b[i], a[i]
  def Move(x, isRight):
    transpose(x, le[li[x]+(1 if isRight else -1)])
  def isComparable(x, y):
    #return x==0 and y==2 or x==1 and y==3 #or y==0 and x==2 or y==1 and x==3
    return (x & 1) == 0 and (y & 1) == 0 and x < y or (x & 1) == 0 and y == x+1
  def Right(x, i): return li[x] != n-1 and not isComparable(le[li[x]], le[li[x]+1]) and (i is None or le[li[x]+1] != b[i])
  def GenLE(i):
    if i >= 0:
      yield from GenLE(i-1)
      mrb = 0; typical = False
      while i < MaxPair and Right(b[i], None):
        mrb += 1; Move(b[i], True)
        if IsPlus[0]: yield toSet()
        yield from GenLE(i-1)
        mra = 0
        if Right(a[i], i):
          typical = True
          while True:
            mra += 1; Move(a[i], True)
            if IsPlus[0]: yield toSet()
            yield from GenLE(i-1)
            if not Right(a[i], i): break
        if typical:
          Switch(i-1)
          if IsPlus[0]: yield toSet()
          yield from GenLE(i-1)
          mla = mra + (-1 if (mrb & 1) != 0 else 1)
          for _ in range(0, mla):
            Move(a[i], False)
            if IsPlus[0]: yield toSet()
            yield from GenLE(i-1)
      if typical and (mrb & 1) != 0: Move(a[i], False)
      else: Switch(i-1)
      if IsPlus[0]: yield toSet()
      yield from GenLE(i-1)
      for _ in range(mrb):
        Move(b[i], False)
        if IsPlus[0]: yield toSet()
        yield from GenLE(i-1)
  yield from GenLE(MaxPair)
#print(list(complete_graph_perf_match(range(2))))
#print(list(complete_graph_perf_match(range(4))))
#assert sorted(list(complete_graph_perf_match(range(6)))) == sorted(list(linear_extension_perf_match(range(6))))
#assert sorted(list(complete_graph_perf_match(range(8)))) == sorted(list(linear_extension_perf_match(range(8))))
def hafnian_perf_match(mat, isLoop=False, linExt=True): #symmetric matrix
  if (len(mat) & 1) != 0: return 0
  n = len(mat) // 2
  if n == 0: return 1
  if isLoop: matchfunc = complete_graph_single_pair_match
  elif linExt: matchfunc = linear_extension_perf_match
  else: matchfunc = complete_graph_perf_match
  return sum(multiprod((mat[u][v] for u, v in matches)) for matches in matchfunc(range(2*n)))
#for Hafnian, adding [[0 1] [1 0]] direct sum along the added pairs of diagonals preserves the computation
#print(list(hafnian_perf_match([[1 if i>=4 and j>=4 and (i==j+1 or j==i+1) or i < 4 and j < 4 else 0 for j in range(n)] for i in range(n)]) for n in range(4, 11, 2)))
#for loop Hafnian, adding ones on the diagonal extends the matrix while preserving the computation
#print(list(hafnian_perf_match([[1 if i==j or i < 4 and j < 4 else 0 for j in range(n)] for i in range(n)], isLoop=True) for n in range(4, 11, 2)))
def addPoly(a, b):
  alen, blen = len(a), len(b)
  c = [0] * max(alen, blen)
  clen = len(c)
  for i in range(0, clen):
    if (i >= alen): c[i] = b[i]
    elif (i >= blen): c[i] = a[i]
    else: c[i] = a[i] + b[i]
  import itertools
  return c if clen == 0 or c[-1] != 0 else list(reversed(list(itertools.dropwhile(lambda cr: cr == 0, reversed(c)))))
def mulPolyR(a, b, clen):
  alen, blen = len(a), len(b)
  if (alen == 0): return a
  if (blen == 0): return b
  if clen is None: clen = alen + blen -1
  p = [0] * min(clen, (alen + blen - 1))
  for i in range(0, blen):
    if (b[i] == 0): continue
    for j in range(0, alen):
      if (a[j] == 0) or i + j >= clen: continue
      p[i + j] += a[j] * b[i]
  import itertools
  return list(reversed(list(itertools.dropwhile(lambda c: c == 0, reversed(p)))))
#https://en.wikipedia.org/wiki/Polynomial_long_division#Pseudo-code
def divmodPoly(a, b):
  #if (len(b) == 0) raise ValueError
  alen, blen = len(a), len(b)
  q, r = [0] * alen, a
  bneg = mulPolyR(b, [-1], None)
  rlen = len(r)
  d = (rlen - 1) - (blen - 1)
  while (rlen != 0 and d >= 0):
    aoffs = d
    assert r[-1] == (r[-1] // b[-1]) * b[-1]
    q[aoffs] = r[-1] // b[-1]
    if (q[aoffs] == 0): break
    r = addPoly(r, q[:aoffs] + mulPolyR(bneg, [q[aoffs]], None))
    rlen = len(r)
    d = (rlen - 1) - (blen - 1)
  assert r == []
  import itertools
  return list(reversed(list(itertools.dropwhile(lambda c: c == 0, reversed(q)))))
#https://arxiv.org/abs/1107.4466
def hafnian_ryser_time(mat):
  if (len(mat) & 1) != 0: return 0
  n = len(mat) // 2
  if n == 0: return 1
  B = [[[c] for c in r] for r in mat]
  h = 0
  for X in powerset(list(range(1, n+1))):
    X = set(X)
    g = [1]
    Bl = B
    for i in range(1, n+1):
      if i in X:
        g = mulPolyR(g, [1] + Bl[0][1], n+1)
        Bl = [[addPoly([0] + addPoly(mulPolyR(Bl[0][j+2], Bl[1][k+2], n), mulPolyR(Bl[0][k+2], Bl[1][j+2], n))[:n-1], Bl[j+2][k+2])
              if j != k else 0 for k in range(2*(n-i))] for j in range(2*(n-i))]
      else:
        Bl = [r[2:] for r in Bl[2:]]
    if len(g) > n: h += dosign(((n - len(X)) & 1) != 0, g[n])
  return h
def characteristicPolynomial(mat):
  import functools
  ref, oddswaps, mulFactor = gaussianElimIntegerPolynomial([[[-c] if j != i else [-c, 1] for j, c in enumerate(r)] for i, r in enumerate(mat)])
  #ref, oddswaps, mulFactor = gaussianElimIntegerPolynomial([[[c] if j != i else [c, -1] for j, c in enumerate(r)] for i, r in enumerate(mat)])
  print(list(ref[i][i] for i in range(len(ref))))
  poly = [dosign(oddswaps, x) for x in functools.reduce(lambda x, y: mulPolyR(x, y, None), (ref[i][i] for i in range(len(ref))))]  
  poly = divmodPoly(poly, mulFactor)
  #import numpy as np
  #assert poly == [round(x) for x in reversed(np.poly(np.array(mat)))], (poly, [round(x) for x in reversed(np.poly(np.array(mat)))])
  return poly #if poly[-1] != -1 else [-x for x in poly]
#assert characteristicPolynomial([[-1, 4, 0, 0, 0], [0, 3, 0, 0, 0], [0, -4, -1, 0, 0], [3, -8, -4, 2, 1], [1, 5, 4, 1, 4]]) == [-21, -17, 20, 8, -7, 1]
def matMul(mat1, mat2):
  m1, n1, m2, n2 = len(mat1), len(mat1[0]), len(mat2), len(mat2[0])
  assert n1 == m2
  out = [[0 for _ in range(n2)] for _ in range(m1)]
  for i in range(m1):
    for j in range(n2):
      for k in range(n1):
        out[i][j] += mat1[i][k] * mat2[k][j]
  return out
def directSum(mat1, mat2):
  m1, n1, m2, n2 = len(mat1), len(mat1[0]), len(mat2), len(mat2[0])
  return ([[r[i] if i < len(r) else 0 for i in range(n1+n2)] for r in mat1] +
          [[r[i-m1] if i >= m1 else 0 for i in range(n1+n2)] for r in mat2])
#http://bulletin.pan.pl/(56-4)391.pdf
def minimalPolynomial(mat):
  #lbda = characteristicPolynomial(mat)
  z, n = mat, len(mat)
  M = [[0 for _ in range(n+1)] for _ in range(n*n)]
  for i in range(n):
    for j in range(n):
      if i == j: M[i*n+j][0] = 1
  for k in range(n):
    for i in range(n):
      for j in range(n):
        M[i*n+j][k+1] = -z[i][j] if k == n-1 else z[i][j]
    if k != n-1: z = matMul(z, mat)
  import numpy as np
  ref, _, _ = gaussianElimInteger(np.array(M))
  rank = sum(1 for i in range(n+1) if ref[i, i] != 0)
  #assert rank == np.linalg.matrix_rank(np.array(M)), (rank, np.linalg.matrix_rank(np.array(M)))
  return [-x for x in backSubstitution(ref[:rank, :rank+1])[:, rank]] + [1]
#assert minimalPolynomial([[3, -3, 2], [-1, 5, -2], [-1, 3, 0]]) == [8, -6, 1]
#http://oeis.org/A003418 Least common multiple (or LCM) of {1, 2, ..., n} for n >= 1, a(0) = 1.
def factoriallcms(n):
  import math, functools
  return functools.reduce(lambda x, y: x*y//math.gcd(x, y), range(1, n+1))
def householder(vec):
    n = len(vec)
    s = np.dot(vec[1:n].conj(), vec[1:n])
    v = np.hstack(([1], vec[1:n]))
    if s == 0: B = 0
    else:
        m = np.sqrt(vec[0].conj()*vec[0]+s)
        if np.iscomplex(vec[0]):
            sign = 1 if vec[0]==0 else vec[0] / abs(vec[0])
            v[0] = vec[0] - sign * m
        elif vec[0] <= 0: v[0] = vec[0] - m
        else: v[0] = -s / (vec[0] + m)
        B = 2*v[0].conj()*v[0]/(s + v[0].conj()*v[0])
        v = v / v[0]
    #print(vec, v, B)
    return v, B
#https://netlib.org/lapack/lawnspdf/lawn148.pdf
def givens(a, b): #counter-clockwise rotation
    if b == 0: c, s, r = 1, 0, a
    else:
        if np.iscomplex(a) or np.iscomplex(b):
            sign = 1 if a==0 else a / abs(a)
            t = np.linalg.norm([a,b])
            d = 1/t
            c = abs(a) * d
            s = -sign.conj() * b * d
            r = sign * t
        elif abs(b) > abs(a): #cb+sa=0, -s*b+ca=r then s=-b/r, c=a/r so (b/r)*b+(a/r)a=r r=b/|b|*sqrt(a^2+b^2)
            sign = 1 if b==0 else b / abs(b)
            tau = abs(a)/abs(b)
            s = 1/np.sqrt(1+tau*tau) #s=b/sqrt(a^2+b^2)
            c = s * a / abs(b) * -sign.conj()
            r = -b / s
        else:
            sign = 1 if a==0 else a / abs(a)
            tau = abs(b)/abs(a)
            c = 1/np.sqrt(1+tau*tau)
            s = c * b / abs(a) * -sign.conj()
            r = a / c
    assert np.allclose(np.array([[c, s], [-s.conj(), c]]).T @ np.array([a, b]), np.array([r, 0])), (np.array([[c, s], [-s.conj(), c]]).T @ np.array([a, b]), np.array([r, 0]))
    return c, s
def fastgivens(x, dorig):
    d = dorig.copy()
    if x[1] != 0:
        a = -x[0].conj()/x[1].conj()
        b = (-a*d[1]/d[0]).conj()
        g = -a*b #a=-x[0]x[1]*/abs(x[1])
        if abs(g) <= 1: #g*x[1]==b*x[0] x[0]+ax[1]=0 g=-ab=x[0]x[1]*/abs(x[1])b  or x[1]*(1+g)=b*x[0]+x[1]        
            typ = 1
            tau = d[0]
            d[0] = (1 + g)*d[1]
            d[1] = (1 + g)*tau
        else:
            typ = 2
            a, b, g = 1 / a, 1 / b, 1 / g
            d[0] = (1 + g)*d[0]
            d[1] = (1 + g)*d[1]
    else:
        typ, a, b, g = 2, 0, 0, 0
    M = np.array([[b, 1], [1, a]]) if typ == 1 else np.array([[1, a], [b, 1]])
    assert np.allclose(M.conj().T @ x, np.array([x[2-typ]*(1+g), 0])), (M.conj().T @ x, np.array([x[2-typ]*(1+g), 0]))
    assert np.allclose(M.conj().T @ np.diag(dorig) @ M, np.diag(d)), (M.conj().T @ np.diag(dorig) @ M, np.diag(d))
    return a, b, typ, d
def qr_linalg(mat):
    Q, R = np.linalg.qr(mat, mode='complete')
    #print(np.linalg.qr(mat, mode='raw'))
    #print(Q, R, Q @ R, mat)
    assert np.allclose(np.eye(len(mat)), Q @ Q.conj().T)
    assert np.allclose(Q @ R, mat)
    return Q, R
def qr_householder(mat):
    R = mat.copy()
    m = n = len(mat)
    Q = np.eye(n, dtype=mat.dtype)
    Bs = []
    for j in range(n):
        v, B = householder(R[j:m, j])
        Bs.append(B)
        IBvv = np.eye(m-j, dtype=R.dtype)-B*v[:,np.newaxis]@v[np.newaxis,:].conj()
        R[j:m, j:n] = IBvv @ R[j:m, j:n]
        #Q[:,j:n] = Q[:,j:n] @ IBvv #forward accumulation
        if j < m-1: R[j+1:m, j] = v[1:m-j+1] #save for backward accumulation
    for j in range(n-1, -1, -1): #backward accumulation
        v = np.hstack(([1], R[j+1:m, j]))
        IBvv = np.eye(m-j, dtype=R.dtype)-Bs[j]*v[:,np.newaxis]@v[np.newaxis,:].conj()
        Q[j:m, j:n] = IBvv @ Q[j:m, j:n]
    R = np.triu(R)
    #print(Q, R, Q @ R, mat)
    assert np.allclose(np.eye(len(mat)), Q @ Q.conj().T)
    assert np.allclose(Q @ R, mat)
    return Q, R
def qr_givens(mat):
    R = mat.copy()
    m = n = len(R)
    Q = np.eye(n, dtype=mat.dtype)
    for j in range(n):
        for i in range(m - 1, j, -1):
            c, s = givens(R[i-1,j], R[i,j])
            G = np.array([[c, s], [-s.conj(), c]])
            R[i-1:i+1,j:n] = G.T @ R[i-1:i+1,j:n]
            Q[:,i-1:i+1] = Q[:,i-1:i+1] @ G.conj()
    R = np.triu(R)
    #print(Q, R, Q @ R, mat)
    assert np.allclose(np.eye(len(mat)), Q @ Q.conj().T)
    assert np.allclose(Q @ R, mat)
    return Q, R
def qr_fastgivens(mat):
    R = mat.copy()
    m = n = len(R)
    Q = np.eye(n, dtype=mat.dtype)
    d = np.ones((m,), dtype=mat.dtype)
    for j in range(n):
        for i in range(m - 1, j, -1):
            a, b, typ, d[i-1:i+1] = fastgivens(R[i-1:i+1,j], d[i-1:i+1])
            G = np.array([[b, 1], [1, a]]) if typ == 1 else np.array([[1, a], [b, 1]])
            R[i-1:i+1,j:n] = G.conj().T @ R[i-1:i+1,j:n]
            Q[:,i-1:i+1] = Q[:,i-1:i+1] @ G
    R = np.triu(R)
    assert np.allclose(Q.conj().T @ mat, R), (Q.conj().T @ mat, R)
    assert np.allclose(d, np.diag(Q.conj().T @ Q)), (d, np.diag(Q.conj().T @ Q))
    D = np.diag(1/np.sqrt(d))
    Q = Q @ D
    R = D @ R
    #print(Q, R, Q @ R, mat)
    assert np.allclose(np.eye(len(mat)), Q @ Q.conj().T)
    assert np.allclose(Q @ R, mat)
    return Q, R
def qr_mgs(mat): #classical Gram-Schmidt
    R = np.zeros(mat.shape, dtype=mat.dtype); R[0,0] = np.linalg.norm(mat[:,0])    
    m = n = len(mat)
    Q = np.hstack(((mat[:,0] / R[0,0])[:,np.newaxis], np.zeros((m, n-1))))
    for k in range(1, n):
        R[0:k,k] = Q[0:m,0:k].conj().T @ mat[0:m,k]
        z = mat[0:m,k] - Q[0:m,0:k] @ R[0:k,k]
        R[k,k] = np.linalg.norm(z)
        Q[0:m,k] = z/R[k,k]
    #print(Q, R, Q @ R, mat)
    assert np.allclose(np.eye(len(mat)), Q @ Q.conj().T)
    assert np.allclose(Q @ R, mat)
    return Q, R
def hessenberg_to_qr(mat):
    n = len(mat)
    R = mat.copy()
    Q = np.eye(n, dtype=mat.dtype)
    for j in range(n-1):
        c, s = givens(R[j,j], R[j+1,j])
        G = np.array([[c, s], [-s.conj(), c]])
        R[j:j+2, j:n] = G.T @ R[j:j+2, j:n]
        Q[:,j:j+2] = Q[:,j:j+2] @ G.conj()
    R = np.triu(R)
    #print(Q, R, Q @ R, mat)
    assert np.allclose(np.eye(len(mat)), Q @ Q.conj().T)
    assert np.allclose(Q @ R, mat)
    return Q, R
def qr_hessenberg(mat): return hessenberg_to_qr(hessenberg_scipy(mat)[0])
def hessenberg_scipy(mat):
    from scipy.linalg import hessenberg
    H, Q = hessenberg(mat, calc_q=True)
    assert np.allclose(Q @ H @ Q.conj().T, mat)
    return H, Q
def hessenberg_givens(mat):
    #print(hessenberg_scipy(mat))
    n = len(mat)
    H = mat.copy()
    Q = np.eye(n, dtype=mat.dtype)
    for j in range(n-2):
        for i in range(n-1, j+1, -1):
            c, s = givens(H[i-1,j], H[i,j])
            G = np.array([[c, s], [-s.conj(), c]])
            H[i-1:i+1,j:n] = G.T @ H[i-1:i+1,j:n]
            H[:,i-1:i+1] = H[:,i-1:i+1] @ G.conj()
            Q[:,i-1:i+1] = Q[:,i-1:i+1] @ G.conj()
    H = np.triu(H, -1)
    #print(H, Q, Q @ H @ Q.conj().T, mat)
    assert np.allclose(Q @ H @ Q.conj().T, mat)
    return H, Q
def hessenberg_fastgivens(mat):
    n = len(mat)
    H = mat.copy()
    Q = np.eye(n, dtype=mat.dtype)
    d = np.ones((n,), dtype=mat.dtype)
    for j in range(n-2):
        for i in range(n-1, j+1, -1):
            a, b, typ, d[i-1:i+1] = fastgivens(H[i-1:i+1,j], d[i-1:i+1])
            G = np.array([[b, 1], [1, a]]) if typ == 1 else np.array([[1, a], [b, 1]])
            H[i-1:i+1,j:n] = G.conj().T @ H[i-1:i+1,j:n]
            H[:,i-1:i+1] = H[:,i-1:i+1] @ G
            Q[:,i-1:i+1] = Q[:,i-1:i+1] @ G
    H = np.triu(H, -1)
    #assert np.allclose(Q.conj().T @ mat, H @ Q.T), (Q.conj().T @ mat, H @ Q.T)
    assert np.allclose(d, np.diag(Q.conj().T @ Q)), (d, np.diag(Q.conj().T @ Q))    
    D = np.diag(1/np.sqrt(d))
    Q = Q @ D
    H = D @ H @ D
    #print(H, Q, Q @ H @ Q.conj().T, mat)
    assert np.allclose(Q @ H @ Q.conj().T, mat)
    return H, Q    
def hessenberg_householder(mat):
    n = len(mat)
    H = mat.copy()
    m = n
    Q = np.eye(n, dtype=mat.dtype)
    Bs = []
    for j in range(0, n-2):
        v, B = householder(H[j+1:m, j])
        Bs.append(B)
        IBvv = np.eye(m-j-1, dtype=H.dtype)-B*v[:,np.newaxis]@v[np.newaxis,:].conj()
        H[j+1:m, j:n] = IBvv @ H[j+1:m, j:n]
        H[:,j+1:n] = H[:,j+1:n] @ IBvv 
        #H[j+1:m, j:n] = H[j+1:m, j:n] - B*v[:,np.newaxis]@(v[np.newaxis,:] @ H[j+1:m, j:n])
        #H[:,j+1:n] = H[:,j+1:n] - B*(H[:,j+1:n]@v[:,np.newaxis])@v[np.newaxis,:]        
        if j < m-1: H[j+2:m, j] = v[1:m-j+1]
    for j in range(n-3, -1, -1): #backward accumulation
        v = np.hstack(([1], H[j+2:m, j]))
        IBvv = np.eye(m-j-1, dtype=H.dtype)-Bs[j]*v[:,np.newaxis]@v[np.newaxis,:].conj()
        Q[j+1:m, j:n] = IBvv @ Q[j+1:m, j:n]       
    H = np.triu(H, -1)
    #print(H, Q, Q @ H @ Q.conj().T, mat)
    assert np.allclose(Q @ H @ Q.conj().T, mat)
    return H, Q
def hessenberg_arnoldi(mat): #Arnoldi iteration
    m = n = len(mat)
    H = np.zeros((m+1,n), dtype=mat.dtype)    
    Q = np.hstack(((mat.diagonal() / np.linalg.norm(mat.diagonal()))[:,np.newaxis], np.zeros((m, n)))) #arbitrary
    for k in range(n):
        z = mat @ Q[:,k]
        for j in range(k+1):
            H[j,k] = Q[:,j].conj().T @ z
            z = z - H[j,k] * Q[:,j]
        #R[0:k,k] = Q[0:m,0:k].conj().T @ mat[0:m,k]
        #z = mat[0:m,k] - Q[0:m,0:k] @ R[0:k,k]        
        #H[:k+1,k] = Q[:,0:k+1].conj().T @ z
        #z = v - Q[:,0:k+1] @ H[:k+1,k]
        H[k+1,k] = np.linalg.norm(z)
        Q[:,k+1] = z/H[k+1,k]
    #print(H, Q, mat @ Q[:,:n], Q @ H)
    #assert np.allclose(Q @ H @ Q.conj().T, mat)
    assert np.allclose(mat @ Q[:,:n], Q @ H)
    return H, Q
def hessenberg_prime_charpoly(intmat, p):
    n = len(intmat)
    P = np.zeros((n+1, n+1), dtype=intmat.dtype)
    P[0,n] = 1
    """
    #lower hessenberg
    for i in range(n-1):
        P[i+1] = (sum((p - intmat[i,j])*pow(intmat[i,i+1], p-2, p)*P[j] % p for j in range(n)) % p + pow(intmat[i,i+1], p-2, p) * np.concatenate((P[i,1:], [0])) % p) % p
    P[n] = (sum(intmat[n-1,j]*P[j] % p for j in range(n)) % p - np.concatenate((P[n-1,1:], [0])) % p) % p
    """
    for i in range(n-1):
        P[i+1] = (sum((p - intmat[j,i])*pow(intmat[i+1,i], p-2, p)*P[j] % p for j in range(n)) % p + pow(intmat[i+1,i], p-2, p) * np.concatenate((P[i,1:], [0])) % p) % p
    P[n] = (sum(intmat[j,n-1]*P[j] % p for j in range(n)) % p - np.concatenate((P[n-1,1:], [0])) % p) % p
    P[n] = P[n] * pow(P[n,0], p-2, p) % p
    P[n] = np.where(P[n] < (p-1)//2, P[n], P[n] - p)
    return P[n]
def hessenberg_prime(mat): #https://www.sciencedirect.com/science/article/pii/0898122178900081/pdf?md5=5cc1225a5bc5f72092ba50220caec53a&pid=1-s2.0-0898122178900081-main.pdf
    print(hessenberg_scipy(mat))
    mant, exp = np.frexp(mat.real)
    #53/largest mantissa is not optimal - should do a count of trailing zero bits...  but __builtin_ctz not available in numpy?
    intmat = np.array(np.rint(np.ldexp(mant, exp+53-np.amin(exp))).astype(np.int64).tolist(), dtype=np.object)
    assert np.allclose(mat, intmat.astype(np.float64) / (2**(53-np.amin(exp).astype(np.float64))))
    print(np.poly(intmat.astype(np.float64)))
    n = len(intmat)
    mm = intmat @ intmat.conj().T
    M = np.trace(mm) #max(np.linalg.norm(mm), np.trace(mm))
    Pmin = 2 * max(M**n, n*(n+1)*M**(n-1))
    print(Pmin.bit_length())
    Q = np.eye(n, dtype=intmat.dtype)
    #origmat = intmat / (2**(53-np.amin(exp).astype(np.float64)))
    p = 2**400-593 #2**54 - 33 #https://primes.utm.edu/lists/2small/0bit.html 
    #p = 101
    """
    i = 0 #lower Hessenberg
    while i <= n-3:
        if intmat[i,i+1] == 0:
            j = next(iter(j for j in range(i+2, n) if a[i,j] != 0), None)
            if j is None: break
            intmat[:,i+1], intmat[:,j] = intmat[:,j], intmat[:,i+1]
            intmat[i+1,:], intmat[j,:] = intmat[j,:], intmat[i+1,:]
        for k in range(i+2, n):
            a = (intmat[i,k] * pow(intmat[i,i+1], p-2, p)) % p
            intmat[:,k] = (intmat[:,k] - a * intmat[:,i+1]) % p
            intmat[i+1,:] = (a * intmat[k,:] + intmat[i+1,:]) % p
        i += 1
    """
    i = n-1
    while i >= 2:
        if intmat[i,i-1] == 0:
            j = next(iter(j for j in range(i-2, -1, -1) if a[i,j] != 0), None)
            if j is None: break
            intmat[:,i-1], intmat[:,j] = intmat[:,j], intmat[:,i-1]
            intmat[i-1,:], intmat[j,:] = intmat[j,:], intmat[i-1,:]
        for k in range(i-2, -1, -1):
            a = (intmat[i,k] * pow(intmat[i,i-1], p-2, p)) % p
            intmat[:,k] = (intmat[:,k] - a * intmat[:,i-1]) % p
            intmat[i-1,:] = (a * intmat[k,:] + intmat[i-1,:]) % p
        i -= 1
    print(intmat)
    print(np.poly(intmat.astype(np.float64)))
    charpoly = hessenberg_prime_charpoly(intmat, p) / np.array([(2**(x*(53-np.amin(exp).astype(np.float64)))) for x in range(n+1)])
    print(charpoly)
    intmat = np.where(intmat < (p-1)//2, intmat, intmat - p)
    H = intmat / np.array([(2**(6.5*(53-np.amin(exp).astype(np.float64)))) for x in range(1,n+1)])[:,np.newaxis] #(2**(53-np.amin(exp).astype(np.float64)))
    print(H)
    return H, Q
def hessenberg_gaussian(mat):
    n = len(mat)
    H = mat.copy()
    Q = np.eye(n, dtype=mat.dtype)
    i = 0
    while i <= n-3:
        if H[i+1,i] == 0:
            j = next(iter(j for j in range(i+2, n) if a[j,i] != 0), None)
            if j is None: break
            H[:,i+1], H[:,j] = H[:,j], H[:,i+1]
            H[i+1,:], H[j,:] = H[j,:], H[i+1,:]
        for k in range(i+2, n):
            a = H[k,i]/H[i+1,i]
            H[k,:] -= a * H[i+1,:]
            H[:,i+1] += a * H[:,k]
            #Q[:,i+1] -= a * H[:,k]
            Q[k,i+1] = a #need to set Q[k,k], Q[k,i+1], Q[i+1,k], Q[i+1,i+1]
            #assert np.allclose(Q @ H @ Q.conj().T, mat), (Q @ H @ Q.conj().T, mat, i, k)
        i += 1
    print(H, Q, Q @ H @ Q.conj().T, mat)
    #assert np.allclose(Q @ H @ Q.conj().T, mat)
    return H, Q
def hessenberg_gaussian_int(mat):
    mant, exp = np.frexp(mat.real)
    #53/largest mantissa is not optimal - should do a count of trailing zero bits...  but __builtin_ctz not available in numpy?
    intmat = np.array(np.rint(np.ldexp(mant, exp+53-np.amin(exp))).astype(np.int64).tolist(), dtype=np.object)
    assert np.allclose(mat, intmat.astype(np.float64) / (2**(53-np.amin(exp).astype(np.float64))))
    n = len(intmat)
    i = 0
    while i <= n-3:
        if intmat[i+1,i] == 0:
            j = next(iter(j for j in range(i+2, n) if a[j,i] != 0), None)
            if j is None: break
            intmat[:,i+1], intmat[:,j] = intmat[:,j], intmat[:,i+1]
            intmat[i+1,:], intmat[j,:] = intmat[j,:], intmat[i+1,:]
        for k in range(i+2, n):
            gcd = np.gcd(intmat[k,i], intmat[i+1,i])
            a, b = intmat[k,i]//gcd, intmat[i+1,i]//gcd
            print(a, b)
            intmat[k,:] += (b-1) * intmat[k,:] #intmat[k,:] *= b
            intmat[k,:] -= a * intmat[i+1,:]
            intmat[:,i+1] -= (b-1) * intmat[:,i+1]
            intmat[:,i+1] += a * intmat[:,k]
        i += 1
    print(intmat,np.gcd.reduce(intmat.flatten()))
    intmat //= np.gcd.reduce(intmat.flatten())
    charpoly = labudde(intmat) #/ np.array([(2**(x*(53-np.amin(exp).astype(np.float64)))) for x in range(n+1)])
    print(charpoly)
    return intmat, None
#hessenberg_prime(np.array([[4,3,2],[2,3,4],[4,3,5]])) #paper example
def labudde(mat): #https://ipsen.math.ncsu.edu/ps/charpoly3.pdf
    n = len(mat); k = n - 1
    c = np.zeros(mat.shape, dtype=mat.dtype)
    c[0,0]  = -mat[0,0]
    c[0,1] = c[0,0] - mat[1,1]
    c[1,1] = mat[0,0]*mat[1,1] - mat[0,1] * mat[1,0]
    def betaprod(s, e): return multiprod(mat[d,d-1] for d in range(s, e+1)) if s <= e else 1
    for i in range(2, k+1):
        c[0,i] = c[0,i-1] - mat[i,i]
        for j in range(1, i):
            c[j, i] = c[j, i-1] - mat[i,i]*c[j-1,i-1] - sum(mat[i-m-1,i]*betaprod(i-m, i)*c[j-m-2,i-m-2] for m in range(0, j-1)) - mat[i-j,i]*betaprod(i-j+1, i)
        c[i,i] = -mat[i,i]*c[i-1,i-1] - sum(mat[i-m-1,i]*betaprod(i-m, i)*c[i-m-2,i-m-2] for m in range(0, i-1)) - mat[0,i]*betaprod(1, i)
    for i in range(k+1, n):
        c[0,i] = c[0,i-1] - mat[i,i]
        if k >= 1:
            for j in range(1, k+1):
                c[j, i] = c[j, i-1] - mat[i,i]*c[j-1,i-1] - sum(mat[i-m-1,i]*betaprod(i-m, i)*c[j-m-2,i-m-2] for m in range(0, j-1)) - mat[i-j,i]*betaprod(i-j+1, i)
    return [1] + [c[j,n-1] for j in range(0, k+1)]
def compute_powertrace_quartic(mat):
    n = len(mat)
    tr = [sum(B[i][i] for i in range(len(B)))]
    M = mat
    while len(tr) != n:
        M = matMul(M, mat) 
        tr.append(sum(M[i][i] for i in range(len(M))))
    return tr
def compute_charpoly(mat):    
    return labudde(hessenberg_gaussian(mat)[0])
def test_hess_qr():
    dim = 5
    mat = np.random.rand(dim, dim)*2-1 #+ np.random.rand(dim, dim)*2j-1j
    hessenberg_scipy(mat)
    hessenberg_householder(mat)
    hessenberg_givens(mat)
    hessenberg_fastgivens(mat)
    hessenberg_arnoldi(mat)
    hessenberg_gaussian(mat)
    #hessenberg_gaussian_int(mat)
    #hessenberg_prime(mat)
    qr_linalg(mat)
    qr_householder(mat)
    qr_givens(mat)
    qr_fastgivens(mat)
    qr_hessenberg(mat)
    qr_mgs(mat)
    assert np.allclose(compute_charpoly(mat), np.poly(mat)), (compute_charpoly(mat), np.poly(mat))
#test_hess_qr()
def power_traces(mat):
    charpoly = compute_charpoly(mat)
#[factoriallcms(n) for n in range(1, 15)]
#https://arxiv.org/pdf/1805.12498.pdf
def hafnian_eff(mat, isInt=False, isLoop=False):
  if (len(mat) & 1) != 0: return 0
  n = len(mat) // 2
  import functools, math
  if isInt: fact = math.factorial(n); nfact = multiprod([fact] * n)
  if n == 0: return 1
  h = 0
  for X in powerset(list(range(0, n))):
    AZ = matix(mat, [z for y in ((2*x, 2*x+1) for x in X) for z in y])
    colswap = functools.reduce(directSum, [[[0, 1], [1, 0]]] * len(X))
    B = matMul(colswap, AZ) #paper says column swap but this is a row swap AZ*colswap is a column swap, does not matter as long as loop correction is opposite to it
    #minpoly = minimalPolynomial(B)
    #characteristicPolynomial(B)
    tr = [sum(B[i][i] for i in range(len(B)))]
    Bpow = B
    while len(tr) != n:
      Bpow = matMul(Bpow, B)
      #if len(tr) >= len(minpoly)-1:
      #  assert sum(Bpow[i][i] for i in range(len(Bpow))) == -sum((0 if j >= len(minpoly) else minpoly[j])*tr[j + len(tr)-len(minpoly)+1] for j in range(len(minpoly)-1))
      tr.append(sum(Bpow[i][i] for i in range(len(Bpow))))
      #tr.append(-sum((0 if j >= len(minpoly) else minpoly[j])*tr[j + len(tr)-len(minpoly)+1] for j in range(len(minpoly)-1)))
    if isLoop:
      v = [[AZ[i][i] for i in range(len(AZ))]]; vt = [[x] for x in v[0]]
      loopCorrections = [] #[matMul(matMul(v,  colswap), vt)[0][0]]
      #w = [[v[0][i^1]] for i in range(len(v[0]))]
      w = matMul(colswap, vt)
      #paper has mistake that (XB)^(k-1) where it should be (B^(k-1))X
      #v*B^(k-1)X*v^T=v*(XAz)^(k-1)X*v^T=v*Az^(k-1)*X^k*v^T
      while len(loopCorrections) != n:
        #Bpow = B if len(loopCorrections) == 1 else matMul(Bpow, B)
        #loopCorrections.append(matMul(matMul(v, matMul(Bpow, colswap)), vt)[0][0])
        loopCorrections.append(matMul(v, w)[0][0])
        w = matMul(B, w)
        #w = matMul(AZ, w); w = matMul(colswap, w)
      if isInt: lbda = [0] + [tr[k-1] * (fact // k) + loopCorrections[k-1] * fact for k in range(1, n+1)]
      else: lbda = [0] + [tr[k-1] / (2 * k) + loopCorrections[k-1] / 2 for k in range(1, n+1)]
    else:
      if isInt: lbda = [0] + [tr[k-1] * (fact // k) for k in range(1, n+1)]
      else: lbda = [0] + [tr[k-1] / (2 * k) for k in range(1, n+1)]
    if isInt: poly, powfact, fixfact, = [1], nfact, fact
    else: poly, fixfact = [1], 1
    z = 0
    for j in range(1, n+1):
      if isInt: fixfact //= j; powfact //= fact
      else: fixfact *= j
      poly = mulPolyR(lbda, poly, None)
      if len(poly) > n:
        if isInt: z += ((poly[n] * fixfact) * powfact) << (n-j)
        else: z += poly[n] / fixfact
    h += -z if ((n - len(X)) & 1) != 0 else z
  return (h >> n) // (fact * nfact) if isInt else h
def run_haftest():
    haftest = (
  ([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]], 1),
  ([[-1, 1, 1, -1, 0, 0, 1, -1], [1, 0, 1, 0, -1, 0, -1, -1], [1, 1, -1, 1, -1, -1, 0, -1], [-1, 0, 1, -1, -1, 1, -1, 0], [0, -1, -1, -1, -1, 0, 0, -1], [0, 0, -1, 1, 0, 0, 1, 1], [1, -1, 0, -1, 0, 1, 1, 0], [-1, -1, -1, 0, -1, 1, 0, 1]], 4),
  ([[1, 1, 0, 0, 0, 0, 0, 1, 0, 0], [1, 1, -1, 0, -1, 1, 1, 1, 0, -1], [0, -1, -1, -1, 0, -1, -1, 0, -1, 1], [0, 0, -1, 1, -1, 1, -1, 0, 1, -1], [0, -1, 0, -1, -1, -1, -1, 1, -1, 1], [0, 1, -1, 1, -1, 1, -1, -1, 1, -1], [0, 1, -1, -1, -1, -1, 1, 0, 0, 0], [1, 1, 0, 0, 1, -1, 0, 1, 1, -1], [0, 0, -1, 1, -1, 1, 0, 1, 1, 1], [0, -1, 1, -1, 1, -1, 0, -1, 1, 1]], -13),
  ([[-1, 0, -1, -1, 0, -1, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 1, -1, -1, -1, -1], [-1, 0, 0, 1, 0, 0, 0, 1, -1, 1, -1, 0], [-1, 0, 1, -1, 1, -1, -1, -1, 0, -1, -1, -1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, -1, 0], [-1, -1, 0, -1, 0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 0, -1, 0, 1, 1, -1, -1, 0, 1, 0], [1, 1, 1, -1, 0, 1, -1, 1, -1, -1, -1, -1], [-1, -1, -1, 0, 0, 1, -1, -1, -1, 1, -1, 0], [0, -1, 1, -1, 1, 1, 0, -1, 1, -1, 1, 1], [0, -1, -1, -1, -1, 1, 1, -1, -1, 1, 0, -1], [0, -1, 0, -1, 0, 0, 0, -1, 0, 1, -1, 1]], 13),
  ([[-1, 1, 0, 1, 0, -1, 0, 0, -1, 1, -1, 1, 0, -1], [1, -1, 1, -1, 1, 1, -1, 0, -1, 1, 1, 0, 0, -1], [0, 1, 1, 1, -1, 1, -1, -1, 0, 0, -1, 0, -1, -1], [1, -1, 1, -1, 1, 0, 1, 1, -1, -1, 0, 0, 1, 1], [0, 1, -1, 1, 0, 1, 0, 1, -1, -1, 1, 1, 0, -1], [-1, 1, 1, 0, 1, 1, -1, 0, 1, -1, -1, -1, 1, -1], [0, -1, -1, 1, 0, -1, -1, -1, 0, 1, -1, 0, 1, -1], [0, 0, -1, 1, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1], [-1, -1, 0, -1, -1, 1, 0, 0, 1, 1, 0, 1, -1, 0], [1, 1, 0, -1, -1, -1, 1, -1, 1, 1, 1, 0, 1, 0], [-1, 1, -1, 0, 1, -1, -1, 0, 0, 1, -1, 0, -1, 0], [1, 0, 0, 0, 1, -1, 0, 0, 1, 0, 0, 1, 1, 1], [0, 0, -1, 1, 0, 1, 1, 0, -1, 1, -1, 1, 1, -1], [-1, -1, -1, 1, -1, -1, -1, 1, 0, 0, 0, 1, -1, -1]], 83),
  ([[0, 4.7, 4.6, 4.5], [4.7, 0, 2.1, 0.4], [4.6, 2.1, 0, 1.2], [4.5, 0.4, 1.2, 0]], 16.93),
  ([[0, 0, 1, -1, 1, 0, -1, -1, -1, 0, -1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 0, -1, -1, -1, -1, 0, 1, 1, 1, 1, 0, -1, -1, 0, 0, 1, 1, -1, 0, 0], [-1, -1, 0, 1, 0, 1, -1, 1, -1, 1, 0, 0, 1, -1, 0, 0, 0, -1, 0, -1, 1, 0, 0, 0], [1, 0, -1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -1, -1, -1, 1, 0, -1], [-1, 0, 0, -1, 0, 0, 1, -1, 0, 1, -1, -1, -1, 1, 1, 0, 1, 1, 1, 0, -1, 1, -1, -1], [0, 1, -1, -1, 0, 0, 1, -1, -1, -1, 0, -1, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1], [1, 1, 1, 0, -1, -1, 0, -1, -1, 0, 1, 1, -1, 0, 1, -1, 0, 0, 1, -1, 0, 0, 0, -1], [1, 1, -1, -1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, -1, 1, 0, 0], [1, 1, 1, -1, 0, 1, 1, 0, 0, -1, 1, -1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 1, 1, -1], [0, 0, -1, 0, -1, 1, 0, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 1, -1, -1], [1, -1, 0, 0, 1, 0, -1, 0, -1, -1, 0, 0, 1, 0, 0, -1, 0, -1, -1, -1, -1, -1, 1, -1], [-1, -1, 0, 0, 1, 1, -1, -1, 1, 0, 0, 0, -1, 0, 0, -1, 0, -1, -1, 0, 1, -1, 0, 0], [0, -1, -1, -1, 1, -1, 1, 0, -1, 0, -1, 1, 0, 1, -1, -1, 1, -1, 1, 0, 1, -1, 1, -1], [-1, -1, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 1, -1, -1, 0, 1, 0, -1, -1], [-1, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, -1, -1, 0, -1, -1], [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 1, -1, -1, -1, 0, -1, -1], [0, 1, 0, -1, -1, 1, 0, -1, 1, -1, 0, 0, -1, -1, -1, 0, 0, -1, 1, 0, 0, -1, -1, 1], [-1, 0, 1, 1, -1, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1, 1, 0, 1, 1, -1, -1, -1, 1], [0, 0, 0, 1, -1, 0, -1, -1, 1, 0, 1, 1, -1, 1, -1, 1, -1, -1, 0, 1, 1, 0, 0, -1], [0, -1, 1, 1, 0, -1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, -1, -1, 0, 0, 0, 1, 0], [-1, -1, -1, 1, 1, 0, 0, 1, 0, 0, 1, -1, -1, -1, 1, 1, 0, 1, -1, 0, 0, 0, 0, 0], [0, 1, 0, -1, -1, 0, 0, -1, -1, -1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0], [-1, 0, 0, 0, 1, 0, 0, 0, -1, 1, -1, 0, -1, 1, 1, 1, 1, 1, 0, -1, 0, -1, 0, 1], [-1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, -1, -1, 1, 0, 0, 0, -1, 0]], -6773))
    import timeit
    for mat, sol in haftest:
      #assert hafnian(mat) == sol
      #import numpy as np;from thewalrus import hafnian
      #print(hafnian(np.array(mat), loop=True))
      assert hafnian_perf_match(mat, isLoop=True) == hafnian_eff(mat, isInt=isinstance(sol, int), isLoop=True), (hafnian_perf_match(mat, isLoop=True), hafnian_eff(mat, isInt=isinstance(sol, int), isLoop=True))
      assert hafnian_perf_match(mat) == sol
      assert hafnian_ryser_time(mat) == sol
      assert hafnian_eff(mat, isInt=isinstance(sol, int)) == sol, (hafnian_eff(mat), sol)
      #print(timeit.timeit(lambda: hafnian_ryser_time(mat), number=1000))
      #print(timeit.timeit(lambda: hafnian_eff(mat, isInt=True), number=1000))
      print(sol)
#run_haftest()

def groqCounter(): #MXM matmul by constant, VXM comparison by constant, VXM mask to clamp to 1/0 or -1/0
    import numpy as np
    bits = 5
    counter = np.zeros((1, bits), dtype=np.int8)
    for i in range(1 << bits):
        nextcounter = counter @ np.triu(np.ones((bits, bits), dtype=np.int8), k=1) #count of trailing ones ~CTZ
        print(counter)
        counter = np.where(nextcounter >= np.arange(0, bits, dtype=np.int8), (counter+1)%2, counter)
#groqCounter()
def groqCounterVXM(): #VXM addition by one, VXM bitwise_and with power of 2 constants, VXM mask to clamp to 1/0 or -1/0
    import numpy as np
    bits = 5
    counter = np.zeros((1, bits), dtype=np.int64)
    for i in range(1 << bits):
        realcounter = ((counter & np.array([1<<i for i in range(bits)], dtype=np.int64)) != 0).astype(np.int64)
        print(realcounter)
        counter = counter + 1
#groqCounterVXM()

def add64(tensors1, tensors2, issub=False):
    res = []
    maskadd = g.constant_tensor(shape=(1, dim), dtype=g.uint32)
    maskadd.data = np.array([[(1<<31)-1]*dim], dtype=np.uint32)
    if issub: res.append(g.sub(tensors1[0], tensors2[0]))
    else: res.append(g.add(tensors1[0], tensors2[0]))
    carry1 = tensors1[0].greater(maskadd)
    carry2 = tensors2[0].greater(maskadd)
    notcarry3 = res[0].less_equal(maskadd)
    #carry=carry1&carry2 | (carry1 ^ carry2) & !carry3
    #borrow=!carry1&carry2 | (carry1 & carry2) & carry3
    certaincarry = g.bitwise_and(carry1, carry2)
    resultcarry = g.bitwise_and(g.bitwise_xor(carry1, carry2), notcarry3)
    carry = g.bitwise_or(certaincarry, resultcarry)
    res.append(g.add(g.add(tensors1[1], tensors2[1]), carry))
    return res

def karatsuba_mul32(tensor1, tensor2, dim):
    shiftqrt = g.constant_tensor(shape=(1, dim), dtype=g.uint8)
    shiftqrt.data = np.array([[16]*dim], dtype=np.uint8)
    maskqrt = g.constant_tensor(shape=(1, dim), dtype=g.uint32)
    maskqrt.data = np.array([[(1<<16)-1]*dim], dtype=np.uint32)
    with g.ResourceScope(name="mask1", is_buffered=True, time=0) as mask1:
        tl1 = g.bitwise_and(tensor1, maskqrt).write(name="tl1")
    with g.ResourceScope(name="mask2", is_buffered=True, predecessors=[mask1], time=None) as mask2:
        tl2 = g.bitwise_and(tensor2, maskqrt).write(name="tl2")
        #tl2 = inst.bitwise_and(tensor2, maskqrt)
    with g.ResourceScope(name="shift1", is_buffered=True, predecessors=[mask2], time=None) as shift1:
        tu1 = g.right_shift(tensor1, shiftqrt).write(name="tu1")
    with g.ResourceScope(name="shift2", is_buffered=True, predecessors=[shift1], time=None) as shift2:        
        tu2 = g.right_shift(tensor2, shiftqrt).write(name="tu2")
    with g.ResourceScope(name="sub1", is_buffered=True, predecessors=[shift2], time=None) as sub1:
        diff1 = g.add(tu1, tl1).write(name="diff1")
    with g.ResourceScope(name="sub2", is_buffered=True, predecessors=[sub1], time=None) as sub2:
        diff2 = g.add(tu2, tl2).write(name="diff2")
    with g.ResourceScope(name="subfix1", is_buffered=True, predecessors=[sub2], time=None) as subfix1:
        difffix1 = g.bitwise_and(diff1, maskqrt).write(name="difffix1")
    with g.ResourceScope(name="subfix2", is_buffered=True, predecessors=[subfix1], time=None) as subfix2:        
        difffix2 = g.bitwise_and(diff2, maskqrt).write(name="difffix2")
    #9 bits * 9 bits = all([i*j==((i&255)*(j&255)+(i&256)*j+(j&256)*(i&255)) for i in range(512) for j in range(512)])
    with g.ResourceScope(name="extra1", is_buffered=True, predecessors=[subfix2], time=None) as extra1:
        ex1 = g.mask(g.greater(diff1, maskqrt), diff2).cast(g.uint32).write(name="ex1")
    with g.ResourceScope(name="extra2", is_buffered=True, predecessors=[extra1], time=None) as extra2:
        ex2 = g.mask(g.greater(diff2, maskqrt), difffix1).cast(g.uint32).write(name="ex2")
    with g.ResourceScope(name="mul1", is_buffered=True, predecessors=[extra2], time=None) as mul1:
        z0 = karatsuba_mul16(tl1, tl2)
    with g.ResourceScope(name="mul2", is_buffered=True, predecessors=[mul1], time=None) as mul2:
        z2 = g.left_shift(karatsuba_mul16(tu1, tu2), shiftqrt).write(name="z2")
    with g.ResourceScope(name="mul3", is_buffered=True, predecessors=[mul2], time=None) as mul3:
        z1 = g.left_shift(karatsuba_mul16(difffix1, difffix2), shiftqrt).write(name="z1")
    with g.ResourceScope(name="add1", is_buffered=True, predecessors=[mul3], time=None) as add1:
        res1 = g.add(g.add(g.add(g.left_shift(z2, shiftqrt), z0), z1), ex1).write(name="res1")
    with g.ResourceScope(name="add2", is_buffered=True, predecessors=[add1], time=None) as add2:
        res2 = g.sub(g.add(res1, ex2), z2).write(name="res2")
    with g.ResourceScope(name="s1", is_buffered=True, predecessors=[add2], time=None) as s1:
        z0s = g.left_shift(z0, shiftqrt).write(name="z0s")
    with g.ResourceScope(name="add3", is_buffered=True, predecessors=[s1], time=None) as add3:
        res = g.sub(res2, z0s).write(name="res")
    return res
    
def karatsuba_mul16(tensor1, tensor2, dim):
    #if tensor1.tensor_type == g.uint16:
    shiftqrt = g.constant_tensor(shape=(1, dim), dtype=g.uint8)
    shiftqrt.data = np.array([[8]*dim], dtype=np.uint8)
    shifthalf = g.constant_tensor(shape=(1, dim), dtype=g.uint8)
    shifthalf.data = np.array([[16]*dim], dtype=np.uint8)
    maskqrt = g.constant_tensor(shape=(1, dim), dtype=g.uint16)
    maskqrt.data = np.array([[(1<<8)-1]*dim], dtype=np.uint16)
    with g.ResourceScope(name="mask1", is_buffered=True, time=0) as mask1:
        tl1 = g.bitwise_and(tensor1, maskqrt, alus=[0]).write(name="tl1")
        tu2 = g.right_shift(tensor2, shiftqrt, alus=[4]).write(name="tu2")
    with g.ResourceScope(name="mask2", is_buffered=True, time=1) as mask2:
        tl2 = g.bitwise_and(tensor2, maskqrt, alus=[0]).write(name="tl2")
        tu1 = g.right_shift(tensor1, shiftqrt, alus=[4]).write(name="tu1")
    with g.ResourceScope(name="sub1", is_buffered=True, predecessors=[mask2], time=None) as sub1:
        diff1 = g.add(tu1, tl1, alus=[0]).write(name="diff1")
    with g.ResourceScope(name="sub2", is_buffered=True, predecessors=[sub1], time=None) as sub2:
        diff2 = g.add(tu2, tl2, alus=[4]).write(name="diff2")
    with g.ResourceScope(name="subfix1", is_buffered=True, predecessors=[sub2], time=None) as subfix1:
        difffix1 = g.bitwise_and(diff1, maskqrt).write(name="difffix1")
    with g.ResourceScope(name="subfix2", is_buffered=True, predecessors=[subfix1], time=None) as subfix2:        
        difffix2 = g.bitwise_and(diff2, maskqrt).write(name="difffix2")
    #9 bits * 9 bits = all([i*j==((i&255)*(j&255)+(i&256)*j+(j&256)*(i&255)) for i in range(512) for j in range(512)])
    with g.ResourceScope(name="extra1", is_buffered=True, predecessors=[subfix2], time=None) as extra1:
        ex1 = g.left_shift(g.mask(g.greater(diff1, maskqrt), diff2).cast(g.uint32), shifthalf).write(name="ex1")
    with g.ResourceScope(name="extra2", is_buffered=True, predecessors=[extra1], time=None) as extra2:
        ex2 = g.left_shift(g.mask(g.greater(diff2, maskqrt), difffix1).cast(g.uint32), shifthalf).write(name="ex2")
    with g.ResourceScope(name="mul1", is_buffered=True, predecessors=[extra2], time=None) as mul1:
        z0 = g.mul(tl1, tl2, time=0).cast(g.uint32).write(name="z0")
    with g.ResourceScope(name="mul2", is_buffered=True, predecessors=[mul1], time=None) as mul2:
        z2 = g.left_shift(g.mul(tu1, tu2, time=0).cast(g.uint32), shiftqrt).write(name="z2")
    with g.ResourceScope(name="mul3", is_buffered=True, predecessors=[mul2], time=None) as mul3:
        z1 = g.left_shift(g.mul(difffix1, difffix2, time=0).cast(g.uint32), shiftqrt).write(name="z1")
    with g.ResourceScope(name="add1", is_buffered=True, predecessors=[mul3], time=None) as add1:
        res1 = g.add(g.add(g.add(g.left_shift(z2, shiftqrt), z0), z1), ex1).write(name="res1")
    with g.ResourceScope(name="add2", is_buffered=True, predecessors=[add1], time=None) as add2:
        res2 = g.sub(g.add(res1, ex2), z2).write(name="res2")
    with g.ResourceScope(name="s1", is_buffered=True, predecessors=[add2], time=None) as s1:
        z0s = g.left_shift(z0, shiftqrt).write(name="z0s")
    with g.ResourceScope(name="add3", is_buffered=True, predecessors=[s1], time=None) as add3:
        res = g.sub(res2, z0s).write(name="res")
    return res
    
def num_to_bits(num, chunks):
    res, shp = np.repeat(num[np.newaxis,...], chunks, axis=0), [chunks] + [1] * len(num.shape)
    return ((res >> np.arange(0, 7 * chunks, 7).reshape(shp)) & np.array([((1 << 7)-1)]*(chunks-1)+[-1]).reshape(shp)).astype(np.int8)
def bits_to_num(num, offset=7):
    return np.sum(num.astype(np.int64) << np.arange(-offset, 7 * num.shape[1]-offset, 7), axis=1) #the high byte can be 0/-1 or this overflows...
def normalize_doubles(num, dimension, fractionbits=63):
    mantissas, exponents = np.frexp(num)
    maxexp = np.amax(exponents, axis=dimension)
    adjustmant = mantissas / (1 << ((maxexp[:,np.newaxis] if dimension==1 else maxexp) - exponents))
    return maxexp, np.rint(np.ldexp(adjustmant, fractionbits)).astype(np.int64) #(64, -62) bit fixed point integers
def renormalize_doubles(num, exp):
    return num.astype(np.float64) / (2 ** exp.astype(np.float64))
def vector_complex_to_real(cplx):
    dim = cplx.shape[1]//2 #len(cplx)//2
    return cplx[:,:dim] + cplx[:,dim:]*1j #return cplx[:dim] + cplx[dim:]*1j
def vector_real_to_complex(vec):
    return np.hstack((vec.real, vec.imag))
def matrix_real_to_complex(mtx):
    return np.vstack((np.hstack((mtx.real, -mtx.imag)), np.hstack((mtx.imag, mtx.real))))
def cond_runfunc(ft, ff, x, cond):
    return ft(x) if cond else ff(x)
def extract_int8(var):
    return g.concat_vectors([x.reinterpret(g.int8).split_vectors([1]*4)[0] for x in var], (len(var), var[0].shape[1]))
WEST, EAST = 0, 1
def get_slice4(drctn, start, end, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S4(" + str(start) + "-" + str(end) + "), B1(" + str(bank) + ")"
def get_slice2(drctn, start, end, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S2(" + str(start) + "-" + str(end) + "), B1(" + str(bank) + ")"
def get_slice1(drctn, start, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S1(" + str(start) + "), B1(" + str(bank) + ")"
s16rangeW = list(range(25, 27+1))+list(range(29, 37+1))+list(range(39,42+1))
s16rangeE = list(range(26, 27+1))+list(range(29,42+1))
def get_slice16(drctn, slices, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S16(" + ",".join(str(x) for x in slices) + "), B1(" + str(bank) + ")"
class VecMatMul(g.Component):
    def __init__(self, chunks, dim, **kwargs):
        super().__init__(**kwargs)
        self.chunks, self.dim = chunks, dim
        self.maskqrt, self.maskqrttop, self.shiftqrt, self.zeros = [], [], [], []
        for drctn in (WEST, EAST):
            self.maskqrt.append(g.concat_inner_splits(
                [g.from_data(np.array([[(1<<7)-1]*dim], dtype=np.int32), layout=get_slice4(drctn, 8, 11, drctn))] * chunks))
            self.maskqrttop.append(g.concat_inner_splits(
                [g.from_data(np.array([[(1<<7)-1]*dim], dtype=np.int32), layout=get_slice4(drctn, 32, 35, drctn))] * (chunks-1)+
                [g.from_data(np.array([[-1]*dim], dtype=np.int32), layout=get_slice4(drctn, 32, 35, drctn))]))
            self.shiftqrt.append(g.concat_inner_splits(
                [g.from_data(np.array([[7]*dim], dtype=np.int32), layout=get_slice4(drctn, 12, 15, drctn))] * chunks))
            self.shiftqrt.append(g.concat_inner_splits(
                [g.from_data(np.array([[7]*dim], dtype=np.int32), layout=get_slice4(drctn, 8, 11, drctn))] * chunks))
            g.add_mem_constraints(self.maskqrttop, self.maskqrttop, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints([self.shiftqrt[-1]], [self.maskqrt[-1]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            #g.add_mem_constraints([maskqrt[-1], maskqrttop[-1], shiftqrt[-1]], [maskqrt[-1], maskqrttop[-1], shiftqrt[-1]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            self.zeros.append(g.zeros(shape=(1,dim), dtype=g.int32, layout=get_slice4(drctn, 4, 7, 0)))
            self.zeros.append(g.zeros(shape=(1,dim), dtype=g.int32, layout=get_slice4(drctn, 4, 7, 1)))
            g.add_mem_constraints([self.zeros[-1]], [self.zeros[-2]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        self.maskreqs = []
        self.splitreqs = []
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            self.maskreqs.append([])
            self.splitreqs.append([])
            for i in range(chunks+1):            
                self.maskreqs[-1].append(g.tensor.create_storage_request(layout=get_slice4(drctn, 4, 7, plane)))
                self.splitreqs[-1].append(g.tensor.create_storage_request(layout=get_slice4(drctn, 0, 3, plane)))
    def build(self, tvec, tmat, inittime=0):
        # Instantiate matmul component.   
        # Build matmul component.
        g.add_mem_constraints(self.maskqrttop, tmat, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
    
        final_result = []
        #split_result = []
        #allshifts = [zeros[0], zeros[1]]
        MATMULDELAY = 76 #4 (stream group size minus 1) for S16(25-27,29-37,39-42) to boundary, 9 for SXM crossing, 5 (stream group size) for weight install completion (not depended upon)
        #3 delay for S1(43) to allow weight install, 9 for SXM and 1 MXM weight crossing, #chunks for vector install (none of these are depended on)
        #weight install takes 3 ticks to start to finish on MXM basic
        #13 for MXM basic multiplication, 34 for SXM accumulation, 19 to stream across (6 leaving SXM, 10 stream crossings, 3 entering ALU)
        #4+9+3+13+34+19 - 6 = 76 #-6 because the S4(8-11)
        #mxm_rqs = [g.tensor.create_mxm_request(planes=[x], num_planes=1) for x in range(4)]
        #g.latch(maskqrt[1-drctn].read(streams=SG4_TO[3]), alu=3)
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            if plane == 0:
                split_result = []
                allshifts = [self.zeros[drctn*2], self.zeros[drctn*2+1]]
            #mm = nn.MatMul(time=0, buffer_output=False, planes=mxm_rqs[drctn*2+plane].planes)
            result_mt = [g.concat_inner_splits(x) for x in zip(*(g.split_vectors(x, [self.dim]*self.chunks) for x in g.split_inner_splits(tmat[drctn*2+plane])))]
            #result_mt = g.split_vectors(tmat[drctn*2+plane], [self.dim]*self.chunks)
            SG4_FROM = g.SG4_E if drctn == WEST else g.SG4_W
            SG4_TO = g.SG4_W if drctn == WEST else g.SG4_E
            rev_last_alu = [4] if drctn == WEST else [7]
            rev_alu = [6] if drctn == WEST else [9]
            first_alu = [0] if drctn == WEST else [3]
            second_alu = [1] if drctn == WEST else [2]
            dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P" + "_t" + str(inittime)
            t = inittime+plane*max(self.dim//16, 10)
            for i in range(self.chunks):
                with g.ResourceScope(name="matmul" + dirstr + str(i), is_buffered=True, time=t) as pred: #mm.end_time==20 #for plane 0 returns on SG4_E[4] #for nn.matmul time=plane*21+(20+12+9+1)*i due to SXM DIST
                    #result_mt[i] = mm.build(tvec[drctn*2+plane], result_mt[i])
                    #g.clear_mxm(planes=[plane], time=0)
                    mxm_rq = g.tensor.create_mxm_request(planes=[drctn*2+plane], num_planes=1)
                    iw = g.install_weights(result_mt[i], planes=mxm_rq, time=0) #.read(streams=g.SG16_W[plane] if drctn == WEST else g.SG16_E[plane]), time=0 if plane==0 else -18)
                    #iw = g.load_weight_buffer(result_mt[i], planes=mxm_rq, time=0)
                    #print(tvec[drctn*2+plane].shape, tvec[drctn*2+plane].physical_shape, result_mt[i].shape, result_mt[i].physical_shape)
                    result_mt[i] = tvec[drctn*2+plane].matmul(iw, planes=mxm_rq, num_planes=1, accum_input=None, time=0)
                    #result_mt[i] = tvec[drctn*2+plane].matmul(result_mt[i], planes=[plane], time=0)
                    split_result.append(g.concat_inner_splits(g.split_vectors(result_mt[i], [1]*self.chunks)))
                    #must be an arithmetic right shift (sign filled), not logical, but with signed types, this occurs
                    if i == 0:
                        nextmasks = g.bitwise_and(split_result[-1], self.maskqrt[drctn].read(streams=SG4_FROM[3]), alus=rev_last_alu, output_streams=SG4_TO[3]).write(name="mask" + dirstr + str(i), storage_req=self.maskreqs[drctn*2+plane][i])
                        split_result[-1] = split_result[-1].write(name="split" + dirstr + str(i), storage_req=self.splitreqs[drctn*2+plane][i])
                    else:
                        masks = g.concat_inner_splits(g.split_inner_splits(nextmasks)[1:] + [self.zeros[drctn*2+plane]]).read(streams=SG4_FROM[1])
                        shifts = g.right_shift(split_result[-2].read(streams=SG4_FROM[2 if drctn == WEST else 6]), self.shiftqrt[2*drctn].read(streams=SG4_FROM[0]), alus=[0] if drctn == WEST else [15], output_streams=SG4_FROM[2 if drctn == WEST else 6])
                        split_result[-1] = g.add(g.add(shifts, masks, alus=[1] if drctn == WEST else [14], output_streams=SG4_FROM[2 if drctn == WEST else 6]), split_result[-1], alus=rev_alu, output_streams=SG4_TO[2 if drctn == WEST else 6])
                        if i != self.chunks - 1:
                            nextmasks = g.bitwise_and(self.maskqrt[1-drctn].read(streams=SG4_TO[3 if drctn==WEST else 7]), split_result[-1], alus=[4] if drctn==WEST else [11], output_streams=SG4_TO[3 if drctn==WEST else 7]).write(name="mask" + dirstr + str(i), storage_req=self.maskreqs[drctn*2+plane][i])
                        else:
                            nextshifts = g.right_shift(split_result[-1], self.shiftqrt[2*(1-drctn)+1].read(streams=SG4_TO[3 if drctn==WEST else 7]), alus=[4] if drctn==WEST else [11], output_streams=SG4_TO[3 if drctn==WEST else 7]).write(name="shiftpre" + dirstr, storage_req=self.maskreqs[drctn*2+plane][i])
                        split_result[-1] = split_result[-1].write(name="split" + dirstr + str(i), storage_req=self.splitreqs[drctn*2+plane][i])
                        g.add_mem_constraints(split_result[:-1], [split_result[-1]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                    allshifts.append(nextshifts if i == self.chunks-1 else nextmasks)
                    g.add_mem_constraints(allshifts[:-1], [allshifts[-1]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                    t += max(self.dim//16*2, (27 if i==0 else 30))
            #with complex multiplication at dimension 80 and 64-bit, we surely need 2 of these rounds to converge
            with g.ResourceScope(name="finsma" + dirstr, is_buffered=True, time=t+MATMULDELAY-18): #+predecessors=[pred], time=None) as pred: #they are not all fitting in int8 yet but after first iteration, the final computation can occur
                cursplit = split_result[-1].read(streams=SG4_FROM[5])
                shifts = g.concat_inner_splits([self.zeros[drctn*2+plane]] + g.split_inner_splits(nextshifts)[:-1]).read(streams=SG4_FROM[3 if drctn==WEST else 7])
                masks = g.bitwise_and(cursplit, self.maskqrttop[drctn].read(streams=SG4_FROM[4], time=0), alus=rev_last_alu, output_streams=SG4_FROM[4])
                split_result.append(g.add(masks, shifts, alus=rev_alu, output_streams=SG4_TO[2 if drctn == WEST else 6]))
                #nextshifts = g.bitwise_and(split_result[-1], self.maskqrt[1-drctn].read(streams=SG4_TO[0]), alus=[0] if drctn == WEST else [15], output_streams=SG4_TO[3 if drctn==WEST else 7]).write(name="fixmask" + dirstr, storage_req=self.maskreqs[drctn*2+plane][self.chunks])
                nextshifts = g.right_shift(split_result[-1], self.shiftqrt[2*(1-drctn)+1].read(streams=SG4_TO[0]), alus=[0] if drctn == WEST else [15], output_streams=SG4_TO[3 if drctn==WEST else 7]).write(name="fixshift" + dirstr, storage_req=self.maskreqs[drctn*2+plane][self.chunks])
                split_result[-1] = split_result[-1].write(name="finsplit" + dirstr, storage_req=self.splitreqs[drctn*2+plane][self.chunks])
                g.add_mem_constraints(split_result[:-1], [split_result[-1]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                allshifts.append(nextshifts) #allshifts.append(nextmasks)
                g.add_mem_constraints(allshifts[:-1], [allshifts[-1]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                
            #chunks-1 correct 7-bit int8s
            #final adjustment for between 0-7 bit addition extra bits, for 64x64-127x127 this is exactly 7 bits
            with g.ResourceScope(name="fixsma" + dirstr, is_buffered=True, time=t+MATMULDELAY+9+10): #predecessors=[pred], time=None) as pred:
                cursplit = split_result[-1].read(streams=SG4_FROM[3])
                #masks = g.concat_inner_splits(g.split_inner_splits(nextmasks)[1:] + [self.zeros[drctn*2+plane]]).read(streams=SG4_FROM[0])
                #shifts = g.right_shift(cursplit, self.shiftqrt[2*drctn].read(streams=SG4_FROM[1], time=0), alus=first_alu, output_streams=SG4_FROM[1])
                masks = g.bitwise_and(cursplit, self.maskqrttop[drctn].read(streams=SG4_FROM[1], time=0), alus=first_alu, output_streams=SG4_FROM[1])
                shifts = g.concat_inner_splits([self.zeros[drctn*2+plane]] + g.split_inner_splits(nextshifts)[:-1]).read(streams=SG4_FROM[0])
                #split_result.append(g.add(shifts, masks, alus=second_alu, output_streams=SG4_TO[1]).write(name="fixsplit" + dirstr, layout=get_slice4(drctn, 0, 3, plane)))
                final_result.append(extract_int8(g.split_inner_splits(g.add(shifts, masks, alus=second_alu, output_streams=SG4_TO[1]))).write(name="extract" + dirstr, layout=get_slice1(drctn, 43, plane)))
                #final_result.append(g.add(shifts, masks, alus=second_alu, output_streams=SG4_TO[1]).write(name="extract" + dirstr, storage_req=self.splitreqs[drctn*2+plane][self.chunks]))
            #print("Cycle time: ", t+MATMULDELAY+9+10+31+19) #31 through ALU, 19 to write to S43
        g.add_mem_constraints(tvec + final_result, final_result, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        return final_result, t+MATMULDELAY+9+10+31+19

class ColumnSwapVector(g.Component):
    def __init__(self, dim, **kwargs):
        self.permmap = []
        for drctn in (WEST, EAST):
            self.permmap.append(g.from_data(np.array(inst.encode_permute_map([(x&~1)+1-(x&1) for x in range(dim)])).astype(np.uint8), layout=get_slice1(drctn, 43)))
    def build(self, tvec, inittime = 0):        
        #g.add_mem_constraints(tmat, self.permmap, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        final_result = []
        for drctn in (WEST, EAST):
            dirstr = ("W" if drctn == WEST else "E") + "_t" + str(inittime)
            SG1_FROM = g.SG1_E if drctn == WEST else g.SG1_W
            SG1_TO = g.SG1_W if drctn == WEST else g.SG1_E
            #SG2_TO = g.SG2_W if drctn == WEST else g.SG2_E
            with g.ResourceScope(name="colswap" + dirstr, time=inittime): #this should use the distributor, not the slow more resource intensive permutor!
                final_result.extend(g.split_inner_splits(g.permute_inner(g.concat_inner_splits(tvec[drctn*2:drctn*2+2]), self.permmap[drctn], drctn, [SG1_TO[0], SG1_TO[24]], SG1_FROM[0], time=0).write(name="swap" + dirstr, layout=get_slice1(drctn, 43))))
        g.add_mem_constraints(tvec, final_result, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        return final_result #8 cycles for permute_map to finish stream to SXM perm + 48 cycle delay (#chunks*2 initialization in parallel) + 4 cycles to exit permute + #chunks*2 cycles to write output = 80 cycles 
def flatten_zip(z): return [item for sublist in z for item in sublist]
def flatten_unzip(z, interleave=2): return list(zip(*zip(*([iter(z)] * interleave))))
class UnpackComplexMatrix(g.Component):
    def __init__(self, chunks, dim, **kwargs):
        super().__init__(**kwargs)
        self.chunks, self.dim = chunks, dim
        self.imagpermmap = [g.from_data(np.array(inst.encode_permute_map(list(range(dim, dim*2)) + list(range(dim)) + list(range(dim*3, dim*4)) + list(range(dim*2, dim*3)))).astype(np.uint8), layout=get_slice1(hemi, 43)) for hemi in (WEST, EAST)]
        #map_tensor = g.from_data(np.array([i for i in range(16)]*dim*2//16 + [16]*dim*2, dtype=np.uint8))
        #map_tensor_rev = g.from_data(np.array([16]*dim*2 + [i for i in range(16)]*dim*2//16, dtype=np.uint8))
        #map_tensor_rev, add1_tensor = [], []
        self.map_tensor, self.negate_tensor = [], []
        for hemi in (WEST, EAST):
            self.map_tensor.append(g.from_data(np.array([[-1]*dim*2+[0]*dim*2], dtype=np.int8), layout=get_slice4(hemi, 0, 3, 0)))
            #map_tensor_rev.append(g.from_data(np.array([[0]*dim*2+[-1]*dim*2], dtype=np.int8)))
            self.negate_tensor.append(g.from_data(np.array([1]*dim+[-1]*dim+[0]*dim*2, dtype=np.int8), layout=get_slice4(hemi, 4, 7, 0)))
            #add1_tensor.append(g.from_data(np.array([0]*dim+[1]*dim, dtype=np.int8), layout=get_slice4(hemi, 8, 11, 0)))        
    def build(self, tmat):
        tmatsplit = g.split_vectors(tmat, [self.chunks*self.dim*self.dim//320]*2) #split into WEST and EAST (self.chunks//2, dim*dim*2//320)
        with g.ResourceScope(name="imagreal", is_buffered=True, time=0) as pred:
            imagreal = []
            for hemi in (WEST, EAST):
                imagreal.append(g.permute_inner(tmatsplit[hemi], self.imagpermmap[hemi], hemi, [g.SG1[0], g.SG1[24]], g.SG1[0], time=0).write(name="imagreal" + ("W" if hemi==WEST else "E"), layout=get_slice1(hemi, 43, 1))) #layout=get_slice16(hemi, s16rangeW if hemi==WEST else s16rangeE)))
        #g.add_mem_constraints(tmatsplit, imagreal, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        g.add_mem_constraints(self.imagpermmap, imagreal, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        tmatjoin = [g.concat_vectors(flatten_zip(zip(g.split_vectors(tmatsplit[0], [self.dim*self.dim*2//320]*(self.chunks//2)), g.split_vectors(imagreal[0], [self.dim*self.dim*2//320]*(self.chunks//2)))), (self.chunks*self.dim*self.dim*2//320, 320)),
                    g.concat_vectors(flatten_zip(zip(g.split_vectors(tmatsplit[1], [self.dim*self.dim*2//320]*(self.chunks//2)), g.split_vectors(imagreal[1], [self.dim*self.dim*2//320]*(self.chunks//2)))), (self.chunks*self.dim*self.dim*2//320, 320))] #(self.chunks//2, self.dim*2*self.dim*2//320, 320)
        with g.ResourceScope(name="shifted", is_buffered=True, time=None, predecessors=[pred]) as pred:
            result = []
            for hemi in (WEST, EAST):
                result_mt_split = flatten_unzip(g.split_vectors(tmatjoin[hemi], [self.dim//2]*self.chunks))
                result_mt_split = g.concat_vectors((*result_mt_split[1], *result_mt_split[0]), (self.chunks*self.dim//2, 320))
                result_mt = g.bitwise_and(result_mt_split.read(streams=g.SG4[2 if hemi==WEST else 5]), self.map_tensor[hemi].read(streams=g.SG4[1 if hemi==WEST else 4]), alus=[1 if hemi==WEST else 6], output_streams=g.SG4[2 if hemi==WEST else 5], time=0)
                negalu = g.tensor.create_alu_request([2 if hemi==WEST else 5])
                result_mt_split = g.split_vectors(result_mt, [self.chunks//2*self.dim//2]*2)
                result_mt_split[0] = g.split_vectors(result_mt_split[0], [self.dim//2]*(self.chunks//2))
                result_mt_split[1] = g.split_vectors(g.mul(result_mt_split[1], self.negate_tensor[hemi].read(streams=g.SG4[1 if hemi==WEST else 4]), alus=negalu, output_streams=g.SG4[2 if hemi==WEST else 5]), [self.dim//2]*(self.chunks//2))
                result_mt = g.concat_vectors(flatten_zip(zip(result_mt_split[1], result_mt_split[0])), (self.chunks*self.dim//2, 320))
                #dist = g.distribute_8(tmat, self.map_tensor, distributor_req=4, map_stream_req=g.SG1[0], time=0)
                #dist_rev = g.distribute_8(tmat, self.map_tensor_rev, distributor_req=6, map_stream_req=g.SG1[0])
                #result_mt = g.transpose_null(dist, transposer_req=2, stream_order=[0,1,2,3,4,5,6,7]) #.write(name="test", layout="-1, S8")
                #160 shift is a concurrency of 1...SG1 36 delay in south direction
                result_mt2 = g.shift(tmatjoin[hemi], self.dim*2, permutor_id=hemi, shift_src=[inst.NEW_SRC], dispatch_set=inst.DispatchSet.SET_0, input_streams=g.SG1[0 if hemi==WEST else 12], output_streams=g.SG1[0 if hemi==WEST else 12], time=500) #.write(name="test", layout="-1, S16")
                negalu = g.tensor.create_alu_request([0 if hemi==WEST else 7])
                result_mt2_split = flatten_unzip(g.split_vectors(result_mt2, [self.dim//2]*self.chunks))
                result_mt2_split[0] = g.split_vectors(g.mul(g.concat_vectors(result_mt2_split[0], (self.chunks//2*self.dim//2, 320)), self.negate_tensor[hemi].read(streams=g.SG4[1 if hemi==WEST else 4]), alus=negalu, output_streams=g.SG4[3 if hemi==WEST else 4]), [self.dim//2]*(self.chunks//2))
                result_mt2 = g.concat_vectors(flatten_zip(zip(result_mt2_split[0], result_mt2_split[1])), (self.chunks*self.dim//2, 320))
                #result_mt2=tmatjoin[hemi].read(streams=g.SG8[2 if hemi==WEST else 3], time=500)
                #result_rev_mt = g.shift(tmatjoin[hemi], -self.dim*2, permutor_id=hemi, shift_src=[inst.NEW_SRC], dispatch_set=inst.DispatchSet.SET_0, input_streams=g.SG1[0], output_streams=g.SG1[0], time=0)
                result.append(g.concat_vectors(flatten_zip(zip(g.split_vectors(result_mt, [1]*(self.chunks*self.dim//2)), g.split_vectors(result_mt2, [1]*(self.chunks*self.dim//2)))), (self.chunks*self.dim, self.dim*2*2)).write(name="origmat" + str(hemi), layout=get_slice16(hemi, s16rangeW if hemi==WEST else s16rangeE, 1)))
                #result = g.concat_vectors(imagreal, (self.chunks*self.dim*self.dim*2//320, 320))    
        #g.resolve_storage_requests() #g.update_resources(pred, "shifted")
        #result = g.from_addresses(np.vstack((result[0].addrs.reshape(self.chunks//2, self.dim*2), result[1].addrs.reshape(self.chunks//2, self.dim*2))).reshape(-1, g.int8.size), 320, g.int8, "truncresult")
        g.add_mem_constraints(result, tmatsplit, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        result = g.concat_vectors(result, (self.chunks*self.dim*2, 320))
        return result
class DiagonalExtractor(g.Component):
    def __init__(self, chunks, dim, **kwargs):
        super().__init__(**kwargs)
        self.chunks, self.dim = chunks, dim      
        self.identmat, self.allones = [], []
        for hemi in (WEST, EAST):
            self.identmat.append(g.from_data(np.hstack((np.tile(np.eye(dim, dtype=np.int8), (2, 1)), np.zeros((dim*2, dim*3), dtype=np.int8))), layout=get_slice4(hemi, 0, 3, 0)))
            self.allones.append(g.ones((1,320), dtype=g.int8, layout=get_slice1(hemi, 43, 0)))
    def build(self, tmat):
        tmatsplit = g.split_vectors(tmat, [self.chunks*self.dim*2*self.dim*2//320]*2)
        with g.ResourceScope(name="identmask", is_buffered=True, time=0) as pred:
            result = []
            for hemi in (WEST, EAST):
                result_mt = g.mask(
                    g.concat_vectors(flatten_zip(flatten_unzip(g.split_vectors(g.concat_vectors([self.identmat[hemi]] * (self.chunks//2), (self.dim*self.chunks, 320)), [1]*(self.chunks*self.dim)), 16)), (self.chunks*self.dim, 320)).read(streams=g.SG4[1 if hemi==WEST else 4]),
                    g.concat_vectors(flatten_zip(flatten_unzip(g.split_vectors(tmatsplit[hemi], [1]*(self.chunks*self.dim)), 16)), (self.chunks*self.dim, 320)).read(streams=g.SG4[2 if hemi==WEST else 5], time=0),
                    alus=[0 if hemi==WEST else 7], output_streams=g.SG4[2 if hemi==WEST else 5])
                result_mt = g.concat_vectors(flatten_zip(flatten_unzip(g.split_vectors(result_mt, [1]*(self.chunks*self.dim)), self.chunks*self.dim//16)), (self.chunks*self.dim, 320))
                result.append(result_mt.write(layout=get_slice16(hemi, s16rangeW if hemi==WEST else s16rangeE, 0)))
        g.add_mem_constraints(tmatsplit, result, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        with g.ResourceScope(name="extract", is_buffered=True, time=None, predecessors=[pred]) as pred:
            for hemi in (WEST, EAST):
                result_mt = result[hemi].read(streams=g.SG16)
                mxm_rq = g.tensor.create_mxm_request(planes=[hemi*2+0], num_planes=1)
                result_mt = g.split_vectors(result_mt, [self.dim*2] * (self.chunks//2))
                for i in range(self.chunks//2):
                    with g.ResourceScope(name="matmulextract" + str(i), is_buffered=False, time=10+i*30):
                        iw = g.install_weights(result_mt[i], planes=mxm_rq, time=0)
                        result_mt[i] = extract_int8([self.allones[hemi].matmul(iw, planes=mxm_rq, num_planes=1, accum_input=None, time=0)])
                result[hemi] = g.concat_vectors(result_mt, (self.chunks//2,self.dim*2)).write(name="origvec" + str(hemi), layout=get_slice1(hemi, 43, 1))
        #g.resolve_storage_requests()
        #result = g.from_addresses(np.vstack((result[0].addrs.reshape(self.chunks//2, 1), result[1].addrs.reshape(self.chunks//2, 1))).reshape(-1, g.int8.size), 320, g.int8, "vecresult")
        g.add_mem_constraints(result, self.allones, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        result = g.concat_vectors(result, (self.chunks, self.dim*2))
        return result
class LoopCorrections(g.Component):
    def __init__(self, chunks, dim, **kwargs):
        super().__init__(**kwargs)
        self.chunks, self.dim = chunks, dim
        #self.matmem, self.matzero = [], []
        #for hemi in (WEST, EAST):
            #self.matmem.append(inst.malloc([hemi], s16rangeW if hemi==WEST else s16rangeE, [0], chunks*dim*2*2 // 16, "B_W" if hemi==WEST else "B_E").reshape(chunks*2, dim*2))
            #self.matmem.append(g.tensor.create_storage_request(layout="S16(" + ",".join(str(x) for x in (s16rangeW if hemi==WEST else s16rangeE)) + "), B1(0)"))
            #self.matzero.append(g.zeros(shape=(16, 320), dtype=g.int8, layout="-1, H1(" + ("W" if hemi==WEST else "E") + "), S16(" + ",".join(str(x) for x in (s16rangeW if hemi==WEST else s16rangeE)) + "), B1(0)"))
        #g.add_mem_constraints(matzero, imagreal, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        #g.add_mem_constraints(matzero, tmatsplit, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        #inst.free_mem_by_key("B_W"); inst.free_mem_by_key("E_W")
    def build(self, tmat):
        with g.ResourceScope(name="init", is_buffered=True, time=0) as pred:
            result_mt = UnpackComplexMatrix(self.chunks, self.dim).build(tmat)
        with g.ResourceScope(name="next", is_buffered=True, time=None, predecessors=[pred]):
            result_mt = DiagonalExtractor(self.chunks, self.dim).build(result_mt)
        return result_mt        
def main():
    import timeit
    dim = 80 #dim X dim complex matrix
    bitsize = 64 #for fixed point representation will round up to nearest multiple of 7
    chunks = (bitsize + 7-1)//7 #ceiling division to be exact
      
    max_dim_bits = (dim*2).bit_length() #complex domain
    #64 signed bits * 64 signed bits = 63 unsigned bits * 63 unsigned bits + sign bit = 127 bits...
    signedhighbit = bitsize * 2 - 1 + max_dim_bits
    signedlowbit = signedhighbit - bitsize
    #print(signedlowbit, signedhighbit) for dim=64-127, 70 and 134

    
    # IMP DETAILS on using nn.Matmul component:
    # - Expects both inputs as rank-2 tensor.
    # - The inner dimension on both tensors should match.

    # Create 2 input tensors.
    #t1 = g.input_tensor(shape=(100, 1000), dtype=g.float16, name="A")
    #t2 = g.input_tensor(shape=(400, 1000), dtype=g.float16, name="B")
    #long double has 63+1 significant bits + 1 sign bits, 65-8=57 8*7=56+1 requires int8, 8 7-bit int8 and a 1-bit int8
    #long double has 16 exponent bits, can reduce it to 15, fits in int16
    #must first adjust each row vs column to fixed point to handle the additions
    #t1 = g.input_tensor(shape=(1, dim), dtype=g.uint16, name="A")
    #t2 = g.input_tensor(shape=(1, dim), dtype=g.uint16, name="B")
    #slices (16, 20, 24, and 28 on the east, and 16, 20, 24, 28, and 38 on the west) are reserved for system use
    #WEST8_0, WEST8_1, EAST8_0, EAST8_1 = "25-27,29-33", "34-37,39-42", "26-27,29-34", "35-42"
    WEST8_0, WEST8_1 = list(range(25, 27+1))+list(range(29, 33+1)), list(range(34, 37+1))+list(range(39, 42+1))
    EAST8_0, EAST8_1 = list(range(26, 27+1))+list(range(29, 34+1)), list(range(35, 42+1))
    #WEST16, EAST16 = "25-27,29-37,39-42", "26-27,29-42" #(10-15,17-19,21-23,25-27,29-37,39-40,42-43)
    """
    tzeromat = []
    for drctn, group in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
        tzeromat.append(g.concat_vectors([g.zeros(shape=(1,dim*2), dtype=g.int8, layout="H1("  + ("W" if drctn==WEST else "E") + "), -1, S8(" + ((WEST8_1 if group==1 else WEST8_0) if drctn==WEST else (EAST8_1 if group==1 else EAST8_0)) + "), B1(0)")]*chunks*dim*2, (chunks*dim*2, dim*2)))
        #tzeromat.append(g.concat_vectors([g.zeros(shape=(dim*2,dim*2), dtype=g.int8, layout="H1(" + ("W" if drctn==WEST else "E") + "), -1, S8(" + ((WEST8_1 if group==1 else WEST8_0) if drctn==WEST else (EAST8_1 if group==1 else EAST8_0)) + "), B1(0)")]*chunks, (chunks*dim*2, dim*2)))
    """
    """
    tvec, tmat = [], []
    for group in (0,):
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P"
            tvec.append(g.input_tensor(shape=(chunks, dim*2), dtype=g.int8, name="A" + dirstr + str(group), layout=get_slice1(drctn, 43, plane)))
            tmat.append(g.input_tensor(shape=(chunks*dim*2, dim*2), dtype=g.int8, name="B" + dirstr + str(group), layout=f"H1(" + ("W" if drctn==WEST else "E") + "), -1, S16(" + (WEST16 if drctn==WEST else EAST16) + "), B1(" + str(plane) + ")")) 
            #tmat.append(g.input_tensor(shape=(chunks*dim*2, dim*2), dtype=g.int8, name="B" + dirstr + str(group), layout=f"H1(" + ("W" if drctn==WEST else "E") + "), -1, S8(" + ((WEST8_1 if group==1 else WEST8_0) if drctn==WEST else (EAST8_1 if group==1 else EAST8_0)) + "), B1(" + str(plane) + ")"))
    g.add_mem_constraints(tvec, tvec, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
    g.add_mem_constraints(tmat+tzeromat, tmat+tzeromat, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
    """
    #tmat = g.input_tensor(shape=(chunks*dim*dim*2//320, 320), dtype=g.int8, name="AZ", layout=f"H2(W,E), -1, S8(" + WEST8_1 + "), B1(0)")
    tmat = g.input_tensor(shape=(chunks*dim*dim*2//320, 320), dtype=g.int8, name="AZ",
        storage_req=g.tensor.create_storage_request(storage=g.tensor.Storage(address_tensor=np.hstack((inst.malloc(hemis=["W"], slices=s16rangeW, banks=[0], count=chunks*dim*dim*2//320//2//16, reserve_key="INP_W").flatten(),
            inst.malloc(hemis=["W"], slices=WEST8_1, banks=[0], count=1, reserve_key="INP_WEXT").flatten(),
            inst.malloc(hemis=["E"], slices=s16rangeE, banks=[0], count=chunks*dim*dim*2//320//2//16, reserve_key="INP_E").flatten(),
            inst.malloc(hemis=["E"], slices=EAST8_1, banks=[0], count=1, reserve_key="INP_EEXT").flatten())).reshape(1, 1, chunks*dim*dim*2//320, 1))))
    #tmat = g.input_tensor(shape=(chunks*dim*dim*2//320, 320), dtype=g.int8, name="AZ", layout=f"H2(W,E), -1, S16(" + WEST16 + "), B1(0)")
    #chunks*dim*dim*2//320==400, 400//16==25, the W will get 1 more than the E bank...
    
    #tveccombine = [g.concat_inner_splits([tvec[i], tvec[i+4]]) for i in range(4)]
    #tmatcombine = [g.concat_inner_splits([g.concat_vectors([tmat[i], tzeromat[i&~1]], (chunks*dim*2*2, dim*2)), g.concat_vectors([tmat[i+4], tzeromat[(i&~1)+1]], (chunks*dim*2*2, dim*2))]) for i in range(4)]
    #print(tveccombine[0].shape, tveccombine[0].physical_shape, tmatcombine[0].shape, tmatcombine[0].physical_shape)
    parallel = 1 #len(tvec)

    #print_utils.infoc(
    #    "\nBuilding FP16 matmul for input tensors " + ", ".join(["{} x {}".format(tvec[i].shape, tmat[i].shape) for i in range(parallel)])
    #)
    #lc = VecMatMul(chunks, dim*2*2)
    #lc = VecMatMul(chunks, dim*2)
    #result_mt, t = lc.build(tvec, tmat)
    #result_mt, t = lc.build(tveccombine, tmatcombine)
    #result_mt = colswap_vector(tvec, tmat, dim*2)
    #result_mt, _ = lc.build(result_mt, tmat, t)
    result_mt = LoopCorrections(chunks, dim).build(tmat)
    g.resolve_storage_requests()

    print_utils.infoc("\nCompiling model ...")
    # Compile program to generate IOP. Also generate groqview JSON dump file and
    # check for potential stream conflicts.
    iop_file = g.compile(
        base_name="mm_fp", gen_vis_data=False, check_stream_conflicts=True, result_tensor=result_mt, #tree_conflicts=True, inspect_raw=True
    )
    json_file = g.write_visualizer_data("mm_fp")
    print("Have a GroqView:\n", "    % groqview --port 8888", json_file)
    g.check_stream_conflicts(json_file)

    # Generate random input data and oracle for comparision.
    #inp1 = np.random.randint(0, (1<<16)-1, size=t1.shape, dtype=np.uint16)
    #inp2 = np.random.randint(0, (1<<16)-1, size=t2.shape, dtype=np.uint16)
    #originpvec = [np.random.rand(dim)*2-1 + (np.random.rand(dim)*2j-1j) for _ in range(parallel)]
    #originpmat = [unitary_group.rvs(dim) for _ in range(parallel)]
    originpmat = unitary_group.rvs(dim)
    #import functools
    #originpmat = [np.array(functools.reduce(directSum, [[[0, 1], [1, 0]]] * (dim//2))) for _ in range(parallel)]
    """
    for i in range(parallel):
        realresult = np.hstack((originpvec[i].real, originpvec[i].imag)) @ np.hstack((originpmat[i].real, -originpmat[i].imag)).transpose()
        imagresult = np.hstack((originpvec[i].real, originpvec[i].imag)) @ np.hstack((originpmat[i].imag, originpmat[i].real)).transpose()
        singlematmult = np.hstack((originpvec[i].real, originpvec[i].imag)) @ np.vstack((np.hstack((originpmat[i].real, -originpmat[i].imag)), np.hstack((originpmat[i].imag, originpmat[i].real)))).transpose()
        result = singlematmult[:dim] + singlematmult[dim:]*1j
        resultalt = realresult + imagresult*1j
        actualresult = originpvec[i] @ originpmat[i].transpose()
        #print("Tolerance", max(abs(actualresult.reshape(-1) - result.reshape(-1))), max(abs(actualresult.reshape(-1) - resultalt.reshape(-1))))
    """
    #originpvec = [np.random.rand(dim)*2-1 for _ in range(parallel)]
    #originpmat = [np.random.rand(dim, dim)*2-1 for _ in range(parallel)] #unitary_group.rvs(dim).real
    #originpvec[0] = np.full((dim,), 9.5)
    #originpmat[0] = np.full((dim, dim), 9.5)
    #originpvec[1] = np.full((dim,), -((1 << 53)-1000000)/(1<<53)+2j)
    #originpmat[1] = np.full((dim, dim), -((1 << 53)-1000000)/(1<<53)+2j)
    #originpvec[0], originpmat[0] = originpvec[1], originpmat[1]
    #originpvec = np.ones((dim,), dtype=np.float64)
    #originpmat = np.ones((dim, dim), dtype=np.float64)

    #originpvec, originpmat = np.ones(dim, dtype=np.float64), np.ones((dim, dim), dtype=np.float64)
    #originpvec = np.random.randint(-(1<<63), (1<<63)-1, size=(dim), dtype=np.int64)
    #originpmat = np.random.randint(-(1<<63), (1<<63)-1, size=(dim, dim), dtype=np.int64)
    oracleres = [None]
    def oracle():
        #B = [originpmat[i].transpose().astype(np.clongdouble) for i in range(parallel)]
        #oracleres[0] = [((originpvec[i].astype(np.clongdouble) @ B[i]) @ B[i]).astype(np.cdouble) for i in range(parallel)]
        oracleres[0] = [np.vstack((originpmat.conjugate(), originpmat.imag + originpmat.real*1j)).diagonal()[np.newaxis,:]]
    toracle = timeit.timeit(oracle, number=10)/10
    print_utils.infoc("\nRunning on HW ...")
    np.set_printoptions(formatter={'int':hex}, threshold=sys.maxsize, floatmode='unique')
    # Create TSP runner and pass input dictionary of "tensor name" : "tensor data".
    runner = g.create_tsp_runner(iop_file)
    #inputs = {t1.name: inp1, t2.name: inp2}
    results = [None]
    def actual():
        fractionbits = 63
        inputs = {}
        """
        exp_inpvecs, exp_inpmats = [], []
        for i in range(parallel):
            exp_inpvec, normals = normalize_doubles(vector_real_to_complex(originpvec[i]), 0, fractionbits)
            inpvec = num_to_bits(normals, chunks)
            exp_inpmat, normals = normalize_doubles(matrix_real_to_complex(originpmat[i]), None, fractionbits) #dimension 1 will cause chaining issues without shift corrections
            #exp_inpmat, normals = np.zeros((dim*2,), dtype=np.int32), np.rint(np.ldexp(matrix_real_to_complex(originpmat[i]), fractionbits)).astype(np.int64)
            inpmat = num_to_bits(normals, chunks)
            inputs[tvec[i].name] = inpvec
            inputs[tmat[i].name] = inpmat.reshape((chunks*dim*2, dim*2))
            exp_inpvecs.append(exp_inpvec); exp_inpmats.append(exp_inpmat)
        """
        exp_inpmat, normals = normalize_doubles(vector_real_to_complex(originpmat), None, fractionbits)
        inputs[tmat.name] = num_to_bits(normals, chunks).reshape((chunks*dim*dim*2//320, 320))
        res = runner(**inputs)
        #result = bits_to_num(res[result_mt.name].reshape(chunks, dim*2, dim*2*2)[:,:,:dim*2].transpose(1, 2, 0).reshape(dim*2*dim*2, chunks), 7) #.reshape(chunks*dim, 320)[:,:160]
        #results[0] = []
        #results[0].append(vector_complex_to_real(renormalize_doubles(result, fractionbits - 7 - exp_inpmat).reshape(dim*2,dim*2)))
        result = bits_to_num(res[result_mt.name].reshape(chunks, 1, dim*2)[:,:,:dim*2].transpose(1, 2, 0).reshape(dim*2, chunks), 7) #.reshape(chunks*dim, 320)[:,:160]
        results[0] = []
        results[0].append(vector_complex_to_real(renormalize_doubles(result, fractionbits - 7 - exp_inpmat).reshape(1,dim*2)))
        """
        results[0] = []
        for i in range(parallel):
            result = bits_to_num(res[result_mt[i].name].reshape(chunks, dim*2).transpose(), 7)
            #the results come back truncating the lower 7*(chunks-1) bits
            results[0].append(vector_complex_to_real(renormalize_doubles(result, fractionbits - 7 - exp_inpvecs[i] - exp_inpmats[i])))
        """
    tactual = timeit.timeit(actual, number=1)/1
    print("CPU Time", toracle, "Groq Time", tactual)
    oracleres, results = oracleres[0], results[0]
    for i in range(parallel):
        print_utils.infoc("\nComparing results with oracle ...")
        print(originpmat[0], oracleres[i][0], results[i][0])
        print(originpmat[0][0], oracleres[i][0][0], results[i][0][0]) #numpy uses "round to nearest even" while Groq strategy uses "round to negative infinity", last bit only should be different
        #print(originpmat[40][0], oracleres[i][40][0], results[i][40][0])
        #print(originpmat[0][0], oracleres[i][80][0], results[i][80][0])
        #print(originpmat[40][0], oracleres[i][120][0], results[i][120][0])
        print([max(abs(oracleres[i][:,j].reshape(-1) - results[i][:,j].reshape(-1))) for j in range(dim)])
        #print([max(abs(oracleres[i][j].reshape(-1) - results[i][j].reshape(-1))) for j in range(dim*2)])
        max_atol = max(abs(oracleres[i].reshape(-1) - results[i].reshape(-1)))
        #print((np.frexp(oracleres[i].real)[0]*(1<<53)).astype(np.int64), (np.frexp(results[i].real)[0]*(1<<53)).astype(np.int64))
        if max_atol <= 0.001:
            print_utils.success(f"Test PASSED with a max tolerance of {max_atol}")
        else:
            print_utils.err(
                f"Test FAILED with a max tolerance of {max_atol} (should be <= 0.001)"
            )


if __name__ == "__main__":
    main()
