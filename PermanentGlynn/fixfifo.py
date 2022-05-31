import xml.etree.ElementTree as ET
import os
class DAryHeap:
    def __init__(self, d):
        self.d = d
        self.heap = []
        self.end = 0
        self.pos = {} #lookup dictionary
    def insert(self, new, newkey):
        self.end+=1
        self.pos[newkey] = self.end
        self.heap.append((new, newkey))
        self.siftup(self.end)
    def siftup(self, i):
        key = self.heap[i-1]
        while i > 1 and self.heap[(i - 2) // self.d] > key:
            self.heap[i - 1] = self.heap[(i - 2) // self.d]
            self.pos[self.heap[i - 1][1]] = i
            i = (i-2) // self.d + 1
        self.heap[i-1] = key
        self.pos[key[1]] = i
    def siftdown(self, i):
        key = self.heap[i-1]
        j = self.d*(i-1)+1+self.d
        while j <= self.end:
            j = min((j-x for x in range(self.d)), key=lambda x: self.heap[x-1])
            if self.heap[j-1] < key:
                self.heap[i-1] = self.heap[j-1]
                self.pos[self.heap[i-1][1]] = i; i=j; j=self.d*(i-1)+1+self.d
            else: j=self.end+self.d
        #j-x for x in range(self.d) if j-x <= self.end
        j = min(range(self.end, j-self.d, -1), key=lambda x: self.heap[x-1], default=self.end+self.d)
        if j <= self.end and self.heap[j-1] < key:
            self.heap[i-1] = self.heap[j-1]
            self.pos[self.heap[i-1][1]] = i; i = j
        self.heap[i-1] = key
        self.pos[key[1]] = i
    def deletemin(self):
        tmp = self.heap[0]
        self.heap[0] = self.heap[self.end-1]
        self.pos[self.heap[0][1]] = 1
        self.heap[self.end-1] = tmp
        self.end -= 1
        self.siftdown(1)
        del self.pos[tmp[1]]
        return self.heap.pop()[1]
    def decreasekey(self, i, delta):
        self.heap[i-1] = (self.heap[i-1][0] - delta, self.heap[i-1][1])
        self.siftup(i)
def dijkstra(G, s):
    g, c = G
    P, S = {s}, set()
    M = set(g) - P
    K, p = {s: 0}, {s: None} #dist, prev
    while len(P) != 0:
        u = min(P, key=lambda x: K[x]) #should use a min-heap...
        P.remove(u); S.add(u)
        for v in g[u]:
            if v in P and (not v in K or K[u] + c[(u,v)] < K[v]):
                K[v] = K[u] + c[(u,v)]; p[v] = u
            if v in M:
                K[v] = K[u] + c[(u,v)]; p[v] = u; M.remove(v); P.add(v)
    return K, p
def nuutila_reach_scc(succ, subg=None):
  index, s, sc, sccs, reach = 0, [], [], {}, {}
  indexes, lowlink, croot, stackheight = {}, {}, {}, {}
  def nuutila(v, index):
    #index/D, lowlink/CCR/component candidate root, final component/C, component stack height/H
    stack = [(v, None, iter(succ[v]))]
    while len(stack) != 0:
      v, w, succv = stack.pop()
      if w is None:
        indexes[v], lowlink[v] = index, v
        stackheight[v] = len(sc)
        index += 1
        s.append(v)
      elif not w in croot:
        if indexes[lowlink[w]] < indexes[lowlink[v]]: lowlink[v] = lowlink[w]
      else: sc.append(croot[w])
      for w in succv:
        if not subg is None and not w in subg: continue
        forward_edge = False
        if not w in indexes: stack.append((v, w, succv)); stack.append((w, None, iter(succ[w]))); break #index = nuutila(w, index)
        else: forward_edge = indexes[v] < indexes[w]
        if not w in croot:
          if indexes[lowlink[w]] < indexes[lowlink[v]]: lowlink[v] = lowlink[w]
        elif not forward_edge: #(v, w) is not a forward edge - whether w on stack or not...
          sc.append(croot[w])
      else:
        if lowlink[v] == v:
          sccs[v] = set()
          is_self_loop = s[-1] != v or v in succ[v]
          while True:
            w = s.pop()
            sccs[v].add(w)
            if w == v: break
          reach[v] = set(sccs[v]) if is_self_loop else set()
          if not subg is None and len(subg) == len(sccs[v]): return index
          for x in sccs[v]: croot[x] = v
          l = set()
          while len(sc) != stackheight[v]:
            x = sc.pop()
            l.add(x)
            for x in sorted(l, reverse=True, key=lambda y: indexes[y]):
              if not x in reach[v]:
                reach[v] |= reach[x]; reach[v] |= sccs[x]
    return index
  for v in succ:
    if (subg is None or v in subg) and not v in indexes:
      index = nuutila(v, index)
  return sccs, reach #keys are SCCs, values are reachable vertices

project = "PermanentGlynnDFE"
builds = ["PermanentGlynn_singleSIM-", "PermanentGlynn_dualSIM-", "PermanentGlynn_singleSIMF-", "PermanentGlynn_dualSIMF-"]
#project = "PermRepGlynnDFE"
#builds = ["PermRepGlynn_singleSIM-", "PermRepGlynn_singleSIMF-", "PermRepGlynn_dualSIM-", "PermRepGlynn_dualSIMF-"]
filepats = ["SumUpPermDFEKernel", "PermanentGlynnDFEKernel_0", "InitializeColSumDFEKernel_1", "InitializeColSumDFEKernel_0"]
pxgpath = os.path.join(project, "builds", "simulation", "*.pxg")
import glob
import re
pxgfiles = glob.glob(pxgpath)
def get_stacktrace(node):
    s = []
    for st in node.find("OriginStackTrace").text.splitlines():
        m = re.match(r".*\.(.*)\((.*)\.maxj:([0-9]*)\)", st)
        if not m is None and m.group(2) in pxgfile:
            s.append(m.group(1) + ":" + m.group(3))
    return s
for pxgfile in pxgfiles:
    if not "-final-simulation" in pxgfile: continue
    if not any(build in pxgfile for build in builds) or not any(filepat in pxgfile for filepat in filepats): continue 
    print("Analyzing", pxgfile)
    tree = ET.parse(pxgfile)
    root = tree.getroot()
    nodedict, srcdict, destdict, g = {}, {}, {}, {}
    for node in root.findall("./Node"):
        nodedict[node.attrib["id"]] = node
        g[int(node.attrib["id"])] = set()
    for edge in root.findall("./Edge"):
        if "src_node_id" in edge.attrib:
            if not edge.attrib["src_node_id"] in srcdict: srcdict[edge.attrib["src_node_id"]] = [] 
            srcdict[edge.attrib["src_node_id"]].append(edge)
        if "dst_node_id" in edge.attrib:
            if not edge.attrib["dst_node_id"] in destdict: destdict[edge.attrib["dst_node_id"]] = []
            destdict[edge.attrib["dst_node_id"]].append(edge)
        g[int(edge.attrib["src_node_id"])].add(int(edge.attrib["dst_node_id"]))
    for fifo in root.findall("./Node[@type='NodeFIFO']"):
        print(fifo.find("Text").text + " Found", "ID:", fifo.attrib["id"], "Type:", fifo.find("Input").attrib["type"])        
        for edge in destdict[fifo.attrib["id"]]:
            source = nodedict[edge.attrib["src_node_id"]]
            if source.attrib["type"] == "NodeFIFO":
                print("Source FIFO", "ID:", source.attrib["id"])
            else: 
                print(",".join(get_stacktrace(source)), "Source:", source.attrib["type"])
            for e in srcdict[source.attrib["id"]]:
                dest = nodedict[e.attrib["dst_node_id"]]
                if dest.attrib["id"] == fifo.attrib["id"]: continue
                inp = e.attrib["dst_node_input"]
                if dest.attrib["type"] == "NodeFIFO":
                    print("NOT DESTINATION FIFO", "ID:", dest.attrib["id"])
                else: 
                    print(",".join(get_stacktrace(dest)), "NOT DESTINATION:", dest.attrib["type"], "Input:", inp)
        for edge in srcdict[fifo.attrib["id"]]:
            dest = nodedict[edge.attrib["dst_node_id"]]
            inp = edge.attrib["dst_node_input"]
            if dest.attrib["type"] == "NodeFIFO":
                print("Destination FIFO", "ID:", dest.attrib["id"]) 
            else: print(",".join(get_stacktrace(dest)), "Destination:", dest.attrib["type"], "Input:", inp)            
    #for x in g:
    #    fanout = [y for y in g[x] if nodedict[str(y)].attrib["type"] == "NodeRegister"]
    #    if len(fanout) >= 2:
    #        print("Pipeline fanout detected:", len(fanout), ",".join(get_stacktrace(nodedict[str(x)])), "[" + "--".join([",".join(get_stacktrace(nodedict[str(z)])) for z in fanout]) + "]")
    scc, reach = nuutila_reach_scc(g)
    revscc, sources, sinks, gnew, gpred, c = {}, set(scc), set(scc), {}, {}, {}
    for x in scc:
        for y in scc[x]: revscc[y] = x
    for x in scc: #build DAG from SCC information, identify graph sources
        gnew[x] = {revscc[z] for y in scc[x] for z in g[y] if revscc[z] != x}
        sources -= gnew[x]
    for x in scc: #constructor predecessors
        if len(gnew[x]) != 0: sinks.remove(x)
        for y in gnew[x]:
            c[(x ,y)] = 1 if nodedict[str(x)].attrib["type"] == "NodeRegister" else 0
            if not y in gpred: gpred[y] = set()
            gpred[y].add(x)
    #unnecessary pipelining detection - collapse to SCCs as in loops does not occur anyway
    #determine path length down any path, all will have the same latency
    #determine shortest path when only counting latency of registers
    #if shortest path is not 0, then there are registers down all pathways - must determine an optimal set corresponding to a graph slice to show for removal - with maximum bitsize...
    for s in sources:
        K, p = dijkstra((gnew, c), s)        
        #print(nodedict[str(s)].attrib["type"], get_stacktrace(nodedict[str(s)]), [(get_stacktrace(nodedict[str(d)]), K[d]) for d in sinks if d in K])
    for x in scc: #merge any paths with NodeRegister or not NodeRegister on both sides node1->node2->node3 transforms to node1->node3 
        if x in gpred and len(gpred[x]) == 1 and len(gnew[x]) == 1:
            y = next(iter(gpred[x]))
            b1 = nodedict[str(y)].attrib["type"] == "NodeRegister"
            b2 = nodedict[str(x)].attrib["type"] == "NodeRegister"
            if b1 == b2:
                z = next(iter(gnew[x]))
                gnew[y].remove(x); gnew[y].add(z)
                gpred[z].remove(x); gpred[z].add(y)
                del gnew[x]; del gpred[x]
    for x in set(gnew.keys()):
        if x in gpred and nodedict[str(x)].attrib["type"] != "NodeRegister" and all(nodedict[str(y)].attrib["type"] != "NodeRegister" for y in gpred[x]):
            for z in gnew[x]: gpred[z].remove(x)
            for y in gpred[x]:
                gnew[y].remove(x)
                for z in gnew[x]:
                    gnew[y].add(z)
                    gpred[z].add(y)
            del gnew[x]; del gpred[x]
    with open(os.path.basename(pxgfile) + ".dot", "w") as f:
        f.write("digraph G {\n" + ";\n".join(str(x) + "->" + str(y) for x in gnew for y in gnew[x]) + ";\n" +
            ";\n".join(str(x) + "[label=\"" + nodedict[str(x)].attrib["type"] + "\"]" for x in gnew) + ";\n}")
    #print(sources, len(gnew))
    
