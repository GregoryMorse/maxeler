import xml.etree.ElementTree as ET
import os
project = "PermanentGlynnDFE"
builds = ["PermanentGlynn_singleSIM-"]#, "PermanentGlynn_dualSIM-"]
#project = "PermRepGlynnDFE"
#builds = ["PermRepGlynn_singleSIM-"]#, "PermRepGlynn_dualSIM-"]
filepats = ["SumUpPermDFEKernel", "PermanentGlynnDFEKernel_0", "InitializeColSumDFEKernel_1", "InitializeColSumDFEKernel_0"]
pxgpath = os.path.join(project, "builds", "simulation", "*.pxg")
import glob
import re
pxgfiles = glob.glob(pxgpath)
for pxgfile in pxgfiles:
    if not "-final-simulation" in pxgfile: continue
    if not any(build in pxgfile for build in builds) or not any(filepat in pxgfile for filepat in filepats): continue 
    print("Analyzing", pxgfile)
    tree = ET.parse(pxgfile)
    root = tree.getroot()
    nodedict, srcdict, destdict = {}, {}, {}
    for node in root.findall("./Node"):
        nodedict[node.attrib["id"]] = node 
    for edge in root.findall("./Edge"):
        if "src_node_id" in edge.attrib:
            if not edge.attrib["src_node_id"] in srcdict: srcdict[edge.attrib["src_node_id"]] = [] 
            srcdict[edge.attrib["src_node_id"]].append(edge)
        if "dst_node_id" in edge.attrib:
            if not edge.attrib["dst_node_id"] in destdict: destdict[edge.attrib["dst_node_id"]] = []
            destdict[edge.attrib["dst_node_id"]].append(edge)
    for fifo in root.findall("./Node[@type='NodeFIFO']"):
        print(fifo.find("Text").text + " Found", "ID:", fifo.attrib["id"], "Type:", fifo.find("Input").attrib["type"])        
        for edge in destdict[fifo.attrib["id"]]:
            source = nodedict[edge.attrib["src_node_id"]]
            s = []
            for st in source.find("OriginStackTrace").text.splitlines():
                m = re.match(r".*\.(.*)\((.*)\.maxj:([0-9]*)\)", st)
                if not m is None and m.group(2) in pxgfile:                
                    s.append((m.group(1) + ":" if len(s) == 0 else "") + m.group(3))
            if source.attrib["type"] == "NodeFIFO":
                print("Source FIFO", "ID:", source.attrib["id"])
            else: 
                print(",".join(s), "Source:", source.attrib["type"])
            for e in srcdict[source.attrib["id"]]:
                dest = root.find("./Node[@id='" + e.attrib["dst_node_id"] + "']")
                if dest.attrib["id"] == fifo.attrib["id"]: continue
                inp = e.attrib["dst_node_input"]
                s = []
                for st in dest.find("OriginStackTrace").text.splitlines():
                    m = re.match(r".*\.(.*)\((.*)\.maxj:([0-9]*)\)", st)
                    if not m is None and m.group(2) in pxgfile:                
                        s.append((m.group(1) + ":" if len(s) == 0 else "") + m.group(3))
                if dest.attrib["type"] == "NodeFIFO":
                    print("NOT DESTINATION FIFO", "ID:", dest.attrib["id"])
                else: 
                    print(",".join(s), "NOT DESTINATION:", dest.attrib["type"], "Input:", inp)
        for edge in srcdict[fifo.attrib["id"]]:
            dest = root.find("./Node[@id='" + edge.attrib["dst_node_id"] + "']")
            inp = edge.attrib["dst_node_input"]
            s = []       
            for st in dest.find("OriginStackTrace").text.splitlines():
                m = re.match(r".*\.(.*)\((.*)\.maxj:([0-9]*)\)", st)
                if not m is None and m.group(2) in pxgfile:                
                    s.append((m.group(1) + ":" if len(s) == 0 else "") + m.group(3))
            if dest.attrib["type"] == "NodeFIFO":
                print("Destination FIFO", "ID:", dest.attrib["id"]) 
            else: print(",".join(s), "Destination:", dest.attrib["type"], "Input:", inp)            
        
