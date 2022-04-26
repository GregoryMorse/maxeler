import xml.etree.ElementTree as ET
import os
project = "PermRepGlynnDFE"
pxgpath = os.path.join(project, "builds", "simulation", "*.pxg")
import glob
import re
pxgfiles = glob.glob(pxgpath)
for pxgfile in pxgfiles:
    print("Analyzing", pxgfile)
    tree = ET.parse(pxgfile)
    root = tree.getroot()
    for fifo in root.findall("./Node[@type='NodeFIFO']"):
        print(fifo.find("Text").text + " Found", "ID:", fifo.attrib["id"])
        for edge in root.findall("./Edge[@dst_node_id='" + fifo.attrib["id"] + "']"):
            source = root.find("./Node[@id='" + edge.attrib["src_node_id"] + "']")
            for st in source.find("OriginStackTrace").text.splitlines():
                m = re.match(r".*\((.*)\.maxj:([0-9]*)\)", st)
                if not m is None and m.group(1) in pxgfile:                
                    print(m.group(1) + ":" + m.group(2), "Source:", source.attrib["type"])
            if source.attrib["type"] == "NodeFIFO":
                print("Source FIFO", "ID:", source.attrib["id"]) 
            for e in root.findall("./Edge[@src_node_id='" + source.attrib["id"] + "']"):
                dest = root.find("./Node[@id='" + e.attrib["dst_node_id"] + "']")
                if dest.attrib["id"] == fifo.attrib["id"]: continue
                inp = e.attrib["dst_node_input"]
                for st in dest.find("OriginStackTrace").text.splitlines():
                    m = re.match(r".*\((.*)\.maxj:([0-9]*)\)", st)
                    if not m is None and m.group(1) in pxgfile:                
                        print(m.group(1) + ":" + m.group(2), "NOT DESTINATION:", dest.attrib["type"], "Input:", inp)
                if dest.attrib["type"] == "NodeFIFO":
                    print("NOT DESTINATION FIFO", "ID:", dest.attrib["id"]) 
        for edge in root.findall("./Edge[@src_node_id='" + fifo.attrib["id"] + "']"):
            dest = root.find("./Node[@id='" + edge.attrib["dst_node_id"] + "']")
            inp = edge.attrib["dst_node_input"]       
            for st in dest.find("OriginStackTrace").text.splitlines():
                m = re.match(r".*\((.*)\.maxj:([0-9]*)\)", st)
                if not m is None and m.group(1) in pxgfile:                
                    print(m.group(1) + ":" + m.group(2), "Destination:", dest.attrib["type"], "Input:", inp)
            if dest.attrib["type"] == "NodeFIFO":
                print("Destination FIFO", "ID:", dest.attrib["id"]) 
        
