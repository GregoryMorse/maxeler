usefulSizes = (
    #(4, 4), (8, 8), (16, 16), (24, 24),
    (32, 32), (48, 48),
    (53, 53), (64, 64),
    (76, 76), (80, 80),
    (96, 96), (128, 128), (160, 160), (192, 192), (256, 256),
    (384, 384), (512, 512), (1024, 1024))
    
floatSizes = ((32, 32), (64, 64), (79, 79))

IMPNAME = "CUSTOM3"
BASE, SYNTH, IMP, PLACE, ROUTE = (
    "PermanentTestDFE/PermanentTest_singleDFE_XilinxAlveoU250_DFE/scratch/xilinx_vivado/",
    "synth/com_maxeler_platform_max5_board_XilinxAlveoU250Top_post_synth.dcp",
    "implementation/" + IMPNAME + "/xilinx_vivado/",
    "place/com_maxeler_platform_max5_board_XilinxAlveoU250Top_post_place.dcp",
    "route/com_maxeler_platform_max5_board_XilinxAlveoU250Top_post_route.dcp")

def get_tcl(isSynth=True):    
    import os
    tcl = """
    proc get_stats {a e s t f d} {
        upvar 1 $e elements
        upvar 1 $s slices
        upvar 1 $t totnets
        upvar 1 $f regfence
        upvar 1 $d done
        dict append done $a 0
        #set val [get_property PRIMITIVE_LEVEL $a]
        #set val [get_property PRIMITIVE_COUNT $a]
        #set val [get_property PRIMITIVE_GROUP $a]
        #set val [get_property PRIMITIVE_SUBGROUP $a]
        #set val [get_property PRIMITIVE_TYPE $a]
        set outnets [get_pins -of_objects $a -filter {DIRECTION==OUT}]
        if {[get_property PRIMITIVE_GROUP $a] != "REGISTER"} {
            set val [get_property REF_NAME $a]
            if {$val != ""} {
               if {[dict exists $elements $val]} {
                   dict set elements $val [expr [dict get $elements $val] + 1]
               } else {
                   dict append elements $val 1
               }
            }
            set val [get_property LOC $a]
            get_property REF_NAME $a
            if {$val != ""} {
               if {![dict exists $slices $val]} {
                   dict append slices $val {}
               }
               set bel [get_property BEL $a]
               if {$bel != ""} {
                   dict lappend slices $val $bel
                   #dict set slices $val [lappend [dict get $slices $val] $bel]
               }
            }
            lappend totnets [llength $outnets]
        }
        set childs [get_cells -of_objects [get_pins -of_objects [get_nets -segments -of_objects $outnets] -filter DIRECTION==IN]]
        foreach child $childs {
            if {[dict exists $done $child]} { continue }
            if {[get_property PRIMITIVE_GROUP $child] == "REGISTER"} {
                lappend regfence $child
            } else {
                get_stats $child elements slices totnets regfence done
            }
        }
    }
    proc dump_state {name dir} {
        set elements [dict create]
        set slices [dict create]
        set regfence {}
        set totnets {}
        set namedict [dict create]
        set filter [dict create]
        foreach child [get_cells -hier -regexp {.*node_id\\d+_nodeinput_inp\\d+inp\\d+_data0_reg/reg_reg.*}] { get_stats $child elements slices totnets regfence filter }
        set delay """ + ("0" if isSynth else "[get_property DATAPATH_DELAY [get_timing_paths -from [get_pins -of_objects [get_cells -hier -regexp {.*node_id\\d+_nodeinput_inp\\d+inp\\d+_data0_reg/reg_reg.*}] -filter {REF_PIN_NAME==Q}] -to [get_pins -of_objects $regfence -filter {REF_PIN_NAME==D}]]]") + """
        set p [report_power -hier power -return_string -hierarchical_depth 8]
        regexp {PermanentTestKernel_core\\s+\\|\\s+(\\d+\\.\\d+) \\|} $p "" power
        set fp [open [concat ${dir}/results.txt] w]
        puts $fp $slices
        puts $fp $elements
        puts $fp [concat SLICES [dict size $slices]]
        puts $fp [concat NETS [expr [join $totnets +]+0]]
        puts $fp [concat DELAY $delay]
        puts $fp [concat POWER $power]
        close $fp
    }
    open_checkpoint """ + BASE + (SYNTH if isSynth else (IMP + ROUTE)) + """
    dump_state PermanentTestKernel """ + os.getcwd() + """
    """
    return tcl
def flatten_unzip(z, interleave=2): return list(zip(*zip(*([iter(z)] * interleave))))
def get_stats(isSynth):
    import os
    with open("dump.tcl", "w") as f:
        f.write(get_tcl(False))
    result = os.system("vivado -mode batch -source dump.tcl -notrace -nolog -quiet")
    os.remove("dump.tcl")
    if result != 0: return
    with open("results.txt", "r") as f:
        lines = f.readlines()
    slices, key = {}, None
    for x in lines[0].split():
        if x.startswith("{"): slices[key].append(x[1:]) 
        elif x.endswith("}"): slices[key].append(x[:-1]); key = None
        elif key is None: key = x; slices[key] = []
        else:
            slices[key].append(x)
            if len(slices[key]) == 1: key = None
    stats = [slices] + [{x: float(y) if x == "DELAY" or x == "POWER" else int(y) for x, y in zip(*flatten_unzip(line.split()))} for line in lines[1:]]
    print(stats)
    return stats
def dump_stats():
    #synthstats = get_stats(True)
    routestats = get_stats(False)
    return {'Logical LUTs': sum(routestats[1][x] for x in routestats[1] if x.startswith('LUT')),
        'LUTs': sum(1 for x in routestats[0] for y in routestats[0][x] if "6LUT" in y or "5LUT" in y and not y.replace("5LUT", "6LUT") in routestats[0][x]),
        'LUTNMs': sum(1 for x in routestats[0] for y in routestats[0][x] if "5LUT" in y and y.replace("5LUT", "6LUT") in routestats[0][x]),
        'MUXF7s': sum(routestats[1][x] for x in routestats[1] if x.startswith('MUXF7')),
        'MUXF8s': sum(routestats[1][x] for x in routestats[1] if x.startswith('MUXF8')),
        'MUXF9s': sum(routestats[1][x] for x in routestats[1] if x.startswith('MUXF9')),
        **routestats[2], **routestats[4], **routestats[5]} #return synthstats, routestats
def runbuild(isSim, frequency, size, signed, strategy, useFloat, isComplex, addSubMul):
    import os
    wd = os.getcwd()
    os.chdir(os.path.join(wd, "PermanentTestDFE"))
    retval = os.system("ant build -Dmanager=permanenttest.PermanentTestManager -Dtarget=" + ("DFE_SIM" if isSim else "DFE") +
        " -Dmaxfile-name=PermanentTest_single" + ("SIM" if isSim else "DFE") + " -Dresult-dir=./builds/" + ("simulation" if isSim else "bitstream") + " -Dengine-params=\"frequency=" + str(frequency) +
        " inpX=" + str(-size[0] if signed else size[0]) + " inpY=" + str(-size[1] if signed else size[1]) +
        " strategy=" + str(strategy) + " useFloat=" + str(useFloat).lower() + " isComplex=" + str(isComplex).lower() + " addSubMul=" + str(addSubMul) + "\"")
    os.chdir(wd)
    return retval
def runtests(curTests):
    import os
    for (frequency, size, signed, strategy, useFloat, isComplex, addSubMul) in curTests:
        retval = runbuild(True, 100, size, signed, strategy, useFloat, isComplex, addSubMul)
        if retval != 0: return
        retval = os.system("make CPUTEST")
        if retval != 0: return
        retval = os.system("make CPUSIMTEST")
        if retval != 0: return
def rundfebuilds(curTests):
    import os
    for (freq, size, signed, strategy, useFloat, isComplex, addSubMul) in curTests:
        frequency, minfreq, maxfreq = freq
        freqinc = 50
        while (minfreq is None or frequency > minfreq) and (maxfreq is None or frequency < maxfreq) and frequency >= 6.25 and frequency <= 650.0: #725.0:
            retval = runbuild(False, frequency, size, signed, strategy, useFloat, isComplex, addSubMul)
            if retval == 0:
                stats = dump_stats()
                with open("buildstats.txt", "a") as f:
                    f.write("Frequency: " + str(frequency) + " Size: " + str(size) + " Signed: " + str(signed) + " strategy: " + str(strategy) + " useFloat: " + str(useFloat) + " isComplex: " + str(isComplex) + " addSubMul: " + str(addSubMul) + "\n")
                    f.write(str(stats) + "\n")
                if not minfreq is None and frequency < minfreq or not maxfreq is None: freqinc = 10
                minfreq = frequency
                frequency += freqinc
            else: #determine if failed for bad timing score or other reason
                maxfreq = frequency
                if not minfreq is None: freqinc = 10
                frequency -= freqinc
lzctests = [((650, None, None), (size, size), False, strategy, False, False, 3) for size in (8, 16, 32, 64) for strategy in (5,)] #[ for size in (2, 4, 6, 8, 16, 24, 24+2, 32, 53, 53+2, 64, 64+2) for strategy in range(3)]
#lzctests = [(100, (size, size), False, strategy, False, False, 3) for size in range(2, 32+1) for strategy in (5,)]
multests = [(100, size, True, strategy, True, False, 2) for size in floatSizes for strategy in range(2)]
addsubtests = [(100, size, signed, strategy, False, False, addSub) for addSub in range(2) for signed in (False, True) for size in usefulSizes for strategy in range(2) if not (strategy == 2 and (size[0] > 64 or size[1] > 64))] #range(2 if signed else 1, 256+1)
#print(dump_stats()); assert False
#runtests(lzctests)
rundfebuilds(lzctests)
