usefulSizes = (
    #(4, 4), (8, 8), (16, 16), (24, 24),
    (32, 32), (48, 48),
    (53, 53), (64, 64),
    (76, 76), (80, 80),
    (96, 96), (128, 128), (160, 160), (192, 192), (256, 256),
    (384, 384), (512, 512), (1024, 1024))
    
floatSizes = ((32, 32), (64, 64), (79, 79))

IMPNAME = "CUSTOM33"
BASE, SYNTH, IMP, PLACE, ROUTE = (
    "PermanentTestDFE/PermanentTest_singleDFE_XilinxAlveoU250_DFE/scratch/xilinx_vivado/",
    "synth/com_maxeler_platform_max5_board_XilinxAlveoU250Top_post_synth.dcp",
    "implementation/" + IMPNAME + "/xilinx_vivado/",
    "place/com_maxeler_platform_max5_board_XilinxAlveoU250Top_post_place.dcp",
    "route/com_maxeler_platform_max5_board_XilinxAlveoU250Top_post_route.dcp")

def get_tcl(isSynth=True):    
    import os
    tcl = """
    proc get_stats {a b e s t} {
        upvar 1 $b bel
        upvar 1 $e elements
        upvar 1 $s slices
        upvar 1 $t totnets
        set val [get_property PRIMITIVE_LEVEL $a]
        if {$val == "LEAF"} {
            set val [get_property BEL $a]
            if {$val != ""} {
               if {[dict exists $bel $val]} {
                   dict set bel $val [expr [dict get $bel $val] + 1]
               } else {
                   dict append bel $val 1
               }
            }
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
               if {[dict exists $slices $val]} {
                   dict set slices $val [expr [dict get $slices $val] + 1]
               } else {
                   dict append slices $val 1
               }
            }
            #set val [get_property PRIMITIVE_COUNT $a]
            #set val [get_property PRIMITIVE_GROUP $a]
            #set val [get_property PRIMITIVE_SUBGROUP $a]
            #set val [get_property PRIMITIVE_TYPE $a]
        } else {
            set childs [get_cells -regexp [concat [get_property NAME $a]/*]]
            foreach child $childs {
               if {[get_property NAME $child] != [get_property NAME $a]} { 
                   get_stats $child bel elements slices totnets
               }
            }
        }
        set totnets [expr $totnets + [llength [get_nets -of_objects $a]]]
        return [list $bel $elements $slices $totnets]
    }
    proc dump_state {name dir} {
        set bel [dict create]
        set elements [dict create]
        set slices [dict create]
        set totnets 0
        get_stats [get_cells -hier [concat ${name}_core]] bel elements slices totnets
        set fp [open [concat ${dir}/results.txt] w]
        puts $fp $bel
        puts $fp $elements
        puts $fp $slices
        puts $fp [concat SLICES [dict size $slices]]
        puts $fp [concat NETS $totnets]
        close $fp
    }
    open_checkpoint """ + BASE + (SYNTH if isSynth else (IMP + ROUTE)) + """
    dump_state PermanentTestKernel """ + os.getcwd() + """
    """
    return tcl
def flatten_unzip(z, interleave=2): return list(zip(*zip(*([iter(z)] * interleave))))
def dump_stats():
    import os
    with open("dump.tcl", "w") as f:
        f.write(get_tcl(True))
    result = os.system("vivado -mode batch -source dump.tcl -notrace -nolog -quiet")
    if result != 0: return
    with open("results.txt", "r") as f:
        lines = f.readlines()
    synthstats = [{x: int(y) for x, y in zip(*flatten_unzip(line.split()))} for line in lines]
    #print(synthstats)
    with open("dump.tcl", "w") as f:
        f.write(get_tcl(False))
    result = os.system("vivado -mode batch -source dump.tcl -notrace -nolog -quiet")
    os.remove("dump.tcl")
    if result != 0: return
    with open("results.txt", "r") as f:
        lines = f.readlines()
    routestats = [{x: int(y) for x, y in zip(*flatten_unzip(line.split()))} for line in lines]
    #print(routestats)
    return synthstats, routestats
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
def runtests():
    import os
    for size in floatSizes:
        for strategy in [1]:#range(2):
            retval = runbuild(True, 100, size, True, strategy, True, False, 0)
            if retval != 0: return
            retval = os.system("make CPUTEST")
            if retval != 0: return
            retval = os.system("make CPUSIMTEST")
            if retval != 0: return
    for signed in (False, True):
        for size in range(2 if signed else 1, 256+1): #usefulSizes
            for strategy in range(3):
                if strategy == 2 and (size[0] > 64 or size[1] > 64): continue
                retval = runbuild(True, 100, size, signed, strategy, False, False, 2)
                if retval != 0: return
                retval = os.system("make CPUTEST")
                if retval != 0: return
                retval = os.system("make CPUSIMTEST")
                if retval != 0: return
def rundfebuilds():
    import os
    for signed in (False, True):
        for size in usefulSizes:
            for strategy in range(3):            
                frequency, freqinc, minfreq = 350, 50, None
                while frequency != minfreq and frequency >= 6.25 and frequency <= 725.0:
                    retval = runbuild(False, frequency, size, signed, strategy, False, False, 2)
                    if retval == 0:
                        minfreq = frequency
                        if minfreq < 350: freqinc = 10
                        frequency += freqinc
                        dump_stats()
                    else: #determine if failed for bad timing score or other reason
                        if not minfreq is None: freqinc = 10
                        frequency -= freqinc
runtests()
#rundfebuilds()
