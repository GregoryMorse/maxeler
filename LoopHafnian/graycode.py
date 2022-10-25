#Gregory Morse, ELTE and QNL, 2022
import sys, os
MANT_DIG = sys.float_info.mant_dig
import numpy as np
from scipy.stats import unitary_group
import groq.api as g
import groq.api.instruction as inst
import groq.api.nn as nn
import groq.tensor as tensor
from groq.common import print_utils
try:
    import groq.runtime.driver as runtime
except ImportError:
    # raise ModuleNotFoundError("groq.runtime")
    print('Error: ModuleNotFoundError("groq.runtime")')
    
def extract_int8(var):
    return g.concat_vectors([x.reinterpret(g.int8).split_vectors([1]*4)[0] for x in var], (len(var), var[0].shape[1]))
def extract_uint8(var):
    return var.reinterpret(g.uint8).split_vectors([1]*4)[0]
WEST, EAST = 0, 1
def get_slice8(drctn, start, end, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S8(" + str(start) + "-" + str(end) + "), B1(" + str(bank) + ")"
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
def alu_for_hemi(alu, drctn): return alu if drctn==WEST else 15-alu
def sg4_for_hemi(sg4, drctn): return sg4 if drctn==WEST else (9-sg4) % 8 #[(9-x)%8 for x in range(8)] vs list(range(8))

def compile_unit_test(name):
    print_utils.infoc("\nCompiling model ...")
    # Compile program to generate IOP. Also generate groqview JSON dump file and
    # check for potential stream conflicts.
    iop_file = g.compile(
        base_name=name, gen_vis_data=True, check_stream_conflicts=False, #tree_conflicts=True, inspect_raw=True
    )
    json_file = g.write_visualizer_data(name)
    print_utils.cprint("Have a GroqView:\n    % " + print_utils.Colors.GREEN + "groqview --port 8888 " + json_file + print_utils.Colors.RESET, "")
    g.check_stream_conflicts(json_file)
    return iop_file, json_file
def invoke(device, iop, pgm_num, ep_num, tensors):
    """Low level interface to the device driver to access multiple programs. A higher level abstraction
    will be provided in a future release that offers access to multiple programs and entry points."""

    pgm = iop[pgm_num]
    ep = pgm.entry_points[ep_num]
    input_buffer = runtime.BufferArray(ep.input, 1)[0]
    output_buffer = runtime.BufferArray(ep.output, 1)[0]
    if ep.input.tensors:
        for input_tensor in ep.input.tensors:
            if input_tensor.name not in tensors:
                raise ValueError(f"Missing input tensor named {input_tensor.name}")
            input_tensor.from_host(tensors[input_tensor.name], input_buffer)
    device.invoke(input_buffer, output_buffer)
    outs = {}
    if ep.output.tensors:
        for output_tensor in ep.output.tensors:
            result_tensor = output_tensor.allocate_numpy_array()
            output_tensor.to_host(output_buffer, result_tensor)
            outs[output_tensor.name] = result_tensor
    return outs

class AdvanceGrayCode(g.Component):        
    def __init__(self, chunks, dim, lastagc=None, **kwargs):
        super().__init__(**kwargs)
        self.chunks, self.dim = chunks, dim
        self.gcodemaskarr = np.array([([0]*4*2+[1<<(i//2%32) if i <32*2 else 0 for i in range(0, (dim//2-1-3)*2)])*4,
                                      ([0]*4*2+[1<<(i//2%32) if i>=32*2 else 0 for i in range(0, (dim//2-1-3)*2)])*4], dtype=np.uint32)
        self.allones, self.allzeros, self.shiftleft, self.gcodemasks, self.negtwo = [], [], [], [], []
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P"
            if lastagc is None or lastagc == True:
                self.allones.append(g.ones(shape=(1, dim*2*2), dtype=g.uint32, name="allones" + dirstr, layout=get_slice4(drctn, 8, 11, plane)))
                self.allzeros.append(g.zeros(shape=(1, dim*2*2), dtype=g.uint32, name="allzeros" + dirstr, layout=get_slice4(drctn, 8, 11, plane)))
                self.shiftleft.append(g.from_data(np.array([[32-1]*dim*2*2], dtype=np.uint32), layout=get_slice4(drctn, 12, 15, drctn), name="shiftleft" + dirstr))
                self.gcodemasks.append(g.from_data(self.gcodemaskarr,
                                                layout=get_slice8(drctn, 8, 15, plane), name="gcodemask" + dirstr))
                self.negtwo.append(g.from_data(np.array([[-2]*dim*2*2], dtype=np.int32), layout=get_slice4(drctn, 12, 15, drctn), name="negtwo" + dirstr))
            else:
                self.allones.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.allones[drctn*2+plane], name="postallones" + dirstr))
                self.allzeros.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.allzeros[drctn*2+plane], name="postallzeros" + dirstr))
                self.shiftleft.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.shiftleft[drctn*2+plane], name="postshiftleft" + dirstr))
                self.gcodemasks.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.gcodemasks[drctn*2+plane], name="postgcodemask" + dirstr))
                self.negtwo.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.negtwo[drctn*2+plane], name="postnegtwo" + dirstr))
        g.add_mem_constraints(self.allones + self.allzeros + self.gcodemasks, self.allones + self.allzeros + self.gcodemasks, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        g.add_mem_constraints(self.shiftleft + self.gcodemasks + self.negtwo, self.shiftleft + self.gcodemasks + self.negtwo, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)               
        self.counterreqs, self.gcodereqs, self.gcodechangereqs = [], [], []
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            if not lastagc is None:
                self.gcodechangereqs.append(tensor.create_storage_request(layout=get_slice8(drctn, 0, 7, plane)))
            if lastagc is None or lastagc == True:
                self.counterreqs.append(tensor.create_storage_request(layout=get_slice8(drctn, 0, 7, plane)))
                self.gcodereqs.append(tensor.create_storage_request(layout=get_slice8(drctn, 0, 7, plane)))
            else:
                self.counterreqs.append(lastagc.counterreqs[drctn*2+plane])
                self.gcodereqs.append(lastagc.gcodereqs[drctn*2+plane])
    def build(self, tcounter, tgcode, inittime=0):
        counters, gcodes, gcodechanges, negmulvecs = [], [], [], []
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)): 
            dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P"
            t = inittime + plane*max(self.dim//16, 10)
            counter = g.split_vectors(tcounter[drctn*2+plane], [1]*2)
            lastgcode = g.split_vectors(tgcode[(1-drctn)*2+plane], [1]*2)
            with g.ResourceScope(name="inccounter" + dirstr, is_buffered=True, time=t, predecessors=None) as pred:
                x = g.add(self.allones[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(2, drctn)]), counter[0].read(streams=g.SG4[sg4_for_hemi(4, drctn)], time=0), alus=[alu_for_hemi(4, drctn)], output_streams=g.SG4[sg4_for_hemi(4, drctn)])
                #y = g.mask_bar(x, g.split_inner_splits(ones)[1], alus=[alu_for_hemi(1, drctn)], output_streams=g.SG4[sg4_for_hemi(3, drctn)]) #mask only checks the low 8 bits are zero/not zero...
                y = g.equal(x, self.allzeros[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(2, drctn)]), alus=[alu_for_hemi(5, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)]).reinterpret(g.uint32)
                y = g.add(y, counter[1].read(streams=g.SG4[sg4_for_hemi(3, drctn)]), alus=[alu_for_hemi(2, drctn)], output_streams=g.SG4[sg4_for_hemi(3, drctn)])
                x = g.vxm_identity(x, alus=[alu_for_hemi(9, drctn)], output_streams=g.SG4[sg4_for_hemi(4, drctn)])
                counters.append(g.concat_vectors([x, y], (2, self.dim*2*2)).write(name="counter" + dirstr, storage_req=self.counterreqs[drctn*2+plane]))
            counter = g.split_vectors(counters[drctn*2+plane], [1]*2)
            with g.ResourceScope(name="countertogcode" + dirstr, is_buffered=True, time=None, predecessors=[pred]) as pred:
                ones = self.allones[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(3, drctn)])        
                low = counter[0].read(streams=g.SG4[sg4_for_hemi(2, drctn)])    
                high = counter[1].read(streams=g.SG4[sg4_for_hemi(0, drctn)])               
                shl = self.shiftleft[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(6, drctn)])
                y = g.right_shift(high, ones, alus=[alu_for_hemi(0, drctn)], output_streams=g.SG4[sg4_for_hemi(1, drctn)])                                
                x = g.bitwise_or(g.left_shift(high, shl, alus=[alu_for_hemi(12, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)]),
                    g.right_shift(low, ones, alus=[alu_for_hemi(4, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)]),
                    alus=[alu_for_hemi(1, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                x = g.bitwise_xor(x, counter[0].read(streams=g.SG4[sg4_for_hemi(3, drctn)]), alus=[alu_for_hemi(2, drctn)], output_streams=g.SG4[sg4_for_hemi(3, drctn)])
                y = g.bitwise_xor(y, counter[1].read(streams=g.SG4[sg4_for_hemi(7, drctn)]), alus=[alu_for_hemi(13, drctn)], output_streams=g.SG4[sg4_for_hemi(7, drctn)])
                gcodes.append(g.concat_vectors([x, y], (2, self.dim*2*2)).write(name="gcode" + dirstr, storage_req=self.gcodereqs[(1-drctn)*2+plane]))
                x = g.bitwise_xor(x, lastgcode[0].read(streams=g.SG4[sg4_for_hemi(3, drctn)]), alus=[alu_for_hemi(3, drctn)], output_streams=g.SG4[sg4_for_hemi(3, drctn)])
                y = g.bitwise_xor(y, lastgcode[1].read(streams=g.SG4[sg4_for_hemi(7, drctn)]), alus=[alu_for_hemi(14, drctn)], output_streams=g.SG4[sg4_for_hemi(7, drctn)])
                gcodechanges.append(g.concat_vectors([x, y], (2, self.dim*2*2)).write(name="gcodechange" + dirstr, storage_req=self.gcodechangereqs[drctn*2+plane]))
            gcode = g.split_vectors(gcodechanges[drctn*2+plane], [1]*2)
            with g.ResourceScope(name="gcodeadapter" + dirstr, is_buffered=True, time=None, predecessors=[pred]) as pred:
                gcodemask = g.split_vectors(self.gcodemasks[drctn*2+plane], [1]*2)
                x = g.bitwise_and(gcode[0].read(streams=g.SG4[sg4_for_hemi(0, drctn)]),
                    gcodemask[0].read(streams=g.SG4[sg4_for_hemi(1, drctn)]),
                    alus=[alu_for_hemi(0, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)])
                y = g.bitwise_and(gcode[1].read(streams=g.SG4[sg4_for_hemi(2, drctn)]),
                    gcodemask[1].read(streams=g.SG4[sg4_for_hemi(3, drctn)]),
                    alus=[alu_for_hemi(4, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                x = g.bitwise_or(x, y, alus=[alu_for_hemi(1, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)])
                x = g.not_equal(x, self.allzeros[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(2, drctn)]), alus=[alu_for_hemi(2, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)]).reinterpret(g.int32)
                x = g.neg(x, alus=[alu_for_hemi(3, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                x = g.bitwise_or(x, self.allones[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(3, drctn)]).reinterpret(g.int32), alus=[alu_for_hemi(6, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                #x = g.mask(x, self.negtwo[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(2, drctn)]), alus=[alu_for_hemi(2, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                #x = g.add(self.allones[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(3, drctn)]).reinterpret(g.int32), x, alus=[alu_for_hemi(3, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                negmulvecs.append(extract_int8([x]).write(name="negmulvec" + dirstr, layout=get_slice1(drctn, 0, plane)))
        g.add_mem_constraints(gcodechanges, tcounter + tgcode, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        g.add_mem_constraints(counters + gcodes + gcodechanges + negmulvecs, counters + gcodes + gcodechanges + negmulvecs, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        return counters, gcodes, negmulvecs
    def chain_test(chunks, dim): #verify performance
        import timeit
        origcounter = [np.zeros((2, dim*2*2), dtype=np.uint32)] * 4
        origgcode = [np.array((x[0,:] ^ ((x[1,:] << 31) | (x[0,:] >> 1)), x[1,:] ^ (x[1,:] >> 1))) for x in origcounter]
        pgm_pkg = g.ProgramPackage(name="agcpackage", output_dir=None)
        with pgm_pkg.create_program_context("init_agc") as pcinit:
            agc = AdvanceGrayCode(chunks, dim)
            tcounter, tgcode = [], []
            for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
                dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P"
                tcounter.append(g.from_data(origcounter[drctn*2+plane], name="initcounter" + dirstr, storage_req=agc.counterreqs[drctn*2+plane]))
                tgcode.append(g.from_data(origgcode[drctn*2+plane], name="initgcode" + dirstr, storage_req=agc.gcodereqs[drctn*2+plane]))
            g.add_mem_constraints(tcounter + tgcode, tcounter + tgcode, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)            
            parallel = len(tcounter)
        with pgm_pkg.create_program_context("agc") as pc:
            agc = AdvanceGrayCode(chunks, dim, agc)
            counters_mt, gcodes_mt, negmulvecs = agc.build(tcounter, tgcode)
            g.add_mem_constraints(counters_mt + gcodes_mt + negmulvecs, tcounter + tgcode, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            for x in counters_mt: x.set_program_output()
            for x in gcodes_mt: x.set_program_output()
            for x in negmulvecs: x.set_program_output()
        print_utils.infoc("\nAssembling model ...")
        iops = pgm_pkg.assemble()
        iop = runtime.IOProgram(iops[0])
        device = runtime.devices[0]
        print_utils.infoc("\nRunning on HW ...")    
        def runnerinit():
            device.open()
            device.load(iop[0], unsafe_keep_entry_points=True)
            device.load(iop[1], unsafe_keep_entry_points=True)
            invoke(device, iop, 0, 0, None)
        tloaddata = timeit.timeit(runnerinit, number=1)/1
        runner = lambda: invoke(device, iop, 1, 0, None)        
        oracleres, result = [], []
        def oracle():
            oracleres.clear(); oracleres.append([])
            for i in range(parallel):
                x = np.vstack((origcounter[i][0,:] + 1, origcounter[i][1,:] + np.where(origcounter[i][0,:]==0xFFFFFFFF, np.ones((1,), dtype=np.uint32), np.zeros((1,), dtype=np.uint32))))
                newgcode = np.vstack((x[0,:] ^ ((x[1,:] << 31) | (x[0,:] >> 1)), x[1,:] ^ (x[1,:] >> 1)))
                changegcodemask = (origgcode[i] ^ newgcode) & agc.gcodemaskarr
                changegcode = np.where((changegcodemask[0,:] | changegcodemask[1,:]) != 0, np.ones((1,), dtype=np.int8) * -1, np.ones((1,), dtype=np.int8)).reshape(1, dim*2*2)   
                oracleres[0].append((x, newgcode, changegcode))
                origcounter[i], origgcode[i] = x, newgcode                
        def actual():
            res = runner()
            result.clear(); result.append([])
            for i in range(parallel):
                result[0].append((res[counters_mt[i].name], res[gcodes_mt[i].name], res[negmulvecs[i].name]))        
        np.set_printoptions(formatter={'int':hex}, threshold=sys.maxsize, floatmode='unique')
        for i in range(1024): #first verify accuracy
            oracle()
            actual()
            oracleres, result = oracleres[0], result[0]
            if not all(all(np.all(oracleres[i][j] == result[i][j]) for i in range(parallel)) for j in range(3)):
                print_utils.err("\nAdvance Gray Code Chain Test Failure at " + str(i))
                print_utils.infoc(str([[(oracleres[i][j][oracleres[i][j] != result[i][j]], result[i][j][oracleres[i][j] != result[i][j]]) for i in range(parallel)] for j in range(3)]))
                break
        else:
            print_utils.success("\nAdvance Gray Code Chain Test Success ...")
                        
        #now measure performance
        tactual = timeit.timeit(actual, number=1)/1
        toracle = timeit.timeit(oracle, number=1000)/1000
        tactualbatch = timeit.timeit(actual, number=10000)/10000
        print("File Sizes", iops[0], os.path.getsize(iops[0]))
        print("CPU Time", toracle, "Groq Load Time", tloaddata, "Groq Time", tactual, "Groq Time Avg. 100000 Batch", tactualbatch)
    
    def unit_test(chunks, dim): #verify accuracy
        with g.ProgramContext() as pc:
            agc = AdvanceGrayCode(chunks, dim, True)
            tcounter, tgcode = [], []
            for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
                dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P"
                tcounter.append(g.input_tensor(shape=(2, dim*2*2), dtype=g.uint32, name="initcounter" + dirstr, layout=get_slice8(drctn, 0, 7, plane)))
                tgcode.append(g.input_tensor(shape=(2, dim*2*2), dtype=g.uint32, name="initgcode" + dirstr, layout=get_slice8(drctn, 0, 7, plane)))
            g.add_mem_constraints(tcounter + tgcode, tcounter + tgcode, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            parallel = len(tcounter)
            counters_mt, gcodes_mt, negmulvecs = agc.build(tcounter, tgcode)
            g.add_mem_constraints(counters_mt + gcodes_mt + negmulvecs, tcounter + tgcode, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            for x in counters_mt: x.set_program_output()
            for x in gcodes_mt: x.set_program_output()
            for x in negmulvecs: x.set_program_output()
            iop_file, json_file = compile_unit_test("advgraycode")
        runner = g.create_tsp_runner(iop_file)
        origcounter = [np.random.randint(0, 1<<32, (2, dim*2*2), dtype=np.uint32)] * parallel
        origgcode = [np.array((x[0,:] ^ ((x[1,:] << 31) | (x[0,:] >> 1)), x[1,:] ^ (x[1,:] >> 1))) for x in origcounter]
        oracleres, result = [], []
        def oracle():
            oracleres.append([])
            for i in range(parallel):
                x = np.vstack((origcounter[i][0,:] + 1, origcounter[i][1,:] + np.where(origcounter[i][0,:]==0xFFFFFFFF, np.ones((1,), dtype=np.uint32), np.zeros((1,), dtype=np.uint32))))
                newgcode = np.vstack((x[0,:] ^ ((x[1,:] << 31) | (x[0,:] >> 1)), x[1,:] ^ (x[1,:] >> 1)))
                changegcodemask = (origgcode[i] ^ newgcode) & agc.gcodemaskarr
                changegcode = np.where((changegcodemask[0,:] | changegcodemask[1,:]) != 0, np.ones((1,), dtype=np.int8) * -1, np.ones((1,), dtype=np.int8)).reshape(1, dim*2*2)
                oracleres[0].append((x, newgcode, changegcode))
        def actual():
            inputs = {}
            for i in range(parallel):
                inputs[tcounter[i].name] = origcounter[i]
                inputs[tgcode[i].name] = origgcode[i]
            res = runner(**inputs)
            result.append([])
            for i in range(parallel):
                result[0].append((res[counters_mt[i].name], res[gcodes_mt[i].name], res[negmulvecs[i].name]))
        print_utils.infoc("\nRunning on HW ...")
        oracle()
        actual()
        oracleres, result = oracleres[0], result[0]
        np.set_printoptions(formatter={'int':hex}, threshold=sys.maxsize, floatmode='unique')
        if all(all(np.all(oracleres[i][j] == result[i][j]) for i in range(parallel)) for j in range(3)):
            print_utils.success("\nAdvance Gray Code Unit Test Success ...")
        else:
            print_utils.err("\nAdvance Gray Code Unit Test Failure")
            print_utils.infoc(str([[(oracleres[i][j][oracleres[i][j] != result[i][j]], result[i][j][oracleres[i][j] != result[i][j]]) for i in range(parallel)] for j in range(3)]))
                
def main():
    dim = 80 #dim X dim complex matrix
    bitsize = 64 #for fixed point representation will round up to nearest multiple of 7
    chunks = (bitsize + 7-1)//7 #ceiling division to be exact
    AdvanceGrayCode.unit_test(chunks, dim)
    AdvanceGrayCode.chain_test(chunks, dim)
                
if __name__ == "__main__":
    main()
