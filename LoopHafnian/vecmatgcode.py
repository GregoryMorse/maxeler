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

def num_to_bits(num, chunks):
    res, shp = np.repeat(num[np.newaxis,...], chunks, axis=0), [chunks] + [1] * len(num.shape)
    bits = ((res >> np.arange(0, 7 * chunks, 7).reshape(shp)) & np.array([((1 << 7)-1)]*(chunks-1)+[-1]).reshape(shp)).astype(np.int8) #(70, -63) bit fixed point integers
    #if len(bits.shape)==3: assert np.allclose(num, bits_to_num(bits.transpose(1, 2, 0), 0)), (num, bits_to_num(bits.transpose(1, 2, 0), 0))
    #else: assert np.allclose(num, bits_to_num(bits.transpose(), 0)), (num, bits_to_num(bits.transpose(), 0))
    #assert ((bits >=0) & (bits <= 127)).all()
    return bits
from numba import jit #pip install -U numba=0.54.1
@jit(nopython=True)
def bits_to_num(num, offset=7):
    #assert ((num[:,-1] == 0) | (num[:,-1] == -1)).all()
    #if not (num[:,0:-1] >= 0).all(): print(num[:,0:-1] >= 0)
    #assert (num[:,0:-1] >= 0).all(), num[:,0:-1] >= 0
    shifts = np.arange(-offset, 7 * num.shape[0]-offset, 7)
    shifts[shifts < 0] = 0
    return np.sum(num.astype(np.int64) << shifts.reshape(shifts.shape[0], 1), axis=0) #the high byte can be 0/-1 or this overflows...
def normalize_doubles(num, dimension, fractionbits=63):
    mantissas, exponents = np.frexp(num)
    maxexp = np.amax(exponents, axis=dimension)
    mant = np.rint(np.ldexp(mantissas, exponents-(maxexp[:,np.newaxis] if dimension==1 else maxexp)+fractionbits)).astype(np.int64) #(64, -63) bit fixed point integers
    #assert np.allclose(num, renormalize_doubles(mant, maxexp+fractionbits)), maxexp #, (num, renormalize_doubles(mant, maxexp+fractionbits))
    return maxexp, mant
@jit(nopython=True)
def renormalize_doubles(num, exp):
    return np.ldexp(num.astype(np.float64), -exp)
@jit(nopython=True)
def vector_complex_to_real(cplx):
    dim = cplx.shape[-1]//2 #len(cplx)//2
    result = cplx[...,dim:] * 1j
    result += cplx[...,:dim]
    return result #return cplx[:dim] + cplx[dim:]*1j
def vector_real_to_complex(vec):
    return np.hstack((vec.real, vec.imag))
def matrix_real_to_complex(mtx):
    return np.vstack((np.hstack((mtx.real, -mtx.imag)), np.hstack((mtx.imag, mtx.real))))
def cond_runfunc(ft, ff, x, cond):
    return ft(x) if cond else ff(x)
def extract_int8(var):
    return g.concat_vectors([x.reinterpret(g.int8).split_vectors([1]*4)[0] for x in var], (len(var), var[0].shape[1]))
def extract_uint8(var):
    return var.reinterpret(g.uint8).split_vectors([1]*4)[0]
    
def flatten_zip(z): return [item for sublist in z for item in sublist]
def flatten_unzip(z, interleave=2): return list(zip(*zip(*([iter(z)] * interleave))))

def perf_pro(f):
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()
    f()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

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
def get_slice2(drctn, slices, bank=0):
    return "-1, H1(" + ("W" if drctn==WEST else "E") + "), S2(" + ",".join(str(x) for x in slices) + "), B1(" + str(bank) + ")"
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
def invoke(devices, iop, pgm_num, ep_num, tensors, lastouts=None, buffers=None):
    """Low level interface to the device driver to access multiple programs. A higher level abstraction
    will be provided in a future release that offers access to multiple programs and entry points."""
    pgm = iop[pgm_num]
    ep = pgm.entry_points[ep_num]
    input_buffers, output_buffers = [], []
    for i, device in enumerate(devices):
        input_buffers.append(runtime.BufferArray(ep.input, 1)[0] if buffers is None else buffers[0][i])
        output_buffers.append(runtime.BufferArray(ep.output, 1)[0] if buffers is None else buffers[1][i])
        if ep.input.tensors:
            for input_tensor in ep.input.tensors:
                if input_tensor.name not in tensors[i]:
                    raise ValueError(f"Missing input tensor named {input_tensor.name}")
                input_tensor.from_host(tensors[i][input_tensor.name], input_buffers[i])
        device.invoke_nonblocking(input_buffers[i], output_buffers[i])
    l = len(devices)
    outs = [{} for _ in range(l)]
    i, checks = -1, list(range(l))
    while l != 0:
        i = (i + 1) % l
        idx = checks[i]
        if not output_buffers[idx].ready(): continue
        del checks[i]; l -= 1
        if ep.output.tensors:
            for output_tensor in ep.output.tensors:
                result_tensor = lastouts[idx][output_tensor.name] if not lastouts is None else output_tensor.allocate_numpy_array()
                output_tensor.to_host(output_buffers[idx], result_tensor)
                outs[idx][output_tensor.name] = result_tensor
    return outs, [input_buffers, output_buffers]
class VecNormalize(g.Component):
    def __init__(self, chunks, dim, lastvn=None, **kwargs):
        super().__init__(**kwargs)
        self.chunks, self.dim = chunks, dim
        self.consts = []
        for drctn in (WEST, EAST):
            dirstr = "W" if drctn == WEST else "E"
            if lastvn is None or lastvn == True or lastvn == False:
                self.consts.append({
                    "signshift": g.from_data(np.array([[32-1]*dim], dtype=np.int32), layout=get_slice4(drctn, 0, 3, drctn), name="signshift" + dirstr),
                    "floatbias": g.from_data(np.array([[127-1]*dim], dtype=np.uint32), layout=get_slice4(drctn, 0, 3, drctn), name="floatbias" + dirstr),
                    "initexp": g.from_data(np.array([[(127+23)<<23]*dim], dtype=np.uint32), layout=get_slice4(drctn, 0, 3, drctn), name="initexp" + dirstr), 
                    "subval": g.from_data(np.array([[1<<23]*dim], dtype=np.float32), layout=get_slice4(drctn, 0, 3, drctn), name="subval" + dirstr), 
                    "shiftexp": g.from_data(np.array([[23]*dim], dtype=np.uint32), layout=get_slice4(drctn, 0, 3, drctn), name="shiftexp" + dirstr),
                    "shiftcomp": g.from_data(np.array([[7]*dim], dtype=np.uint8), layout=get_slice1(drctn, 3, drctn), name="shiftcomp" + dirstr),
                    "shiftmask": g.from_data(np.array([[(1<<7)-1]*dim], dtype=np.int32), layout=get_slice4(drctn, 8, 11, drctn), name="shiftmask" + dirstr),
                    "zeros": g.zeros(shape=(1,dim), dtype=g.int8, layout=get_slice1(drctn, 0, drctn), name="zeros" + dirstr),
                    "maps": g.concat_inner_splits([g.from_data(np.array((list(range(i, 16)) + list(range(0, i)))*20, dtype=np.uint8).reshape(1, 320), layout=get_slice1(drctn, 43, drctn)) for i in range(16)]),
                    "perms": g.concat_inner_splits([g.from_data(np.array(inst.encode_permute_map(list(range(16*i, 160))+list(range(0, 16*i))+list(range(160+16*i, 320))+list(range(160, 160+16*i))), dtype=np.uint8).reshape(1, 320), layout=get_slice1(drctn, 42, drctn)) for i in range(10)]),
                })                
            else: self.consts.append({key: tensor.create_shared_memory_tensor(memory_tensor=lastvn.consts[drctn][key], name="post" + lastvn.consts[drctn][key].name) for key in lastvn.consts[drctn]})
            g.add_mem_constraints(list(self.consts[drctn].values()), list(self.consts[drctn].values()), g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        self.normreqs, self.logtworeqs, self.biasreqs, self.shftreqs, self.shiftedreqs, self.selreqs, self.cmpreqs, self.permreqs = [], [], [], [], [], [], [], []
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            self.normreqs.append(tensor.create_storage_request(layout=get_slice1(drctn, 1, plane)))
            self.biasreqs.append(tensor.create_storage_request(layout=get_slice4(drctn, 0, 3, plane)))
            self.logtworeqs.append(tensor.create_storage_request(layout=get_slice4(drctn, 4, 7, plane)))
            self.shftreqs.append(tensor.create_storage_request(layout=get_slice1(drctn, 0, plane)))
            self.selreqs.append(tensor.create_storage_request(layout=get_slice1(drctn, 2, plane)))            
            self.shiftedreqs.append(tensor.create_storage_request(layout=get_slice1(drctn, 0, plane)))
            self.cmpreqs.append(tensor.create_storage_request(layout=get_slice1(drctn, 1, plane)))
            self.permreqs.append(tensor.create_storage_request(layout=get_slice1(drctn, 43, plane)))
    def build(self, tvec, tnorm, inittime=0):
        for drctn in (WEST, EAST):
            g.add_mem_constraints(list(self.consts[drctn].values()), tnorm, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        absresult, logtworesult, maxresult, maxfinresult, allsels, nextmasks, result, biases, resnorm = [], [], [], [], [], [], [], [], []
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            curvec = g.split_inner_splits(tvec[drctn*2+plane])        
            t = inittime+plane*max(self.dim//16, 10)
            dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P" + "_t" + str(inittime)
            with g.ResourceScope(name="abscomp" + dirstr, is_buffered=True, time=t) as pred:
                x = curvec[-1].read(streams=g.SG4[sg4_for_hemi(0, drctn)], time=0)
                x = g.bitwise_xor(g.vxm_identity(x, alus=[alu_for_hemi(12, drctn)], output_streams=g.SG4[sg4_for_hemi(1, drctn)]),
                    g.right_shift(x, self.consts[drctn]["signshift"].read(streams=g.SG4[sg4_for_hemi(2, drctn)]), alus=[alu_for_hemi(0, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)]),
                    alus=[alu_for_hemi(2, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)]
                    ).reinterpret(g.uint32)
                bias = g.mask(x, self.consts[drctn]["floatbias"].read(streams=g.SG4[sg4_for_hemi(1, drctn)]), alus=[alu_for_hemi(3, drctn)], output_streams=g.SG4[sg4_for_hemi(3, drctn)])
                x = g.vxm_identity(x, alus=[alu_for_hemi(7, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                absresult.append(x.write(name="abscomp" + dirstr, storage_req=self.logtworeqs[drctn*2+plane]))
                biases.append(bias.write(name="bias" + dirstr, storage_req=self.biasreqs[drctn*2+plane]))
            with g.ResourceScope(name="logtwo" + dirstr, is_buffered=True, time=t+28, predecessors=None) as pred:
                x = absresult[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(0, drctn)], time=0)
                x = g.sub(
                        g.right_shift(
                            g.sub(
                                g.bitwise_or(x, self.consts[drctn]["initexp"].read(streams=g.SG4[sg4_for_hemi(1, drctn)]), alus=[alu_for_hemi(0, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)]).reinterpret(g.float32),
                                self.consts[drctn]["subval"].read(streams=g.SG4[sg4_for_hemi(2, drctn)]), alus=[alu_for_hemi(1, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)]).reinterpret(g.uint32),
                            self.consts[drctn]["shiftexp"].read(streams=g.SG4[sg4_for_hemi(2, drctn)]), alus=[alu_for_hemi(2, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)]),
                        biases[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(2, drctn)]), alus=[alu_for_hemi(3, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)])
                logtworesult.append(x.write(name="logtwo" + dirstr, storage_req=self.logtworeqs[drctn*2+plane]))
            with g.ResourceScope(name="maxlogtwo" + dirstr, is_buffered=True, time=t+52, predecessors=None) as pred:
                #reduce_max inner dimension requires  Distributor x 16 -> VXM reduce_max outer -> Shifter -> reduce_max outer
                x = g.concat_inner_splits([logtworesult[drctn*2+plane]]*16).read(streams=g.SG4[4*plane], time=0)
                x = g.distribute_8(x, map=self.consts[drctn]["maps"], distributor_req=drctn*4+2*plane, bypass8=0b11110000, map_stream_req=g.SG1[(4*plane+1)*4])
                x = g.transpose_null(x, transposer_req=drctn*2+plane, stream_order=[0,1,2,3])
                x = g.concat_vectors(g.split_inner_splits(x), (16, 320))
                x = g.reduce_max(x, dims=[0], alus=[alu_for_hemi(8*plane, drctn)], output_streams=g.SG4[4*plane])
                x = extract_uint8(x)
                maxresult.append(x.write(name="maxlogtwo" + dirstr, storage_req=self.permreqs[drctn*2+plane]))
            with g.ResourceScope(name="maxlogtwofin" + dirstr, is_buffered=True, time=t+134, predecessors=None) as pred:
                x = g.concat_inner_splits([maxresult[drctn*2+plane].reshape(1, 320)]*10)#.read(streams=g.SG4[4*plane])
                x = g.permute_inner(x, permute_map=self.consts[drctn]["perms"], permutor_req=drctn, input_streams=[g.SG1[0], g.SG1[24]], output_streams=g.SG1[0], time=0)
                x = g.concat_vectors(g.split_inner_splits(x), (10, 320))
                x = g.reduce_max(x, dims=[0], alus=[alu_for_hemi(0, drctn)], output_streams=g.SG4[0])
                cmpr = g.greater_equal(x, self.consts[drctn]["shiftcomp"].read(streams=g.SG4[sg4_for_hemi(3, drctn)]), alus=[alu_for_hemi(2, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                x = g.vxm_identity(x, alus=[alu_for_hemi(3, drctn)], output_streams=g.SG4[sg4_for_hemi(3, drctn)])
                maxfinresult.append(x.write(name="maxlogtwofin" + dirstr, storage_req=self.shftreqs[drctn*2+plane]))
                compared = cmpr.write(name="maxlogcompare" + dirstr, storage_req=self.cmpreqs[drctn*2+plane])
            with g.ResourceScope(name="firstshiftpre" + dirstr, is_buffered=True, time=t+241, predecessors=None) as pred:
                curvec_st = g.concat_inner_splits(curvec[1:] + [curvec[-1]]).read(streams=g.SG4[sg4_for_hemi(4, drctn)], time=0)
                nextmask = g.bitwise_and(g.concat_inner_splits(g.split_inner_splits(curvec_st)[:-1]),
                    g.concat_inner_splits([self.consts[drctn]["shiftmask"]]*(self.chunks-1)).read(streams=g.SG4[sg4_for_hemi(5, drctn)]),
                    alus=[alu_for_hemi(4, drctn)], output_streams=g.SG4[sg4_for_hemi(5, drctn)])
                topmask = g.right_shift(g.split_inner_splits(curvec_st)[-1],self.consts[drctn]["shiftcomp"].read(streams=g.SG4[sg4_for_hemi(6, drctn)]),
                    alus=[alu_for_hemi(8, drctn)], output_streams=g.SG4[sg4_for_hemi(6, drctn)])                
                norm = g.add(tnorm[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(0, drctn)], time=0), maxfinresult[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(1, drctn)]),
                    alus=[alu_for_hemi(0, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)])
                resnorm.append(norm.write(name="norm"+dirstr, storage_req=self.normreqs[drctn*2+plane]))
                nextmasks.append(g.concat_inner_splits(g.split_vectors(extract_int8(g.split_inner_splits(nextmask) + [topmask]), [1]*self.chunks)).write(name="nextmask"+dirstr, storage_req=self.selreqs[drctn*2+plane]))
            with g.ResourceScope(name="firstshift" + dirstr, is_buffered=True, time=t+280, predecessors=None) as pred:
                cmpr = g.concat_inner_splits([compared]*self.chunks).read(streams=g.SG4[sg4_for_hemi(2, drctn)])
                selected = g.bitwise_or(g.mask(cmpr, nextmasks[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(3, drctn)]), alus=[alu_for_hemi(0, drctn)], output_streams=g.SG4[sg4_for_hemi(3, drctn)]),
                    g.mask_bar(cmpr, g.concat_inner_splits(g.split_vectors(extract_int8(curvec), [1]*self.chunks)).read(streams=g.SG4[sg4_for_hemi(4, drctn)]), alus=[alu_for_hemi(4, drctn)], output_streams=g.SG4[sg4_for_hemi(4, drctn)]),
                    alus=[alu_for_hemi(5, drctn)], output_streams=g.SG4[sg4_for_hemi(3, drctn)])
                remain = g.sub(g.concat_inner_splits([maxfinresult[drctn*2+plane]]*self.chunks).read(streams=g.SG4[sg4_for_hemi(1, drctn)]), g.mask(cmpr, g.concat_inner_splits([self.consts[drctn]["shiftcomp"]]*self.chunks).read(streams=g.SG4[sg4_for_hemi(0, drctn)], time=0), alus=[alu_for_hemi(1, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)]),
                    alus=[alu_for_hemi(2, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                nextshifts = g.right_shift(selected, remain, alus=[alu_for_hemi(3, drctn)], output_streams=g.SG4[sg4_for_hemi(3, drctn)])
                allsels.append(selected.write(name="selected" + dirstr, storage_req=self.shiftedreqs[(1-drctn)*2+plane]))
                remain = g.split_inner_splits(remain)[0].write(name="remain" + dirstr, storage_req=self.cmpreqs[(1-drctn)*2+plane])
                nextshifts = nextshifts.write(name="nextshift" + dirstr, storage_req=self.selreqs[drctn*2+plane])
            with g.ResourceScope(name="lastshift" + dirstr, is_buffered=True, time=t+325, predecessors=None) as pred:
                diff = g.sub(g.concat_inner_splits([self.consts[1-drctn]["shiftcomp"]]*self.chunks).read(streams=g.SG4[sg4_for_hemi(0, drctn)]),
                    g.concat_inner_splits([remain]*self.chunks).read(streams=g.SG4[sg4_for_hemi(1, drctn)], time=0),
                    alus=[alu_for_hemi(3, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)])                
                lshifts = g.left_shift(g.concat_inner_splits(g.split_inner_splits(allsels[drctn*2+plane])[1:]+[self.consts[drctn]["zeros"]]).read(streams=g.SG4[sg4_for_hemi(2, drctn)]), diff, alus=[alu_for_hemi(2, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                maskedshifts = g.bitwise_and(lshifts, g.concat_inner_splits(g.split_vectors(extract_int8([self.consts[drctn]["shiftmask"]]*self.chunks), [1]*self.chunks)).read(streams=g.SG4[sg4_for_hemi(4, drctn)]), alus=[alu_for_hemi(5, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                res = g.bitwise_or(maskedshifts, nextshifts.read(streams=g.SG4[sg4_for_hemi(3, drctn)]), alus=[alu_for_hemi(0, drctn)], output_streams=g.SG4[sg4_for_hemi(0, drctn)])
                result.append(g.concat_vectors(g.split_inner_splits(res), (self.chunks, self.dim)).write(name="finalres" + dirstr, layout=get_slice1(drctn, 43, plane)))
        g.add_mem_constraints(resnorm + maxfinresult + allsels, resnorm + maxfinresult + allsels, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        g.add_mem_constraints(absresult + logtworesult, absresult + logtworesult + tvec, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        for drctn in (WEST, EAST):
            g.add_mem_constraints(result + maxresult, result + maxresult + list(self.consts[drctn].values()), g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints(resnorm + biases, resnorm + biases + list(self.consts[drctn].values()), g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        return result, resnorm
    def unit_test(chunks, dim):
        with g.ProgramContext() as pc:
            vn = VecNormalize(chunks, dim*2*2)
            tvec, tnorm = [], []
            for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
                dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P"
                tvec.append(g.input_tensor(shape=(chunks, dim*2*2), dtype=g.int32, name="inp" + dirstr, layout=get_slice4(drctn, 4, 7, plane)))
                tnorm.append(g.from_data(np.zeros((1, dim*2*2), dtype=np.uint8), name="norm" + dirstr, storage_req=vn.normreqs[drctn*2+plane]))
            g.add_mem_constraints(tvec, tvec, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints(tnorm, tnorm, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            parallel = len(tvec)*2
            result_mt, resnorm_mt = vn.build([g.concat_inner_splits(g.split_vectors(x, [1]*chunks)) for x in tvec], tnorm)
            for x in result_mt: x.set_program_output()
            for x in resnorm_mt: x.set_program_output()
            iop_file, json_file = compile_unit_test("vecnorm")
        runner = g.create_tsp_runner(iop_file)
        originpvec = [np.full((dim,), ((1 << 53)-1)/(1<<53)+((1 << 53)-1)/(1<<53)*1j) for _ in range(parallel)]
       
        np.set_printoptions(threshold=sys.maxsize)
        #log2 for powers of 2: https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
        def abscomp(v): return v ^ (v >> 31) #absolute 1s complement rather than absolute value/2s complement
        def log2(v): v = abscomp(v); r = (v > 0xF) << 2; v >>= r; shift=(v > 0x3) << 1; v >>= shift; r |= shift; r |= (v >> 1); return r+(v!=0)
        def intToFloat(v): import struct; return struct.unpack('<f', struct.pack('<L', v))[0]
        def floatToInt(f): import struct; return struct.unpack('<L', struct.pack('<f', f))[0]
        def float32log2(v): v = abscomp(v); return (floatToInt(intToFloat(((127+23)<<23) | v)-(1<<23)) >> 23) - ((127-1) if (v!=0) else 0)
        assert all([log2(x)==((~x).bit_length() if x < 0 else x.bit_length()) for x in range(-128,127+1)])
        assert all([float32log2(x)==((~x).bit_length() if x < 0 else x.bit_length()) for x in range(-128,127+1)])
        #[((~x).bit_length() if x < 0 else x.bit_length()) for x in range(-128,127+1)]
        results, oracleres = [], []
        def actual():
            inputs, exp_inpvecs = {}, []
            for i in range(parallel//2):
                exp_inpvec0, normals = normalize_doubles(vector_real_to_complex(originpvec[i*2]), 0, 63-i*2)
                inpvec0 = num_to_bits(normals, chunks)
                inpvec0 = np.roll(inpvec0, 1, axis=0)
                exp_inpvec1, normals = normalize_doubles(vector_real_to_complex(originpvec[i*2+1]), 0, 63-i*2-1)
                inpvec1 = num_to_bits(normals, chunks)
                inpvec1 = np.roll(inpvec1, 1, axis=0)
                inputs[tvec[i].name] = np.hstack((inpvec0, inpvec1)).astype(np.int32)
                exp_inpvecs.extend([exp_inpvec0, exp_inpvec1])
            res = runner(**inputs)
            for i in range(parallel//2):
                nums = bits_to_num(res[result_mt[i].name], 7)
                results.append((vector_complex_to_real(
                    renormalize_doubles(nums[:dim*2], 63 - 7 - exp_inpvecs[i*2]).reshape(1, dim*2)),
                    res[resnorm_mt[i].name][0, :dim*2].reshape(dim*2)))
                results.append((vector_complex_to_real(
                    renormalize_doubles(nums[dim*2:], 63 - 7 - exp_inpvecs[i*2]).reshape(1, dim*2)),
                    res[resnorm_mt[i].name][0, dim*2:].reshape(dim*2)))
        def oracle():
            for i in range(parallel):
                oracleres.append((originpvec[i], np.full(dim*2, 7-i, np.uint8)))
        actual()
        oracle()
        assert all(np.all(oracleres[i][j] == results[i][j]) for i in range(parallel) for j in range(2))        
class VecMatMul(g.Component):
    def __init__(self, chunks, dim, lastvmm=None, **kwargs):
        super().__init__(**kwargs)
        self.chunks, self.dim = chunks, dim
        self.consts = []
        self.vn = VecNormalize(chunks, dim, lastvmm if lastvmm is None or lastvmm == True or lastvmm == False else lastvmm.vn)
        for drctn in (WEST, EAST):
            dirstr = "W" if drctn == WEST else "E"
            if lastvmm is None or lastvmm==True or lastvmm==False:
                self.consts.append({
                    "maskqrt": g.from_data(np.array([[(1<<7)-1]*dim], dtype=np.int32), layout=get_slice4(drctn, 8, 11, drctn), name="maskqrt" + dirstr),
                    "maskqrttoppre": g.from_data(np.array([[(1<<7)-1]*dim], dtype=np.int32), layout=get_slice4(drctn, 32, 35, drctn), name="maskqrttoppre" + dirstr),
                    "maskqrttop": g.from_data(np.array([[-1]*dim], dtype=np.int32), layout=get_slice4(drctn, 32, 35, drctn), name="maskqrttop" + dirstr), 
                    "shiftqrtpre": g.from_data(np.array([[7]*dim], dtype=np.int32), layout=get_slice4(drctn, 12, 15, drctn), name="shiftqrtpre" + dirstr), 
                    "shiftqrt": g.from_data(np.array([[7]*dim], dtype=np.int32), layout=get_slice4(drctn, 8, 11, drctn), name="shiftqrt" + dirstr), 
                    "zerospre": g.zeros(shape=(1,dim), dtype=g.int32, layout=get_slice4(drctn, 4, 7, 0), name="zerospre" + dirstr),
                    "zeros": g.zeros(shape=(1,dim), dtype=g.int32, layout=get_slice4(drctn, 4, 7, 1), name="zeros" + dirstr)
                })
            else: self.consts.append({key: tensor.create_shared_memory_tensor(memory_tensor=lastvmm.consts[drctn][key], name="post" + lastvmm.consts[drctn][key].name) for key in lastvmm.consts[drctn]})
            g.add_mem_constraints([self.consts[drctn]["maskqrttop"]], [self.consts[drctn]["maskqrttoppre"]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints([self.consts[drctn]["shiftqrt"]], [self.consts[drctn]["maskqrt"]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints([self.consts[drctn]["zerospre"]], [self.consts[drctn]["zeros"]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        if not lastvmm is None:
            self.maskqrt, self.maskqrttop, self.shiftqrt, self.zeros = [], [], [], []
            for drctn in (WEST, EAST):
                self.maskqrt.append(g.concat_inner_splits(
                    [self.consts[drctn]["maskqrt"]] * self.chunks))
                self.maskqrttop.append(g.concat_inner_splits(
                    [self.consts[drctn]["maskqrttoppre"]] * (self.chunks-1)+
                    [self.consts[drctn]["maskqrttop"]]))
                self.shiftqrt.append(g.concat_inner_splits(
                    [self.consts[drctn]["shiftqrtpre"]] * self.chunks))
                self.shiftqrt.append(g.concat_inner_splits(
                    [self.consts[drctn]["shiftqrt"]] * self.chunks))
                self.zeros.append(self.consts[drctn]["zerospre"])
                self.zeros.append(self.consts[drctn]["zeros"])
            self.maskreqs, self.splitreqs, self.extractreqs = [], [], []
            for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
                self.maskreqs.append([])
                self.splitreqs.append([])
                for i in range(self.chunks+1):            
                    self.maskreqs[-1].append(tensor.create_storage_request(layout=get_slice4(drctn, 4, 7, plane)))
                    self.splitreqs[-1].append(tensor.create_storage_request(layout=get_slice4(drctn, 0, 3, plane)))
                self.extractreqs.append(tensor.create_storage_request(layout=get_slice4(drctn, 4, 7, plane)))
    def build(self, tvec, tmat, tnorm, inittime=0):
        # Instantiate matmul component.   
        # Build matmul component.
    
        final_result = []
        #split_result = []
        #allshifts = [zeros[0], zeros[1]]
        MATMULDELAY = 76 #4 (stream group size minus 1) for S16(25-27,29-37,39-42) to boundary, 9 for SXM crossing, 5 (stream group size) for weight install completion (not depended upon)
        #3 delay for S1(43) to allow weight install, 9 for SXM and 1 MXM weight crossing, #chunks for vector install (none of these are depended on)
        #weight install takes 3 ticks to start to finish on MXM basic
        #13 for MXM basic multiplication, 34 for SXM accumulation, 19 to stream across (6 leaving SXM, 10 stream crossings, 3 entering ALU)
        #4+9+3+13+34+19 - 6 = 76 #-6 because the S4(8-11)
        #mxm_rqs = [tensor.create_mxm_request(planes=[x], num_planes=1) for x in range(4)]
        #g.latch(maskqrt[1-drctn].read(streams=g.SG4[sg4_for_hemi(3, drctn)]), alu=3)
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            if plane == 0:
                split_result = []
                allshifts = [self.zeros[drctn*2], self.zeros[drctn*2+1]]
            #mm = nn.MatMul(time=0, buffer_output=False, planes=mxm_rqs[drctn*2+plane].planes)
            result_mt = [g.concat_inner_splits(x) for x in zip(*(g.split_vectors(x, [self.dim]*self.chunks) for x in g.split_inner_splits(tmat[drctn*2+plane])))]
            #result_mt = g.split_vectors(tmat[drctn*2+plane], [self.dim]*self.chunks)
            rev_last_alu = [alu_for_hemi(4, drctn)]
            rev_alu = [alu_for_hemi(6, drctn)]
            first_alu = [alu_for_hemi(0, drctn)]
            second_alu = [alu_for_hemi(1, drctn)]
            dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P" + "_t" + str(inittime)
            t = inittime+plane*max(self.dim//16, 10)
            for i in range(self.chunks):
                with g.ResourceScope(name="matmul" + dirstr + str(i), is_buffered=True, time=t) as pred: #mm.end_time==20 #for plane 0 returns on SG4_E[sg4_for_hemi(4, drctn)] #for nn.matmul time=plane*21+(20+12+9+1)*i due to SXM DIST
                    #result_mt[i] = mm.build(tvec[drctn*2+plane], result_mt[i])
                    #g.clear_mxm(planes=[plane], time=0)
                    mxm_rq = tensor.create_mxm_request(planes=[drctn*2+plane], num_planes=1)
                    iw = g.install_weights(result_mt[i], planes=mxm_rq, time=0) #.read(streams=g.SG16_W[plane] if drctn == WEST else g.SG16_E[plane]), time=0 if plane==0 else -18)
                    #iw = g.load_weight_buffer(result_mt[i], planes=mxm_rq, time=0)
                    #print(tvec[drctn*2+plane].shape, tvec[drctn*2+plane].physical_shape, result_mt[i].shape, result_mt[i].physical_shape)
                    result_mt[i] = tvec[drctn*2+plane].matmul(iw, planes=mxm_rq, num_planes=1, accum_input=None, time=0)
                    #result_mt[i] = tvec[drctn*2+plane].matmul(result_mt[i], planes=[plane], time=0)
                    split_result.append(g.concat_inner_splits(g.split_vectors(result_mt[i], [1]*self.chunks)))
                    #must be an arithmetic right shift (sign filled), not logical, but with signed types, this occurs
                    if i == 0:
                        nextmasks = g.bitwise_and(split_result[-1], self.maskqrt[drctn].read(streams=g.SG4[sg4_for_hemi(3, drctn)]), alus=rev_last_alu, output_streams=g.SG4[sg4_for_hemi(3, drctn)]).write(name="mask" + dirstr + str(i), storage_req=self.maskreqs[drctn*2+plane][i])
                        split_result[-1] = split_result[-1].write(name="split" + dirstr + str(i), storage_req=self.splitreqs[drctn*2+plane][i])
                    else:
                        masks = g.concat_inner_splits(g.split_inner_splits(nextmasks)[1:] + [self.zeros[drctn*2+plane]]).read(streams=g.SG4[sg4_for_hemi(1, drctn)])
                        shifts = g.right_shift(split_result[-2].read(streams=g.SG4[sg4_for_hemi(2, drctn)]), self.shiftqrt[2*drctn].read(streams=g.SG4[sg4_for_hemi(0, drctn)]), alus=first_alu, output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                        split_result[-1] = g.add(g.add(shifts, masks, alus=second_alu, output_streams=g.SG4[sg4_for_hemi(2, drctn)]), split_result[-1], alus=rev_alu, output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                        if i != self.chunks - 1:
                            nextmasks = g.bitwise_and(self.maskqrt[1-drctn].read(streams=g.SG4[sg4_for_hemi(3, drctn)]), split_result[-1], alus=rev_last_alu, output_streams=g.SG4[sg4_for_hemi(3, drctn)]).write(name="mask" + dirstr + str(i), storage_req=self.maskreqs[drctn*2+plane][i])
                        else:
                            nextshifts = g.right_shift(split_result[-1], self.shiftqrt[2*(1-drctn)+1].read(streams=g.SG4[sg4_for_hemi(3, drctn)]), alus=rev_last_alu, output_streams=g.SG4[sg4_for_hemi(3, drctn)]).write(name="shiftpre" + dirstr, storage_req=self.maskreqs[drctn*2+plane][i])
                        split_result[-1] = split_result[-1].write(name="split" + dirstr + str(i), storage_req=self.splitreqs[drctn*2+plane][i])
                        g.add_mem_constraints(split_result[:-1], [split_result[-1]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                    allshifts.append(nextshifts if i == self.chunks-1 else nextmasks)
                    g.add_mem_constraints(allshifts[:-1], [allshifts[-1]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                    t += max(self.dim//16*2, (27 if i==0 else 30))
            #with complex multiplication at dimension 80 and 64-bit, we surely need chunks-1 of these rounds to converge with numerical safety
            for i in range(self.chunks-2):
                with g.ResourceScope(name="finsma" + dirstr + str(i), is_buffered=True, time=t+MATMULDELAY-18) as pred: #+predecessors=[pred], time=None) as pred: #they are not all fitting in int8 yet but after first iteration, the final computation can occur
                    cursplit = split_result[-1].read(streams=g.SG4[sg4_for_hemi(5, drctn)])
                    shifts = g.concat_inner_splits([self.zeros[drctn*2+plane]] + g.split_inner_splits(nextshifts)[:-1]).read(streams=g.SG4[sg4_for_hemi(3, drctn)])
                    masks = g.bitwise_and(cursplit, self.maskqrttop[drctn].read(streams=g.SG4[sg4_for_hemi(4, drctn)], time=0), alus=rev_last_alu, output_streams=g.SG4[sg4_for_hemi(4, drctn)])
                    split_result.append(g.add(masks, shifts, alus=rev_alu, output_streams=g.SG4[sg4_for_hemi(2, drctn)]))
                    #nextshifts = g.bitwise_and(split_result[-1], self.maskqrt[1-drctn].read(streams=g.SG4[sg4_for_hemi(0, drctn)]), alus=first_alu, output_streams=g.SG4[sg4_for_hemi(3, drctn)]).write(name="fixmask" + dirstr, storage_req=self.maskreqs[drctn*2+plane][self.chunks])
                    nextshifts = g.right_shift(split_result[-1], self.shiftqrt[2*(1-drctn)+1].read(streams=g.SG4[sg4_for_hemi(0, drctn)]), alus=first_alu, output_streams=g.SG4[sg4_for_hemi(3, drctn)]).write(name="fixshift" + str(i) + dirstr, storage_req=self.maskreqs[drctn*2+plane][self.chunks])
                    split_result[-1] = split_result[-1].write(name="finsplit" + dirstr + str(i), storage_req=self.splitreqs[drctn*2+plane][self.chunks])
                    g.add_mem_constraints(split_result[:-1], [split_result[-1]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                    allshifts.append(nextshifts) #allshifts.append(nextmasks)
                    g.add_mem_constraints(allshifts[:-1], [allshifts[-1]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
                    t += 30                    
            #chunks-1 correct 7-bit int8s
            #final adjustment for between 0-7 bit addition extra bits, for 64x64-127x127 this is exactly 7 bits
            #for complex multiplication requires one more bit, which requires a shift right adjustment by 1
            with g.ResourceScope(name="fixsma" + dirstr, is_buffered=True, time=t+MATMULDELAY-12) as pred: #predecessors=[pred], time=None) as pred:
                cursplit = split_result[-1].read(streams=g.SG4[sg4_for_hemi(3, drctn)])
                #masks = g.concat_inner_splits(g.split_inner_splits(nextmasks)[1:] + [self.zeros[drctn*2+plane]]).read(streams=g.SG4[sg4_for_hemi(0, drctn)])
                #shifts = g.right_shift(cursplit, self.shiftqrt[2*drctn].read(streams=g.SG4[sg4_for_hemi(1, drctn)], time=0), alus=first_alu, output_streams=g.SG4[sg4_for_hemi(1, drctn)])
                masks = g.bitwise_and(cursplit, self.maskqrttop[drctn].read(streams=g.SG4[sg4_for_hemi(1, drctn)], time=0), alus=first_alu, output_streams=g.SG4[sg4_for_hemi(1, drctn)])
                shifts = g.concat_inner_splits([self.zeros[drctn*2+plane]] + g.split_inner_splits(nextshifts)[:-1]).read(streams=g.SG4[sg4_for_hemi(0, drctn)])
                #split_result.append(g.add(shifts, masks, alus=second_alu, output_streams=g.SG4[sg4_for_hemi(1, drctn)]).write(name="fixsplit" + dirstr, layout=get_slice4(drctn, 0, 3, plane)))
                final_result.append(g.add(shifts, masks, alus=second_alu, output_streams=g.SG4[sg4_for_hemi(1, drctn)]).write(name="extract" + dirstr, storage_req=self.extractreqs[drctn*2+plane])) #extract_int8(g.split_inner_splits()) get_slice1(drctn, 43, plane)
                #final_result.append(g.add(shifts, masks, alus=second_alu, output_streams=g.SG4[sg4_for_hemi(1, drctn)]).write(name="extract" + dirstr, storage_req=self.splitreqs[drctn*2+plane][self.chunks]))
            #print("Cycle time: ", t+MATMULDELAY+9+10+31+19) #31 through ALU, 19 to write to S43
            g.add_mem_constraints(allshifts, final_result, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        norm_result, norms = self.vn.build(final_result, tnorm, t+MATMULDELAY+9+10+31+19); t += 386+5
        g.add_mem_constraints(norm_result, tvec, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        for drctn in (WEST, EAST):        
            g.add_mem_constraints(list(self.vn.consts[drctn].values()) + final_result, tvec + final_result + list(self.consts[drctn].values()) + tmat, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        return norm_result, norms, t+MATMULDELAY+9+10+31+19
    def unit_test(chunks, dim):
        with g.ProgramContext() as pc:
            vmm = VecMatMul(chunks, dim*2*2, True)
            tvec, tmat, tnorm = [], [], []
            for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
                dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P"
                tvec.append(g.input_tensor(shape=(chunks, dim*2*2), dtype=g.int8, name="A" + dirstr, layout=get_slice1(drctn, 43, plane)))
                tmat.append(g.input_tensor(shape=(chunks*dim*2*2, dim*2*2), dtype=g.int8, name="B" + dirstr, layout=get_slice16(drctn, s16rangeW if drctn == WEST else s16rangeE, plane)))
                tnorm.append(g.from_data(np.zeros((1, dim*2*2), dtype=np.uint8), name="norm" + dirstr, storage_req=vmm.vn.normreqs[drctn*2+plane]))
            g.add_mem_constraints(tvec, tvec, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints(tmat, tmat, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints(tnorm, tnorm, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
                g.add_mem_constraints([vmm.consts[drctn]["maskqrttop"], vmm.consts[drctn]["maskqrttoppre"]], [tmat[drctn*2+plane]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)        

            parallel = len(tvec)*2
            result_mt, resnorm, _ = vmm.build(tvec, tmat, tnorm)
            for x in result_mt: x.set_program_output()
            for x in resnorm: x.set_program_output()
            iop_file, json_file = compile_unit_test("vecmatmul")
        np.set_printoptions(formatter={'int':hex}, threshold=sys.maxsize, floatmode='unique')
        runner = g.create_tsp_runner(iop_file)
        print_utils.infoc("\nRunning on HW ...")
        worstCase, useCplx = False, False
        for _ in range(1):
            if worstCase:
                if useCplx:
                    originpvec = [np.full((dim,), ((1 << 53)-1)/(1<<53)+((1 << 53)-1)/(1<<53)*1j) for _ in range(parallel)]
                    originpmat = [np.full((dim, dim), -((1 << 53)-1)/(1<<53)-((1 << 53)-1)/(1<<53)*1j) for _ in range(parallel)]
                else:    
                    originpvec = [np.full((dim,), ((1 << 53)-1)/(1<<53)) for _ in range(parallel)]
                    originpmat = [np.full((dim, dim), -((1 << 53)-1)/(1<<53)) for _ in range(parallel)]
            elif useCplx:
                originpvec = [np.random.rand(dim)*2-1 + (np.random.rand(dim)*2j-1j) for _ in range(parallel)]                
                originpmat = [unitary_group.rvs(dim) for _ in range(parallel)]
                #originpmat = [np.random.rand(dim, dim)*2-1 + (np.random.rand(dim, dim)*2j-1j) for _ in range(parallel)]
            else:
                originpvec = [np.random.rand(dim)*2-1 for _ in range(parallel)]
                originpmat = [np.random.rand(dim, dim)*2-1 for _ in range(parallel)]
    
            oracleres = [None]
            def oracle():
                B = [originpmat[i].transpose().astype(np.clongdouble) for i in range(parallel)]
                oracleres[0] = []
                for i in range(parallel):
                    w = originpvec[i].astype(np.clongdouble)
                    w = w @ B[i]
                    oracleres[0].append(w.astype(np.cdouble))
            oracle()
            # Create TSP runner and pass input dictionary of "tensor name" : "tensor data".        
            results = [None]
            def actual():
                fractionbits = 62 if useCplx else 63 #allow 8/7 integer bits
                inputs = {}
                exp_inpvecs, exp_inpmats, Z = [], [], np.zeros((chunks, dim*2, dim*2), dtype=np.int8)
                for i in range(parallel//2):
                    exp_inpvec0, normals = normalize_doubles(vector_real_to_complex(originpvec[i*2]), 0, fractionbits)
                    inpvec0 = num_to_bits(normals, chunks)
                    exp_inpmat0, normals = normalize_doubles(matrix_real_to_complex(originpmat[i*2]), None, fractionbits) #dimension 1 will cause chaining issues without shift corrections
                    inpmat0 = num_to_bits(normals, chunks).reshape((chunks, dim*2, dim*2))
                    exp_inpvec1, normals = normalize_doubles(vector_real_to_complex(originpvec[i*2+1]), 0, fractionbits)
                    inpvec1 = num_to_bits(normals, chunks)
                    exp_inpmat1, normals = normalize_doubles(matrix_real_to_complex(originpmat[i*2+1]), None, fractionbits) #dimension 1 will cause chaining issues without shift corrections
                    inpmat1 = num_to_bits(normals, chunks).reshape((chunks, dim*2, dim*2))
                    inputs[tvec[i].name] = np.hstack((inpvec0, inpvec1))
                    inputs[tmat[i].name] = np.concatenate((np.concatenate((inpmat0, Z), axis=2), np.concatenate((Z, inpmat1), axis=2)), axis=1).reshape((chunks*dim*2*2, dim*2*2))
                    exp_inpvecs.extend([exp_inpvec0, exp_inpvec1]); exp_inpmats.extend([exp_inpmat0, exp_inpmat1])
                res = runner(**inputs)
                results[0] = []
                for i in range(parallel//2):
                    result = bits_to_num(res[result_mt[i].name], 7)
                    norm = res[resnorm[i].name].reshape(dim*2*2)
                    #the results come back truncating the lower 7*(chunks-1) bits
                    results[0].append(vector_complex_to_real(renormalize_doubles(result[:dim*2], 2*fractionbits-63 - 7 - exp_inpvecs[i*2] - exp_inpmats[i*2] - norm[:dim*2].astype(np.int32)).reshape(1, dim*2)).reshape(dim))
                    results[0].append(vector_complex_to_real(renormalize_doubles(result[dim*2:], 2*fractionbits-63 - 7 - exp_inpvecs[i*2+1] - exp_inpmats[i*2+1] - norm[dim*2:].astype(np.int32)).reshape(1, dim*2)).reshape(dim))
            runner = g.create_tsp_runner(iop_file)
            actual()
            oracleres, results = oracleres[0], results[0]
            for i in range(parallel):
                print_utils.infoc("\nComparing results with oracle ...")
                max_atol = max(abs(oracleres[i].reshape(-1) - results[i].reshape(-1)))
                if max_atol <= 0.001:
                    print_utils.success(f"Test PASSED with a max tolerance of {max_atol}")
                else:
                    print_utils.err(
                        f"Test FAILED with a max tolerance of {max_atol} (should be <= 0.001)"); assert False, (oracleres[i], results[i], abs(oracleres[i].reshape(-1) - results[i].reshape(-1)))

class AdvanceGrayCode(g.Component):        
    def __init__(self, chunks, dim, lastagc=None, **kwargs):
        super().__init__(**kwargs)
        self.chunks, self.dim = chunks, dim
        self.gcodemaskarr = np.array([([0]*4*2+[1<<(i//2%32) if i <32*2 else 0 for i in range(0, (dim//2-1-3)*2)])*4,
                                      ([0]*4*2+[1<<(i//2%32) if i>=32*2 else 0 for i in range(0, (dim//2-1-3)*2)])*4], dtype=np.uint32)
        self.origcounter = np.zeros((2, dim*2*2), dtype=np.uint32)
        self.origgcode = np.array((self.origcounter[0,:] ^ ((self.origcounter[1,:] << 31) | (self.origcounter[0,:] >> 1)), self.origcounter[1,:] ^ (self.origcounter[1,:] >> 1)))                                      
        self.counterreqs, self.gcodereqs, self.gcodechangereqs = [], [], []
        self.tcounter, self.tgcode = [], []        
        self.allones, self.allzeros, self.shiftleft, self.gcodemasks, self.negtwo = [], [], [], [], []
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P"
            if not lastagc is None:
                self.gcodechangereqs.append(tensor.create_storage_request(layout=get_slice8(drctn, 0, 7, plane)))
            if lastagc is None or lastagc == True or lastagc == False:
                self.counterreqs.append(tensor.create_storage_request(layout=get_slice8(drctn, 0, 7, plane)))
                self.gcodereqs.append(tensor.create_storage_request(layout=get_slice8(drctn, 0, 7, plane)))
                self.allones.append(g.ones(shape=(1, dim*2*2), dtype=g.uint32, name="allones" + dirstr, layout=get_slice4(drctn, 8, 11, plane)))
                self.allzeros.append(g.zeros(shape=(1, dim*2*2), dtype=g.uint32, name="allzeros" + dirstr, layout=get_slice4(drctn, 8, 11, plane)))
                self.shiftleft.append(g.from_data(np.array([[32-1]*dim*2*2], dtype=np.uint32), layout=get_slice4(drctn, 12, 15, drctn), name="shiftleft" + dirstr))
                self.gcodemasks.append(g.from_data(self.gcodemaskarr,
                                                layout=get_slice8(drctn, 8, 15, plane), name="gcodemask" + dirstr))
                self.negtwo.append(g.from_data(np.array([[-2]*dim*2*2], dtype=np.int32), layout=get_slice4(drctn, 12, 15, drctn), name="negtwo" + dirstr))
            else:
                self.counterreqs.append(lastagc.counterreqs[drctn*2+plane])
                self.gcodereqs.append(lastagc.gcodereqs[drctn*2+plane])
                self.allones.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.allones[drctn*2+plane], name="postallones" + dirstr))
                self.allzeros.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.allzeros[drctn*2+plane], name="postallzeros" + dirstr))
                self.shiftleft.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.shiftleft[drctn*2+plane], name="postshiftleft" + dirstr))
                self.gcodemasks.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.gcodemasks[drctn*2+plane], name="postgcodemask" + dirstr))
                self.negtwo.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.negtwo[drctn*2+plane], name="postnegtwo" + dirstr))
            if lastagc == True:
                self.tcounter.append(g.input_tensor(shape=(2, dim*2*2), dtype=g.uint32, name="initcounter" + dirstr, storage_req=self.counterreqs[drctn*2+plane]))
                self.tgcode.append(g.input_tensor(shape=(2, dim*2*2), dtype=g.uint32, name="initgcode" + dirstr, storage_req=self.gcodereqs[drctn*2+plane]))
            elif lastagc is None or lastagc == False:
                self.tcounter.append(g.from_data(self.origcounter, name="initcounter" + dirstr, storage_req=self.counterreqs[drctn*2+plane]))
                self.tgcode.append(g.from_data(self.origgcode, name="initgcode" + dirstr, storage_req=self.gcodereqs[drctn*2+plane]))
            else:
                self.tcounter.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.tcounter[drctn*2+plane], name="postinitcounter" + dirstr))
                self.tgcode.append(tensor.create_shared_memory_tensor(memory_tensor=lastagc.tgcode[drctn*2+plane], name="postgcode" + dirstr))  
        g.add_mem_constraints(self.tcounter + self.tgcode, self.tcounter + self.tgcode, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        g.add_mem_constraints(self.allones + self.allzeros + self.gcodemasks, self.allones + self.allzeros + self.gcodemasks, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        g.add_mem_constraints(self.shiftleft + self.gcodemasks + self.negtwo, self.shiftleft + self.gcodemasks + self.negtwo, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)               
    def build(self, tvec=None, tmat=None, inittime=0):
        counters, gcodes, gcodechanges, negmulvecs = [], [], [], []
        vecres, matres = [], []
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)): 
            dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P"
            t = inittime + plane*max(self.dim//16, 10)
            counter = g.split_vectors(self.tcounter[drctn*2+plane], [1]*2)
            lastgcode = g.split_vectors(self.tgcode[(1-drctn)*2+plane], [1]*2)
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
                x = g.bitwise_or(x, self.allones[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(3, drctn)]).reinterpret(g.int32), alus=[alu_for_hemi(6, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)], time=0)
                #x = g.mask(x, self.negtwo[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(2, drctn)]), alus=[alu_for_hemi(2, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                #x = g.add(self.allones[drctn*2+plane].read(streams=g.SG4[sg4_for_hemi(3, drctn)]).reinterpret(g.int32), x, alus=[alu_for_hemi(3, drctn)], output_streams=g.SG4[sg4_for_hemi(2, drctn)])
                negmulvecs.append(extract_int8([x]).write(name="negmulvec" + dirstr, layout=get_slice1(drctn, 0, plane)))
        if not tvec is None and not tmat is None:
            splittings = []
            for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
                splittings.append(np.vsplit(np.array(g.split_vectors(tmat[drctn*2+plane], [1]*(self.chunks*self.dim*2*2))).reshape(self.chunks*self.dim*2*2//16, 16).transpose(), 4))
            with g.ResourceScope(name="adjvecmat", is_buffered=True, time=None, predecessors=[pred]) as pred:
                d = 1+9+2 #read, alu, write/write delay time
                splts = []
                for drctn in (WEST, EAST):
                    s16g = s16rangeW if drctn == WEST else s16rangeE
                    splts.append([[np.arange(d+s16g[4*i+j]//4*2, self.chunks*self.dim*2*2//16, d+s16g[4*i+j]//4*2) for j in range(4)] for i in range(4)])
                pads = max([max((d+s16g[4*i+j]//4*2-self.chunks*self.dim*2*2//16+splts[drctn][i][j][-1]) % (d+s16g[4*i+j]//4*2) for drctn in (WEST, EAST) for i in range(4)) for j in range(4)])
                mulvecs = [g.concat_inner_splits([negmulvecs[drctn*2+0]] * (self.chunks*self.dim*2*2//2 + pads*4)).read(streams=g.SG4[sg4_for_hemi(3+4*drctn, WEST)], time=0) for drctn in (WEST, EAST)]
                def unpadlast(lst, sz):
                    return [[x if i != len(lst[0])-1 else x[:sz] for i, x in enumerate(lst[0])],
                            [x if i != len(lst[1])-1 else x[:sz] for i, x in enumerate(lst[1])]]
                def padlast(lst1, lst2, sz, pad):
                    return ([x if i != len(lst1)-1 else np.concatenate((x, [x[-1]]*(sz-len(x)))) for i, x in enumerate(lst1)],
                            [x if i != len(lst2)-1 else np.concatenate((x, [x[-1]]*min(sz-len(x), pad-sz+len(lst1[-1])), [lst1[-1][-1]]*(pad-sz+len(lst1[-1])-min(sz-len(x), pad-sz+len(lst1[-1]))))) for i, x in enumerate(lst2)])
                for drctn in (WEST, EAST):
                    rows = []
                    s16g = s16rangeW if drctn == WEST else s16rangeE
                    for i in range(4):
                        mulalu = tensor.create_alu_request([alu_for_hemi([0,5,8,13][i], drctn)])
                        rows.append(np.vstack([np.concatenate(flatten_zip(zip(*padlast(np.hsplit(a.flatten(), splts[drctn][i][j]), np.hsplit(b.flatten(), splts[drctn][i][j]), d+s16g[4*i+j]//4*2, pads)))) for j, (a, b) in enumerate(zip(np.vsplit(splittings[drctn*2+0][i], 4), np.vsplit(splittings[drctn*2+1][i], 4)))]))
                        rows[i] = g.concat_inner_splits(rows[i].flatten().tolist()).read(streams=g.SG4[sg4_for_hemi(2*i+drctn, drctn)])
                        rows[i] = rows[i].mul(mulvecs[i//2 if drctn==WEST else 1-i//2], alus=mulalu, output_streams=(g.SG4_W if drctn==WEST else g.SG4_E)[sg4_for_hemi(2*i+drctn, drctn)])
                        rows[i] = np.hstack([np.hstack(flatten_zip(unpadlast(flatten_unzip(np.hsplit(a.flatten(),
                            np.arange(d+s16g[4*i+j]//4*2, self.chunks*self.dim*2*2//16*2+pads, d+s16g[4*i+j]//4*2)
                            )), self.chunks*self.dim*2*2//16-splts[drctn][i][j][-1]))).reshape(2, 4, self.chunks*self.dim*2*2//16//4) for j, a in enumerate(np.vsplit(np.array(g.split_inner_splits(rows[i])).reshape(4, self.chunks*self.dim*2*2//16*2+pads), 4))])
                    for plane in range(2):
                        matres.append(g.concat_vectors(np.vstack([rows[i][plane].reshape(4, 200) for i in range(4)]).transpose().flatten().tolist(), (self.chunks*self.dim*2*2, 320))
                            .write(name="adjmat" + ("W" if drctn == WEST else "E") + "p" + str(plane), storage_req=tmat[drctn*2+plane].storage_request))
        g.add_mem_constraints(gcodechanges + negmulvecs, self.tcounter + self.tgcode, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        g.add_mem_constraints(counters + gcodes + gcodechanges + negmulvecs, counters + gcodes + gcodechanges + negmulvecs, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        if not tvec is None and not tmat is None:
            return vecres, matres, t + 82
        else:
            return counters, gcodes, negmulvecs, t + 82
    def chain_test(chunks, dim): #verify performance
        import timeit
        pgm_pkg = g.ProgramPackage(name="agcpackage", output_dir="agcpackage")
        with pgm_pkg.create_program_context("init_agc") as pcinit:
            agc = AdvanceGrayCode(chunks, dim)
            parallel = len(agc.tcounter)
        with pgm_pkg.create_program_context("agc") as pc:
            agc = AdvanceGrayCode(chunks, dim, agc)
            counters_mt, gcodes_mt, negmulvecs, _ = agc.build()
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
            invoke([device], iop, 0, 0, None)
        tloaddata = timeit.timeit(runnerinit, number=1)/1
        runner = lambda: invoke([device], iop, 1, 0, None)[0][0]
        origcounter = [agc.origcounter for _ in range(parallel)]
        origgcode = [agc.origgcode for _ in range(parallel)]        
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
        batchsize = 100000
        tactualbatch = timeit.timeit(actual, setup='gc.enable()', number=batchsize)/batchsize
        print_utils.cprint("File Size for " + iops[0] + ": " + str(os.path.getsize(iops[0])), "")
        print_utils.cprint("CPU Time: " + str(toracle) + " Groq Load Time: " + str(tloaddata) + " Groq Init Time: " + str(tactual) + " Groq Time Avg. " + str(batchsize) + " Batch (Speed-up " + str(toracle/tactualbatch) + "x): " + str(tactualbatch), "")
        device.close()
    
    def unit_test(chunks, dim): #verify accuracy
        with g.ProgramContext() as pc:
            agc = AdvanceGrayCode(chunks, dim, True)
            parallel = len(agc.tcounter)
            counters_mt, gcodes_mt, negmulvecs, _ = agc.build()
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
                inputs[agc.tcounter[i].name] = origcounter[i]
                inputs[agc.tgcode[i].name] = origgcode[i]
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

class LoopCorrections(g.Component):
    def __init__(self, chunks, dim, matpow, lastlc=None, **kwargs):
        super().__init__(**kwargs)
        self.chunks, self.dim, self.matpow = chunks, dim, (dim//2-1) if matpow is None else matpow
        self.tvec, self.tmat, self.tnorm, self.cx_diag = [], [], [], []       
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            dirstr = ("W" if drctn == WEST else "E") + str(plane) + "P"
            if lastlc is None or lastlc == True:
                self.tvec.append(g.input_tensor(shape=(chunks, dim*2*2), dtype=g.int8, name="A" + dirstr, layout=get_slice1(drctn, 43, plane)))
                self.tmat.append(g.input_tensor(shape=(chunks*dim*2*2, dim*2*2), dtype=g.int8, name="B" + dirstr, layout=get_slice16(drctn, s16rangeW if drctn == WEST else s16rangeE, plane)))
                self.tnorm.append(g.zeros(shape=(1, dim*2*2), dtype=g.uint8, name="norm" + dirstr, layout=get_slice1(drctn, 1, plane)))
                self.cx_diag.append(g.input_tensor(shape=(chunks, dim*2*2), dtype=g.int8, name="C" + dirstr, layout=get_slice1(drctn, 43, plane)))        
            else:
                self.tvec.append(tensor.create_shared_memory_tensor(memory_tensor=lastlc.tvec[drctn*2+plane], name="postA" + dirstr))
                self.tmat.append(tensor.create_shared_memory_tensor(memory_tensor=lastlc.tmat[drctn*2+plane], name="postB" + dirstr))
                self.tnorm.append(tensor.create_shared_memory_tensor(memory_tensor=lastlc.tnorm[drctn*2+plane], name="postnorm" + dirstr))
                self.cx_diag.append(tensor.create_shared_memory_tensor(memory_tensor=lastlc.cx_diag[drctn*2+plane], name="postC" + dirstr))
            g.add_mem_constraints(self.tvec + self.cx_diag, self.tvec + self.cx_diag, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints(self.tmat, self.tmat, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints(self.tnorm, self.tnorm, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        if lastlc == True: lastlc = False
        self.VMM = VecMatMul(self.chunks, self.dim*2*2, lastlc if lastlc is None or lastlc == False else lastlc.VMM)
        self.agc = AdvanceGrayCode(self.chunks, self.dim, lastlc if lastlc is None or lastlc == False else lastlc.agc)
        for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
            g.add_mem_constraints([self.VMM.consts[drctn]["maskqrttop"], self.VMM.consts[drctn]["maskqrttoppre"]], [self.tmat[drctn*2+plane]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
            g.add_mem_constraints(list(self.VMM.consts[drctn].values()) + list(self.VMM.vn.consts[drctn].values()), [self.agc.allzeros[drctn*2+plane], self.agc.allones[drctn*2+plane], self.agc.negtwo[drctn*2+plane], self.agc.shiftleft[drctn*2+plane], self.agc.gcodemasks[drctn*2+plane], self.agc.tcounter[drctn*2+plane],  self.agc.tgcode[drctn*2+plane]], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        for drctn in (WEST, EAST):
            g.add_mem_constraints(list(self.VMM.vn.consts[drctn].values()) + self.agc.tcounter + self.agc.tgcode, self.cx_diag[drctn*2:drctn*2+2] + self.tvec[drctn*2:drctn*2+2] + self.tnorm[drctn*2:drctn*2+2] + self.tmat[drctn*2:drctn*2+2] + list(self.VMM.consts[drctn].values()), g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
    def build(self):
        results_mt = [self.tvec]
        results_norm = [self.tnorm]
        pred = None
        diag_mt, norm = self.tvec, self.tnorm
        curt = 0
        with g.ResourceScope(name="vecmatmul", is_buffered=False, time=0, predecessors=None) as pred: #0 if pred is None else None, predecessors=None if pred is None else [pred]) as pred:
            for i in range(self.matpow):
                diag_mt, norm, curt = self.VMM.build(diag_mt, self.tmat, norm, curt)
                results_mt.append(diag_mt); results_norm.append(norm)
        flat = [y for x in results_mt for y in x]
        g.add_mem_constraints(flat, flat + self.cx_diag, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        flat = [y for x in results_norm for y in x]
        g.add_mem_constraints(flat, flat + self.agc.tcounter + self.agc.tgcode, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
        #VectorsToComplexMatrix(results_mt, results_norm)
        #with g.ResourceScope(name="vecmatfin", is_buffered=True, time=None, predecessors=[pred]) as pred:
        #    finresult_mt = VecMatMul(self.chunks, self.dim*2*2).build(self.cx_diag, results_mt)
        with g.ResourceScope(name="advgraycode", is_buffered=False, time=0, predecessors=None) as pred:
            self.cx_diag, self.tmat, curt = self.agc.build(self.cx_diag, self.tmat, curt)
        result_combined = g.concat_vectors(results_mt[-1], (4, self.chunks, self.dim*2*2))
        norm_combined = g.concat_vectors(results_norm[-1], (4, self.dim*2*2))
        result_combined.name = "result_combined"
        norm_combined.name = "norm_combined"
        result_combined.set_program_output()
        norm_combined.set_program_output()
        return result_combined, norm_combined
    def chain_test(chunks, dim, dual=False):
        import timeit        
        worstCase, useCplx, longDoubleOracle, matpow = False, True, False, dim//2-1
        pgm_pkg = g.ProgramPackage(name="mm", output_dir="mm")
        with pgm_pkg.create_program_context("init_mm_fp") as pcinit:
            lc = LoopCorrections(chunks, dim, matpow)
            tvec, tmat, cx_diag = lc.tvec, lc.tmat, lc.cx_diag
            parallel = len(tvec)*2*(1 if not dual else 2)
        with pgm_pkg.create_program_context("mm_fp") as pc:
            for drctn, plane in ((WEST, 0), (WEST, 1), (EAST, 0), (EAST, 1)):
                g.reserve_tensor(pcinit, pc, lc.tvec[drctn*2+plane])
                g.reserve_tensor(pcinit, pc, lc.tmat[drctn*2+plane])
                g.reserve_tensor(pcinit, pc, lc.cx_diag[drctn*2+plane])
            lc = LoopCorrections(chunks, dim, matpow, lc)
            result_mt, resnorm = lc.build()
        print_utils.infoc("\nAssembling model ...")
        iops = pgm_pkg.assemble()
        iop = runtime.IOProgram(iops[0])
        devices = [runtime.devices[0]] if not dual else runtime.devices
        
        if worstCase:
            if useCplx:
                originpvec = [np.full((dim,), ((1 << 53)-1)/(1<<53)+((1 << 53)-1)/(1<<53)*1j) for _ in range(parallel)]
                originpmat = [np.full((dim, dim), -((1 << 53)-1)/(1<<53)-((1 << 53)-1)/(1<<53)*1j) for _ in range(parallel)]
            else:    
                originpvec = [np.full((dim,), ((1 << 53)-1)/(1<<53)) for _ in range(parallel)]
                originpmat = [np.full((dim, dim), -((1 << 53)-1)/(1<<53)) for _ in range(parallel)]
        elif useCplx:
            originpvec = [np.random.rand(dim)*2-1 + (np.random.rand(dim)*2j-1j) for _ in range(parallel)]
            originpmat = [unitary_group.rvs(dim) for _ in range(parallel)]
        else:
            originpvec = [np.random.rand(dim)*2-1 for _ in range(parallel)]
            originpmat = [np.random.rand(dim, dim)*2-1 for _ in range(parallel)]
    
        oracleres = [None]
        def makeOracle():
            #export OMP_NUM_THREADS=64
            floattype = np.clongdouble if longDoubleOracle else np.cdouble #np.clongdouble uses a single thread, while np.cdouble uses 20 threads for an 80x80 np.cdouble matrix
            B = [np.stack([originpmat[i].transpose().astype(floattype) for i in range(parallel)], axis=0)]
            w = np.vstack([originpvec[i].astype(floattype) for i in range(parallel)]) #.reshape(parallel, 1, dim)
            gcodeCounter, lastgcode = [0], [0]
            @jit(nopython=True)
            def fastoracle(B, gcodeCounter, lastgcode):
                #v = w #3D-version, not supported by numba, but numba appears to be faster
                #for _ in range(matpow):
                #    v = v @ B
                #l = [x.reshape(dim) for x in np.split(v, parallel, axis=0)]
                l = []
                for i in range(parallel):
                    v = w[i]
                    for _ in range(matpow):
                        v = v @ B[i]
                    l.append(v.astype(np.cdouble))
                gcodeCounter += 1
                gcode = gcodeCounter ^ (gcodeCounter >> 1)
                v = gcode ^ lastgcode
                r =     (v > 0xFFFF) << 4; v >>= r
                shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift
                shift = (v > 0xF   ) << 2; v >>= shift; r |= shift
                shift = (v > 0x3   ) << 1; v >>= shift; r |= shift
                r |= (v >> 1) #r = v.bit_length()-1
                change = r + 1+3 #bit_length()-1==oneHotDecode, first 1 + 3 (or 4 for dual) rows are reserved for anchor plus parallelism 2^3/2^4
                B[:,change*2, :] = -B[:,change*2, :]
                B[:,change*2+1, :] = -B[:,change*2+1, :]                
                return l, gcodeCounter, gcode
            def oracle():
                oracleres[0], gcodeCounter[0], lastgcode[0] = fastoracle(B[0], gcodeCounter[0], lastgcode[0])
            return oracle
        oracle = makeOracle()
        batchsize = 30000
        toracle = timeit.timeit(oracle, number=batchsize)/batchsize
        print_utils.infoc("\nRunning on HW ...")
        np.set_printoptions(formatter={'int':hex}, threshold=sys.maxsize, floatmode='unique')
        # Create TSP runner and pass input dictionary of "tensor name" : "tensor data".        
        #from GroqBatch_wrapper import GroqBatch_wrapper
        #GroqBatch_wrapper(iop_file=iops[0])
        
        results, runfunc = [None], [None]
        def loaddata():
            fractionbits = 62 if useCplx else 63 #allow 8/7 integer bits
            inputs, osz = [], parallel//2//len(devices)
            exp_inpvecs, Z = [], np.zeros((chunks, dim*2, dim*2), dtype=np.int8)
            for device in devices:
                inputs.append({})
                device.open()
                device.load(iop[0], unsafe_keep_entry_points=True)
                device.load(iop[1], unsafe_keep_entry_points=True)
            for i in range(parallel//2):
                exp_inpvec0, normals = normalize_doubles(vector_real_to_complex(originpvec[i*2]), 0, fractionbits)
                inpvec0 = num_to_bits(normals, chunks)
                exp_inpmat0, normals = normalize_doubles(matrix_real_to_complex(originpmat[i*2]), None, fractionbits) #dimension 1 will cause chaining issues without shift corrections
                inpmat0 = num_to_bits(normals, chunks).reshape((chunks, dim*2, dim*2))
                exp_inpvec1, normals = normalize_doubles(vector_real_to_complex(originpvec[i*2+1]), 0, fractionbits)
                inpvec1 = num_to_bits(normals, chunks)
                exp_inpmat1, normals = normalize_doubles(matrix_real_to_complex(originpmat[i*2+1]), None, fractionbits) #dimension 1 will cause chaining issues without shift corrections
                inpmat1 = num_to_bits(normals, chunks).reshape((chunks, dim*2, dim*2))
                devidx = 1 if i >= osz else 0
                oidx = i % osz
                inputs[devidx][tvec[oidx].name] = np.hstack((inpvec0, inpvec1))
                inputs[devidx][tmat[oidx].name] = np.concatenate((np.concatenate((inpmat0, Z), axis=2), np.concatenate((Z, inpmat1), axis=2)), axis=1).reshape((chunks*dim*2*2, dim*2*2))
                inputs[devidx][cx_diag[oidx].name] = np.hstack((inpvec0, inpvec1)) #colswap
                exp_inpvecs.append(np.concatenate((2*fractionbits-63 - 7 - exp_inpvec0 - np.full((dim*2), exp_inpmat0), 2*fractionbits-63 - 7 - exp_inpvec1 - np.full((dim*2), exp_inpmat1))))
            inpnormvecs = [np.stack(exp_inpvecs[i*osz:(i+1)*osz]) for i in range(len(devices))]
            invoke(devices, iop, 0, 0, inputs)
            @jit(nopython=True)
            def bits_to_vector(bits, inpvecnorm, norm):
                l = []
                for i in range(osz):
                    normed = renormalize_doubles(bits_to_num(bits[i], 7), inpvecnorm[i] - norm[i].reshape(dim*2*2).astype(np.int32))
                    l.append(vector_complex_to_real(normed[:dim*2])); l.append(vector_complex_to_real(normed[dim*2:]))
                return l
            def actual():
                res, buffers[0] = invoke(devices, iop, 1, 0, None, lastouts[0], buffers[0])
                lastouts[0] = res
                newres = []                
                for i in range(len(devices)):
                    #the results come back truncating the lower 7*(chunks-1) bits
                    newres.extend(bits_to_vector(res[i][result_mt.name], inpnormvecs[i], res[i][resnorm.name]))
                results[0] = newres
            runfunc[0] = actual
        tloaddata = timeit.timeit(loaddata, number=1)/1
        actual = runfunc[0]
        buffers, lastouts = [None], [None] #* max_workers
        tactual = timeit.timeit(actual, number=1)/1; oracle()
        def actualbatch():
            for _ in range(batchsize): actual()
        perf_pro(actualbatch)
        for _ in range(batchsize): oracle()
        tactualbatch = timeit.timeit(actual, setup='gc.enable()', number=batchsize)/batchsize
        print_utils.cprint(("Dual" if dual else "Single") + " Groq Chip Parallelism: " + str(parallel) + " Double precision " + ("Complex" if useCplx else "Real") + " (" + str(dim) + ") Vector multiplied by (" + str(dim) + "x" + str(dim) + ") Matrix raised to power " + str(matpow) + " Oracle intermediate precision: " + ("LongDouble" if longDoubleOracle else "Double") + " Program File Size for " + iops[0] + ": " + str(os.path.getsize(iops[0])), "")
        print_utils.cprint("CPU Time: " + str(toracle) + " Groq Load Time: " + str(tloaddata) + " Groq Init Time: " + str(tactual) + " Groq Time Avg. " + str(batchsize) + " Batch (Speed-up " + str(toracle/tactualbatch) + "x): " + str(tactualbatch), "")
        oracleres, results = oracleres[0], results[0]
        for i in range(parallel):
            print_utils.infoc("\nComparing results with oracle ...")
            max_atol = max(abs(oracleres[i].reshape(-1) - results[i].reshape(-1)))
            if max_atol <= 0.001:
                print_utils.success(f"Test PASSED with a max tolerance of {max_atol}")
            else:
                print_utils.err(
                    f"Test FAILED with a max tolerance of {max_atol} (should be <= 0.001)"
                )
        for device in devices: device.close()
    def unit_test(chunks, dim):
        import timeit
        worstCase, useCplx, longDoubleOracle, matpow = False, True, False, 1 #dim//2-1
        with g.ProgramContext() as pc:
            lc = LoopCorrections(chunks, dim, matpow, True)
            tvec, tmat, cx_diag = lc.tvec, lc.tmat, lc.cx_diag
            parallel = len(tvec)*2
            result_mt, resnorm = lc.build()
            iop_file, json_file = compile_unit_test("lc")
        runner = g.create_tsp_runner(iop_file)
        if worstCase:
            if useCplx:
                originpvec = [np.full((dim,), ((1 << 53)-1)/(1<<53)+((1 << 53)-1)/(1<<53)*1j) for _ in range(parallel)]
                originpmat = [np.full((dim, dim), -((1 << 53)-1)/(1<<53)-((1 << 53)-1)/(1<<53)*1j) for _ in range(parallel)]
            else:    
                originpvec = [np.full((dim,), ((1 << 53)-1)/(1<<53)) for _ in range(parallel)]
                originpmat = [np.full((dim, dim), -((1 << 53)-1)/(1<<53)) for _ in range(parallel)]
        elif useCplx:
            originpvec = [np.random.rand(dim)*2-1 + (np.random.rand(dim)*2j-1j) for _ in range(parallel)]
            originpmat = [unitary_group.rvs(dim) for _ in range(parallel)]
        else:
            originpvec = [np.random.rand(dim)*2-1 for _ in range(parallel)]
            originpmat = [np.random.rand(dim, dim)*2-1 for _ in range(parallel)]
    
        oracleres, results = [None], [None]
        def makeOracle():
            #export OMP_NUM_THREADS=64
            floattype = np.clongdouble if longDoubleOracle else np.cdouble #np.clongdouble uses a single thread, while np.cdouble uses 20 threads for an 80x80 np.cdouble matrix
            B = [originpmat[i].transpose().astype(floattype) for i in range(parallel)]
            w = [originpvec[i].astype(floattype) for i in range(parallel)]
            def oracle():
                oracleres[0] = []
                for i in range(parallel):
                    v = w[i]
                    for _ in range(matpow):
                        v = v @ B[i]
                    oracleres[0].append(v.astype(np.cdouble))
            return oracle
        oracle = makeOracle()
        def actual():
            fractionbits = 62 if useCplx else 63 #allow 8/7 integer bits
            inputs = {}
            exp_inpvecs, Z = [], np.zeros((chunks, dim*2, dim*2), dtype=np.int8)
            for i in range(parallel//2):
                exp_inpvec0, normals = normalize_doubles(vector_real_to_complex(originpvec[i*2]), 0, fractionbits)
                inpvec0 = num_to_bits(normals, chunks)
                exp_inpmat0, normals = normalize_doubles(matrix_real_to_complex(originpmat[i*2]), 1 if matpow==1 else None, fractionbits) #dimension 1 will cause chaining issues without shift corrections
                inpmat0 = num_to_bits(normals, chunks).reshape((chunks, dim*2, dim*2))
                exp_inpvec1, normals = normalize_doubles(vector_real_to_complex(originpvec[i*2+1]), 0, fractionbits)
                inpvec1 = num_to_bits(normals, chunks)
                exp_inpmat1, normals = normalize_doubles(matrix_real_to_complex(originpmat[i*2+1]), 1 if matpow==1 else None, fractionbits) #dimension 1 will cause chaining issues without shift corrections
                inpmat1 = num_to_bits(normals, chunks).reshape((chunks, dim*2, dim*2))
                inputs[tvec[i].name] = np.hstack((inpvec0, inpvec1))
                inputs[tmat[i].name] = np.concatenate((np.concatenate((inpmat0, Z), axis=2), np.concatenate((Z, inpmat1), axis=2)), axis=1).reshape((chunks*dim*2*2, dim*2*2))
                inputs[cx_diag[i].name] = np.hstack((inpvec0, inpvec1)) #colswap
                exp_inpvecs.append(np.concatenate((2*fractionbits-63 - 7 - exp_inpvec0 - np.full((dim*2), exp_inpmat0), 2*fractionbits-63 - 7 - exp_inpvec1 - np.full((dim*2), exp_inpmat1))))
            res = runner(**inputs)
            results[0] = []
            for i in range(parallel//2):
                result = bits_to_num(res[result_mt.name][i], 7)
                norm = res[resnorm.name][i].reshape(dim*2*2)
                #the results come back truncating the lower 7*(chunks-1) bits
                normed = renormalize_doubles(result, exp_inpvecs[i] - norm.astype(np.int32))
                results[0].append(vector_complex_to_real(normed[:dim*2]))
                results[0].append(vector_complex_to_real(normed[dim*2:]))
        print_utils.infoc("\nRunning on HW ...")
        print_utils.cprint("Single Groq Chip Parallelism: " + str(parallel) + " Double precision " + ("Complex" if useCplx else "Real") + " (" + str(dim) + ") Vector multiplied by (" + str(dim) + "x" + str(dim) + ") Matrix raised to power " + str(matpow) + " Oracle intermediate precision: " + ("LongDouble" if longDoubleOracle else "Double") + " Program File Size for " + iop_file + ": " + str(os.path.getsize(iop_file)), "")
        oracle()
        actual()
        oracleres, results = oracleres[0], results[0]
        np.set_printoptions(formatter={'int':hex}, threshold=sys.maxsize, floatmode='unique')
        for i in range(parallel):
            print_utils.infoc("\nComparing results with oracle ...")
            max_atol = max(abs(oracleres[i].reshape(-1) - results[i].reshape(-1)))
            if max_atol <= 0.001:
                print_utils.success(f"Test PASSED with a max tolerance of {max_atol}")
            else:
                print_utils.err(
                    f"Test FAILED with a max tolerance of {max_atol} (should be <= 0.001)"
                )

def main():
    dim = 80 #dim X dim complex matrix
    bitsize = 64 #for fixed point representation will round up to nearest multiple of 7
    chunks = (bitsize + 7-1)//7 #ceiling division to be exact
    #VecNormalize.unit_test(chunks, dim)
    #VecMatMul.unit_test(chunks, dim)
    #AdvanceGrayCode.unit_test(chunks, dim)
    #AdvanceGrayCode.chain_test(chunks, dim)
    LoopCorrections.unit_test(chunks, dim)
    LoopCorrections.chain_test(chunks, dim)
    LoopCorrections.chain_test(chunks, dim, True)
    return

if __name__ == "__main__":
    main()
