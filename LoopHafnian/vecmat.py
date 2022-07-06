
#export PATH=$PATH:/nix/store/myclxgzxiqrlhgw5b6h4mjmvyks2c5lz-Groq.View.server.groqview-streams/bin/
#/nix/store/myclxgzxiqrlhgw5b6h4mjmvyks2c5lz-Groq.View.server.groqview-streams/bin/groqview-streams

import sys
MANT_DIG = sys.float_info.mant_dig
import numpy as np
import groq.api as g
import groq.api.instruction as inst
import groq.api.nn as nn
from groq.common import print_utils

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
    
def num_to_bits(num, bitsize=64):
    chunks = (bitsize + 7-1)//7 #ceiling divide by 7
    if len(num.shape) == 2:
        res, shp = np.repeat(num[np.newaxis,:,:], chunks, axis=0), (chunks, 1, 1)
    else: res, shp = np.repeat(num[np.newaxis,:], chunks, axis=0), (chunks, 1)
    return (res >> np.arange(0, 7 * chunks, 7).reshape(shp)) & ((1 << 7)-1)
def bits_to_num(num):
    return np.sum(num.astype(np.int64) << np.arange(0, 7 * len(num), 7), axis=1)
def normalize_doubles(num, dimension):
    mantissas, exponents = np.frexp(num)
    maxexp = np.amax(exponents, axis=dimension)
    adjustmant = mantissas / (1 << (maxexp - exponents))
    return maxexp, np.rint(np.ldexp(adjustmant, maxexp+62)).astype(np.int64) #(64, -62) bit fixed point integers
def renormalize_doubles(num, exp):
    return num.astype(np.double) / (2 ** exp.astype(np.double))
def main():
    import timeit
    dim = 10
    chunks = 10
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
    tvec = g.input_tensor(shape=(chunks, dim), dtype=g.int8, name="A", layout="H1(W), -1, S16")
    tmat = g.input_tensor(shape=(dim * chunks, dim), dtype=g.int8, name="B", layout="H1(W), -1, S1")

    print_utils.infoc(
        "\nBuilding FP16 matmul for input tensors {} x {}".format(tvec.shape, tmat.shape)
    )
    # Instantiate matmul component.
    #result_mt = karatsuba_mul16(t1, t2, dim)
    mm = nn.MatMul(time=0, buffer_output=True)
    #tpose = nn.TransposeMatrix(time=0, buffer_output=False, is_resource_scope=False)
    # ^^ Don't need to select any mxm plane or set the memory layouts.
    # Only thing you need to set is the time at which matmul should be scheduled.
    # Also you can pass buffer_output=True to avoid explicit write to memory.

    # Build matmul component.
    with g.ResourceScope(name="matmul", is_buffered=True, time=0) as matmul:
        result_mt = mm.build(tmat, tvec) #.reshape(dim, chunks, chunks)
    #with g.ResourceScope(name="tp", is_buffered=True, predecessors=[matmul], time=None) as tp:
    #    result_mt = tpose.build(result_mt)
    
    #split_result[0].split_vectors([chunks]*dim)
    
    #split_result = g.split_pipelines(result_mt.read(), logical_shapes=(chunks, dim))
    maskqrt = g.constant_tensor(shape=(dim, chunks), dtype=g.int32, layout="-1, H1, S4")
    maskqrt.data = np.array([[(1<<7)-1]*chunks for _ in range(dim)], dtype=np.int32)
    maskqrttop = g.constant_tensor(shape=(dim, chunks), dtype=g.int32, layout="-1, H1, S4")
    maskqrttop.data = np.array([[(1<<7)-1]*(chunks-1)+[-1] for _ in range(dim)], dtype=np.int32)
    shiftqrt = g.constant_tensor(shape=(dim, chunks), dtype=g.int32, layout="-1, H1, S4")
    shiftqrt.data = np.array([[7]*chunks for _ in range(dim)], dtype=np.int32)
    #g.resolve_storage_requests()
    print(result_mt.shape)
    print(maskqrt.physical_shape, maskqrt.layout)
    #maskupper = g.constant_tensor(shape=(dim, chunks), dtype=g.uint32)
    #maskupper.data = np.array([[(1<<32)-1]*chunks for _ in range(dim)], dtype=np.uint32)
    #for i in range(1):    
    split_result = g.split_vectors(result_mt, [dim]*chunks)
    checksplits = [x for x in split_result]
    checksplits[0].set_program_output()
    print(len(split_result), split_result[0].shape, split_result[0].physical_shape, split_result[0].layout)
    #g.add_mem_constraints(split_result, split_result, g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)
    pred = matmul
    perm_rq = g.tensor.create_permutor_request(perm=[1], num_perm=1)
    for i in range(chunks-1):
        with g.ResourceScope(name="shift" + str(i), is_buffered=True, predecessors=[pred], time=None) as shift:
            #must be an arithmetic right shift (sign filled), not logical, but with signed types, this occurs
            shifts = g.right_shift(split_result[i].read(streams=g.SG4_E[0], time=0), shiftqrt.read(streams=g.SG4_E[1]), alus=[0], output_streams=g.SG4_W[0]).write(name="shiftres" + str(i))
        with g.ResourceScope(name="mask" + str(i), is_buffered=True, predecessors=[shift], time=None) as mask:
            masks = g.bitwise_and(split_result[i].read(streams=g.SG4_E[0], time=0), maskqrt.read(streams=g.SG4_E[1]), alus=[0], output_streams=g.SG4_E[0]) #.write(name="maskres" + str(i))
            masks = g.shift(masks, 1, permutor_id=perm_rq, shift_src=[g.instruction.NEW_SRC], output_streams=g.SG4_W[1]).write(name="maskres" + str(i)) #element shift right by 1
        with g.ResourceScope(name="add" + str(i), is_buffered=True, predecessors=[mask], time=None) as pred:
            split_result[i+1] = g.add(g.add(shifts, masks, alus=[0]), split_result[i+1].read(streams=g.SG4_E[2]), alus=[1], output_streams=g.SG4_W[1], time=3).write(name="split" + str(i))
            
    for i in range(chunks-1):
        with g.ResourceScope(name="finshift" + str(i), is_buffered=True, predecessors=[pred], time=None) as shift:
            shifts = g.right_shift(split_result[-1].read(streams=g.SG4_E[0], time=0), shiftqrt.read(streams=g.SG4_E[1]), alus=[0], output_streams=g.SG4_E[0]) #.write(name="finshiftres" + str(i))
            shifts = g.shift(shifts, -1, permutor_id=perm_rq, shift_src=[g.instruction.NEW_SRC], output_streams=g.SG4_W[1]).write(name="finshiftres" + str(i)) #element shift left by 1
        with g.ResourceScope(name="finmask" + str(i), is_buffered=True, predecessors=[shift], time=None) as mask:
            masks = g.bitwise_and(split_result[-1].read(streams=g.SG4_E[0], time=0), maskqrttop.read(streams=g.SG4_E[1]), alus=[0], output_streams=g.SG4_W[0]).write(name="finmaskres" + str(i))
        with g.ResourceScope(name="finadd" + str(i), is_buffered=True, predecessors=[mask], time=None) as pred:
            split_result[-1] = g.add(shifts, masks, alus=[0], output_streams=g.SG4_W[1], time=3).write(name="finsplit" + str(i))
        

    # Mark result_mt as program output.
    split_result[-1].set_program_output()    
    result_mt.set_program_output()
    #result_st = tvec.matmul(tmat, planes=[0], time=20)
    #result_mt = result_st.write(
    #    name="mm_result", program_output=True, layout="H1(W), -1, S4"
    #)

    print_utils.infoc("\nCompiling model ...")
    # Compile program to generate IOP. Also generate groqview JSON dump file and
    # check for potential stream conflicts.
    iop_file = g.compile(
        base_name="mm_fp", gen_vis_data=True, check_stream_conflicts=True
    )
    g.write_visualizer_data("mm_fp")

    # Generate random input data and oracle for comparision.
    #inp1 = np.random.randint(0, (1<<16)-1, size=t1.shape, dtype=np.uint16)
    #inp2 = np.random.randint(0, (1<<16)-1, size=t2.shape, dtype=np.uint16)
    originpvec = np.random.rand(dim)
    originpmat = np.random.rand(dim, dim)
    #originpvec = np.random.randint(-(1<<63), (1<<63)-1, size=(dim), dtype=np.int64)
    #originpmat = np.random.randint(-(1<<63), (1<<63)-1, size=(dim, dim), dtype=np.int64)
    exp_inpvec, normals = normalize_doubles(originpvec, 0)
    inpvec = np.array(num_to_bits(normals, 64), dtype=np.int8)
    #print(np.array(num_to_bits(originpmat, 64), dtype=np.int8).shape)
    exp_inpmat, normals = normalize_doubles(originpmat, 1)
    inpmat = np.array(num_to_bits(normals, 64), dtype=np.int8)
    oracleres = [None]
    def oracle():
        oracleres[0] = np.matmul(originpvec, originpmat.transpose())
    toracle = timeit.timeit(oracle, number=100)/100

    print_utils.infoc("\nRunning on HW ...")
    np.set_printoptions(formatter={'int':hex})
    # Create TSP runner and pass input dictionary of "tensor name" : "tensor data".
    runner = g.create_tsp_runner(iop_file)
    #inputs = {t1.name: inp1, t2.name: inp2}
    inputs = {tvec.name: inpvec, tmat.name: inpmat.reshape(tmat.shape)}
    results = [None]
    def actual():
        results[0] = runner(**inputs)
    tactual = timeit.timeit(actual, number=100)/100
    print(toracle, tactual)
    oracleres, results = oracleres[0], results[0]
    """
    chipres = inpmat.astype(np.int32) @ inpvec.transpose().astype(np.int32)
    print(results[checksplits[0].name], chipres[0,:,:])
    
    s = sum(chipres[j,:,:].astype(np.int64) << (7*j) for j in range(chunks))
    print(s)
    print(sum(s[:,j] << (7*j) for j in range(chunks)))
    print(chipres)
    for i in range(chunks-1):
        chipres[i+1,:,:] += (chipres[i,:,:] >> 7) + np.hstack(((chipres[i,:,:] & ((1<<7)-1))[:,1:], np.zeros((dim, 1), dtype=chipres.dtype)))
    print(chipres)
    """
    print_utils.infoc("\nComparing results with oracle ...")
    """
    actual = results[result_mt.name]
    actual = actual.transpose().reshape((chunks, chunks, dim)).astype(np.int64)
    realactual = np.zeros((dim,), dtype=np.int64)
    print(actual.dtype)
    for i in range(chunks):
        for j in range(chunks):
            realactual += actual[i,j,:] <<(7*(i+j))
    print(realactual, exp_inpvec + exp_inpmat + 62)
    actual = renormalize_doubles(realactual, exp_inpvec + exp_inpmat + 62)
    """
    a = bits_to_num(results[split_result[-1].name])
    #the results come back truncating the lower 7*(chunks-1) bits
    #however we must left-align the result
    print(a)
    #a = chipres[-1,:].astype(np.int64)
    actual = renormalize_doubles(a, exp_inpvec + exp_inpmat + 62 - 64+7*(chunks-1))
    print(oracleres.shape, oracleres.dtype, actual.shape, actual.dtype)
    max_atol = max(abs(oracleres.reshape(-1) - actual.reshape(-1)))
    print(oracleres, actual)
    if max_atol <= 0.001:
        print_utils.success(f"Test PASSED with a max tolerance of {max_atol}")
    else:
        print_utils.err(
            f"Test FAILED with a max tolerance of {max_atol} (should be <= 0.001)"
        )


if __name__ == "__main__":
    main()
