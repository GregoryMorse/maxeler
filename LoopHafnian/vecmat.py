import numpy as np
import groq.api as g
import groq.api.instruction as inst
import groq.api.nn as nn
from groq.common import print_utils

def karatsuba_mul16(tensor1, tensor2):
    #if tensor1.tensor_type == g.uint16:
    shift8 = g.constant_tensor(shape=(1, 40), dtype=g.uint8)
    shift8.data = np.array([[8]*40], dtype=np.uint8)
    shift16 = g.constant_tensor(shape=(1, 40), dtype=g.uint8)
    shift16.data = np.array([[16]*40], dtype=np.uint8)
    mask8 = g.constant_tensor(shape=(1, 40), dtype=g.uint16)
    mask8.data = np.array([[(1<<8)-1]*40], dtype=np.uint16)
    with g.ResourceScope(name="mask1", is_buffered=True, time=0) as mask1:
        tl1 = g.bitwise_and(tensor1, mask8).write(name="tl1")
    with g.ResourceScope(name="mask2", is_buffered=True, predecessors=[mask1], time=None) as mask2:
        tl2 = g.bitwise_and(tensor2, mask8).write(name="tl2")
        #tl2 = inst.bitwise_and(tensor2, mask8)
    with g.ResourceScope(name="shift1", is_buffered=True, predecessors=[mask2], time=None) as shift1:
        tu1 = g.right_shift(tensor1, shift8).write(name="tu1")
    with g.ResourceScope(name="shift2", is_buffered=True, predecessors=[shift1], time=None) as shift2:        
        tu2 = g.right_shift(tensor2, shift8).write(name="tu2")
    with g.ResourceScope(name="sub1", is_buffered=True, predecessors=[shift2], time=None) as sub1:
        diff1 = g.add(tu1, tl1).write(name="diff1")
    with g.ResourceScope(name="sub2", is_buffered=True, predecessors=[sub1], time=None) as sub2:
        diff2 = g.add(tu2, tl2).write(name="diff2")
    with g.ResourceScope(name="subfix1", is_buffered=True, predecessors=[sub2], time=None) as subfix1:
        difffix1 = g.bitwise_and(diff1, mask8).write(name="difffix1")
    with g.ResourceScope(name="subfix2", is_buffered=True, predecessors=[subfix1], time=None) as subfix2:        
        difffix2 = g.bitwise_and(diff2, mask8).write(name="difffix2")
    #9 bits * 9 bits = all([i*j==((i&255)*(j&255)+(i&256)*j+(j&256)*(i&255)) for i in range(512) for j in range(512)])
    with g.ResourceScope(name="extra1", is_buffered=True, predecessors=[subfix2], time=None) as extra1:
        ex1 = g.left_shift(g.mask(g.greater(diff1, mask8), diff2).cast(g.uint32), shift16).write(name="ex1")
    with g.ResourceScope(name="extra2", is_buffered=True, predecessors=[extra1], time=None) as extra2:
        ex2 = g.left_shift(g.mask(g.greater(diff2, mask8), difffix1).cast(g.uint32), shift16).write(name="ex2")
    with g.ResourceScope(name="mul1", is_buffered=True, predecessors=[extra2], time=None) as mul1:
        z0 = g.mul(tl1, tl2, time=0).cast(g.uint32).write(name="z0")
    with g.ResourceScope(name="mul2", is_buffered=True, predecessors=[mul1], time=None) as mul2:
        z2 = g.left_shift(g.mul(tu1, tu2, time=0).cast(g.uint32), shift8).write(name="z2")
    with g.ResourceScope(name="mul3", is_buffered=True, predecessors=[mul2], time=None) as mul3:
        z1 = g.left_shift(g.mul(difffix1, difffix2, time=0).cast(g.uint32), shift8).write(name="z1")
    with g.ResourceScope(name="add1", is_buffered=True, predecessors=[mul3], time=None) as add1:
        res1 = g.add(g.add(g.add(g.left_shift(z2, shift8), z0), z1), ex1).write(name="res1")
    with g.ResourceScope(name="add2", is_buffered=True, predecessors=[add1], time=None) as add2:
        res2 = g.sub(g.add(res1, ex2), z2).write(name="res2")
    with g.ResourceScope(name="s1", is_buffered=True, predecessors=[add2], time=None) as s1:
        z0s = g.left_shift(z0, shift8).write(name="z0s")
    with g.ResourceScope(name="add3", is_buffered=True, predecessors=[s1], time=None) as add3:
        res = g.sub(res2, z0s).write(name="res")
    return res

def main():
    # IMP DETAILS on using nn.Matmul component:
    # - Expects both inputs as rank-2 tensor.
    # - The inner dimension on both tensors should match.

    # Create 2 input tensors.
    #t1 = g.input_tensor(shape=(100, 1000), dtype=g.float16, name="A")
    #t2 = g.input_tensor(shape=(400, 1000), dtype=g.float16, name="B")
    #long double has 63+1 significant bits + 1 sign bits, 65-8=57 8*7=56+1 requires int8, 8 7-bit int8 and a 1-bit int8
    #long double has 16 exponent bits, can reduce it to 15, fits in int16 
    #must first adjust each row vs column to fixed point to handle the additions 
    t1 = g.input_tensor(shape=(1, 40), dtype=g.uint16, name="A")
    t2 = g.input_tensor(shape=(1, 40), dtype=g.uint16, name="B")
    #t3 = g.input_tensor(shape=(1, 40), dtype=g.uint16, name="C")
    #t4 = g.input_tensor(shape=(1, 40), dtype=g.uint16, name="D")

    print_utils.infoc(
        "\nBuilding FP16 matmul for input tensors {} x {}".format(t1.shape, t2.shape)
    )
    # Instantiate matmul component.
    #mm = nn.MatMul(time=20, buffer_output=True)
    #mm = g.mul(t1, t2, time=0)
    mm = karatsuba_mul16(t1, t2)
    # ^^ Don't need to select any mxm plane or set the memory layouts.
    # Only thing you need to set is the time at which matmul should be scheduled.
    # Also you can pass buffer_output=True to avoid explicit write to memory.

    # Build matmul component.
    result_mt = mm#.write(name="result_mt") #mm(t1, t2)

    # Mark result_mt as program output.
    result_mt.set_program_output()

    print_utils.infoc("\nCompiling model ...")
    # Compile program to generate IOP. Also generate groqview JSON dump file and
    # check for potential stream conflicts.
    iop_file = g.compile(
        base_name="mm_fp16", gen_vis_data=True, check_stream_conflicts=True
    )

    # Generate random input data and oracle for comparision.
    inp1 = np.random.randint(0, (1<<16)-1, size=t1.shape, dtype=np.uint16)
    inp2 = np.random.randint(0, (1<<16)-1, size=t2.shape, dtype=np.uint16)
    oracle = inp1.astype(np.uint32) * inp2.astype(np.uint32) # np.matmul(inp1, inp2.transpose(), dtype=np.uint32)

    print_utils.infoc("\nRunning on HW ...")
    # Create TSP runner and pass input dictionary of "tensor name" : "tensor data".
    runner = g.create_tsp_runner(iop_file)
    inputs = {t1.name: inp1, t2.name: inp2}
    results = runner(**inputs)

    print_utils.infoc("\nComparing results with oracle ...")
    actual = results[result_mt.name]
    max_atol = max(abs(oracle.reshape(-1) - actual.reshape(-1)))
    print(actual, oracle)
    if max_atol <= 0.001:
        print_utils.success(f"Test PASSED with a max tolerance of {max_atol}")
    else:
        print_utils.err(
            f"Test FAILED with a max tolerance of {max_atol} (should be <= 0.001)"
        )


if __name__ == "__main__":
    main()
