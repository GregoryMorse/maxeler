package permanenttest;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.function.BiFunction;
import java.math.BigInteger;

import maxpower.utils.CollectionUtils;

import com.maxeler.platform.max5.manager.XilinxAlveoU250Manager;
import com.maxeler.platform.max5.manager.BuildConfig;
import com.maxeler.platform.max5.manager.ImplementationStrategy;
import com.maxeler.platform.max5.manager.SynthesisStrategy;
import com.maxeler.platform.max5.toolchain.VivadoClockDef;
import com.maxeler.platform.max5.toolchain.MmcmE3Calc;
import com.maxeler.platform.max5.board.VivadoMaxelerOsPlatform;
import com.maxeler.platform.max5.board.XilinxAlveoU250Board;

import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerKernelBase;
import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerRouting;
//import com.maxeler.maxcompiler.v2.managers.custom.api.ManagerKernel;
import com.maxeler.maxcompiler.v2.managers.custom.DFELink;
import com.maxeler.maxcompiler.v2.managers.custom.blocks.Fanout;
//import com.maxeler.maxcompiler.v2.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v2.managers.custom.ManagerClock;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelConfiguration;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelConfiguration.OptimizationOptions;

import com.maxeler.maxcompiler.v2.kernelcompiler.KernelBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelLite;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelLite.IO.PushInput;
//import com.maxeler.maxcompiler.v2.kernelcompiler.KernelLite.IO.PullInput;
import com.maxeler.maxcompiler.v2.kernelcompiler.KernelLite.IO.PushOutput;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEVar;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFix;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFix.SignMode;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.base.DFEFloat;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVector;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEVectorTypeBase;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEComplex;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.composite.DFEComplexType;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.KernelObject;
import com.maxeler.maxcompiler.v2.kernelcompiler.types.KernelObjectVectorizable;
import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.Bitops;
//import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.memory.Memory;
//import com.maxeler.maxcompiler.v2.utils.MathUtils;
import com.maxeler.maxcompiler.v2.kernelcompiler.RoundingMode;
//import com.maxeler.maxcompiler.v2.kernelcompiler.stdlib.FloatingPoint;
import com.maxeler.maxcompiler.v2.kernelcompiler.Optimization;
import com.maxeler.maxcompiler.v2.kernelcompiler.Optimization.PipelinedOps;
import com.maxeler.maxcompiler.v2.kernelcompiler.op_management.MathOps;
import com.maxeler.maxcompiler.v2.kernelcompiler.op_management.FanoutLimitType;
import com.maxeler.maxcompiler.v2.utils.MathUtils;
//import com.maxeler.maxcompiler.v2.utils.Bits;

import com.maxeler.maxcompiler.v2.kernelcompiler.KernelLib;
import com.maxeler.maxcompiler.v2.kernelcompiler.SMIO;
import com.maxeler.maxcompiler.v2.statemachine.DFEsmStateValue;
import com.maxeler.maxcompiler.v2.statemachine.DFEsmExpr;
import com.maxeler.maxcompiler.v2.statemachine.DFEsmInput;
import com.maxeler.maxcompiler.v2.statemachine.DFEsmOutput;
import com.maxeler.maxcompiler.v2.statemachine.DFEsmValue;
//import com.maxeler.maxcompiler.v2.statemachine.StateMachine;
import com.maxeler.maxcompiler.v2.statemachine.StateMachineLib;
import com.maxeler.maxcompiler.v2.statemachine.kernel.KernelStateMachine;
import com.maxeler.maxcompiler.v2.statemachine.types.DFEsmValueType;

import com.maxeler.maxeleros.resourceestimation.KernelEstimatorVisitor;
import com.maxeler.maxeleros.managercompiler.graph.nodes.KernelGraphVisitor;
import com.maxeler.photon.resource_annotation.ResourceComponent;
import com.maxeler.photon.graph_passes.maxdc_gen.OptionallyInlinedLogicNode;
import com.maxeler.photon.core.PhotonDesignData;
import com.maxeler.photon.core.VarTyped;
import com.maxeler.photon.maxcompilersim.CodeBlockRoot;
import com.maxeler.photon.maxcompilersim.Expression;
import com.maxeler.photon.maxcompilersim.SimCodeType;
import com.maxeler.photon.maxcompilersim.COutput;
import com.maxeler.photon.maxcompilersim.ExpInput;
import com.maxeler.photon.maxcompilersim.CodeContext;
import com.maxeler.photon.core.Node;
import com.maxeler.photon.core.Var;
import com.maxeler.photon.types.HWType;
import com.maxeler.photon.core.PhotonException;
import com.maxeler.photon.nodes.NodeConstant;
import com.maxeler.photon.nodes.ConstantFold;
import com.maxeler.maxdc.resource_usage.EntityResourceUsage;
import com.maxeler.maxdc.Entity;
import com.maxeler.maxdc.Signal;
import com.maxeler.maxdc.Reg;
import com.maxeler.maxdc.LogicSource;
import com.maxeler.maxdc.EntityStructural;
import com.maxeler.maxdc.portable.KeepHierarchy;
import com.maxeler.utils.Bits;
import java.util.Map;
import java.util.Set;
import java.util.EnumSet;
import java.lang.reflect.Field;


public class Utility {
    /**
    @brief Calculates the n-th power of 2.
    @param n An natural number
    @return Returns with the n-th power of 2.
    */
    public static long power_of_2(long n) {
        if (n == 0) return 1;
        if (n == 1) return 2;

        return 2 * power_of_2(n - 1);
    }
    public static int bitlength(long i) {
        int r = 0;
        while ((i >>= 1) != 0) r++;
        return r + 1;
    }
    //Alveo U250: MmcmFInMax, MmcmFInMin, MmcmFOutMax, MmcmFOutMin, MmcmFVcoMax, MmcmFVcoMin, MmcmFVcoMax, MmcmFVcoMin - SPEED_2L("SPEED_2L", 3, "2L", 933.0, 10.0, 725.0, 6.25, 1600.0, 800.0, 500.0, 10.0),
    public static TreeSet<Double> getLowJitterClockFrequency(int scale, XilinxAlveoU250Manager base)
    {
        double minFreq = ((XilinxAlveoU250Board)base.getBuildManager().getPlatform()).getBoardCapabilities().getFpga().getMmcmFOutMin(); //UltrascalePlusPart.FIGD2104
        double maxFreq = ((XilinxAlveoU250Board)base.getBuildManager().getPlatform()).getBoardCapabilities().getFpga().getMmcmFOutMax();
        double mmcmVcoMax = ((XilinxAlveoU250Board)base.getBuildManager().getPlatform()).getBoardCapabilities().getFpga().getMmcmFVcoMax();
        int minRange = (int)(minFreq * scale), maxRange = (int)(maxFreq * scale);
        TreeSet<Double> set = new TreeSet<>();
        LinkedList<Pair<Double, Double>> scores = new LinkedList<>();
        ArrayList<VivadoClockDef> list = new ArrayList<>();
        VivadoMaxelerOsPlatform platform = VivadoMaxelerOsPlatform.get(base.getBuildManager());
        VivadoClockDef defClock = new VivadoClockDef((float)platform.getStreamClockGeneratorInputFrequency());
        for (int i = minRange; i < maxRange; i++) {
            list.clear(); 
            double freq = (double)i / (double)scale;
            VivadoClockDef targetClockDef = new VivadoClockDef((float)freq);
            list.add(targetClockDef);
            int[] divs = null;
            try {
                MmcmE3Calc calc = new MmcmE3Calc(base.getBuildManager(), platform.getFPGAPart(), defClock, list);
                divs = calc.getClkOutDivide();
                double score = (calc.getDivClkDivide() * divs[0] + calc.getClkFbMult() * mmcmVcoMax / calc.getVcoFreq());
                if (scores.size() >= 10 * scale) scores.remove();
                scores.add(new Pair<Double, Double>(freq, score));
                //System.out.println(divs[0] + " " + calc.getDivClkDivide() + " " + calc.getClkFbMult() + " " + calc.getVcoFreq() + " " + calc.getClkOutFreq()[0] + " " + freq + " " + score); 
            } catch (com.maxeler.maxdc.MaxDCException e) { continue; }
            if (scores.size() == 10 * scale) {
                set.add(scores.stream().min((x, y) -> x.second == y.second ? 0 : (x.second < y.second ? -1 : 1)).get().first); 
            }
        }
        System.out.println(set);
        return set;
    }
    
    public static char[] charAdapter(Character[] l)
    {
        char[] res = new char[l.length];
        for (int i = 0; i < l.length; i++) res[i] = l[i];
        return res;
    }
    public static BigInteger factorial(BigInteger x) {
        return x.compareTo(BigInteger.ONE) <= 0 ? BigInteger.ONE : x.multiply(factorial(x.subtract(BigInteger.ONE)));
    }
    public static BigInteger choose(BigInteger n, BigInteger k) {
        return factorial(n).divide(factorial(n.subtract(k)).multiply(factorial(k)));
    }
    public static int getPartialSumIntBits(int n)
    {
        //import math; [max(sum((2*(k+1)-n)**n * math.comb(n-1, k) for k in range(1, n, 2)), sum((2*(k+1)-n)**n * math.comb(n-1, k) for k in range(0, n, 2)))/n**n for n in range(41)]
        BigInteger nton = BigInteger.valueOf(n).pow(n);
        return 
            IntStream.range(1, (n+2) / 2).mapToObj(k -> BigInteger.valueOf(2*(2*k-1+1)-n).pow(n).multiply(Utility.choose(BigInteger.valueOf(n-1), BigInteger.valueOf(2*k-1)))).reduce(BigInteger.ZERO, BigInteger::add).max(
            IntStream.range(0, (n+1) / 2).mapToObj(k -> BigInteger.valueOf(2*(2*k+1)-n).pow(n).multiply(Utility.choose(BigInteger.valueOf(n-1), BigInteger.valueOf(2*k)))).reduce(BigInteger.ZERO, BigInteger::add))
            .add(nton.subtract(BigInteger.ONE))
            .divide(nton).intValue();
    }    
    public static class MagicBinCoeff
    {
        public static class MagicDivision
        {
            BigInteger val;
            int shift;
            public MagicDivision(BigInteger val, int shift) {
                this.val = val; this.shift = shift;
            }
        }
        public static MagicDivision magicgu(BigInteger nmax, BigInteger d)
        {
            BigInteger nc = nmax.add(BigInteger.ONE).divide(d).multiply(d).subtract(BigInteger.ONE);
            int nbits = nmax.bitLength() - 2;
            for (int p = 0; p <= 2 * nbits; p++) {
                BigInteger twotop = BigInteger.ONE.shiftLeft(p);
                if (twotop.compareTo(nc.multiply(d.subtract(BigInteger.ONE).subtract(twotop.subtract(BigInteger.ONE).mod(d)))) > 0) {
                    BigInteger m = twotop.add(d).subtract(BigInteger.ONE).subtract(twotop.subtract(BigInteger.ONE).mod(d)).divide(d);
                    return new MagicDivision(m, p);
                }
            }
            return null;
        }
        public static class MagicInfo
        {
            BigInteger[] vals;
            int[] shifts; 
            int largestBinCoeffBits;
            int largestMpliedBinCoeffBits;
            int largestMagicNumBits;
            public MagicInfo(BigInteger[] vals, int[] shifts, int largestBinCoeffBits, int largestMpliedBinCoeffBits, int largestMagicNumBits) {
                this.vals = vals; this.shifts = shifts;
                this.largestBinCoeffBits = largestBinCoeffBits;
                this.largestMpliedBinCoeffBits = largestMpliedBinCoeffBits;
                this.largestMagicNumBits = largestMagicNumBits;
            } 
        }        
        public static MagicInfo get_bincoeff_magic(int dimension)
        {
            BigInteger dim = BigInteger.valueOf(dimension);
            BigInteger largestbincoeff = choose(dim, dim.shiftRight(1)).multiply(dim.shiftRight(1).add(BigInteger.ONE));
            MagicDivision[] objs = IntStream.range(1, dimension+1+1).mapToObj(d -> magicgu(largestbincoeff, BigInteger.valueOf(d))).toArray(MagicDivision[]::new);
            BigInteger[] vals = Arrays.stream(objs).map(x -> x.val).toArray(BigInteger[]::new);
            int[] shifts = Arrays.stream(objs).mapToInt(x -> x.shift).toArray();
            return new MagicInfo(vals, shifts, choose(dim, dim.shiftRight(1)).bitLength(), largestbincoeff.bitLength(),
                Arrays.stream(vals).mapToInt(x -> x.bitLength()).max().getAsInt());
        }
    }    
    
    public static enum ReduceOperations {
        ADD,
        MUL
    }

    public static class TreeReducer {
        public static DFEComplex reduce(ReduceOperations op, List < DFEComplex > toReduce, int treeFactor) {
            while (true) {
                List < DFEComplex > reduced = new ArrayList < > ();
                for (List < DFEComplex > group: CollectionUtils.grouped(toReduce, treeFactor)) {
                    reduced.add(reduceSingle(op, group));
                }
                toReduce = reduced;
                if (toReduce.size() == 1) {
                    break;
                }
            }
            return toReduce[0];
        }

        private static DFEComplex reduceSingle(ReduceOperations op, List < DFEComplex > toReduce) {
            DFEComplex result = toReduce.get(0);
            for (int i = 1; i < toReduce.size(); i++) {
                switch (op) {
                    case ADD:
                        {
                            result = result + toReduce[i];
                        }
                    case MUL:
                        {
                            result = result * toReduce[i];
                        }
                }
            }
            return result;
        }
    }
    public static class Pair<A, B> {
        private A first;
        private B second;
    
        public Pair(A first, B second) {
            super();
            this.first = first;
            this.second = second;
        }
    
        public int hashCode() {
            int hashFirst = first != null ? first.hashCode() : 0;
            int hashSecond = second != null ? second.hashCode() : 0;
    
            return (hashFirst + hashSecond) * hashSecond + hashFirst;
        }
    
        public boolean equals(Object other) {
            if (other instanceof Pair<?, ?>) {
                Pair<?, ?> otherPair = (Pair<?, ?>) other;
                return 
                ((  this.first == otherPair.first ||
                    ( this.first != null && otherPair.first != null &&
                      this.first.equals(otherPair.first))) &&
                 (  this.second == otherPair.second ||
                    ( this.second != null && otherPair.second != null &&
                      this.second.equals(otherPair.second))) );
            }
    
            return false;
        }
    
        public String toString()
        { 
               return "(" + first + ", " + second + ")"; 
        }
    
        public A getFirst() {
            return first;
        }
    
        public void setFirst(A first) {
            this.first = first;
        }
    
        public B getSecond() {
            return second;
        }
    
        public void setSecond(B second) {
            this.second = second;
        }
    }
    public static DFEVar binaryAdder(DFEVar[] vars, boolean[] issub, int[] pipelineDelay, Integer[] totalDelay, KernelBase<?> base) //O(ceil(log_3 n)) + additional 1 if negation occurs...at 6 additions this routine is more efficient than the default chaining
    {
        int limit = vars.length;
        while (limit > 1) {
            if (pipelineDelay[0] != pipelineDelay[1]) {
                vars[0] = base.optimization.pipeline(vars[0]);
                pipelineDelay[0]++;
            } else {
                int limitNew, i;
                for (i = 0, limitNew = 0; i < limit; i++, limitNew++) {
                    if (i >= limit-1 || pipelineDelay[i] != pipelineDelay[i+1]) {
                        pipelineDelay[limitNew] = pipelineDelay[i];
                        vars[limitNew] = vars[i];
                        issub[limitNew] = issub[i];                        
                    } else {
                        pipelineDelay[limitNew] = pipelineDelay[i] + 1;                        
                        if (!issub[i]) { //first argument must be positive for efficient TriArith
                            List<List<DFEVar>> chunks = getBitChunks128(vars[i], vars[i+1], issub[i+1], true);
                            if (chunks != null) pipelineDelay[limitNew] += chunks.size() - 1;
                            vars[limitNew] = addExact(vars[i], vars[i+1], issub[i+1], base);
                            issub[limitNew] = false;
                        } else if (!issub[i+1]) {
                            List<List<DFEVar>> chunks = getBitChunks128(vars[i+1], vars[i], issub[i], true);
                            if (chunks != null) pipelineDelay[limitNew] += chunks.size() - 1;
                            vars[limitNew] = addExact(vars[i+1], vars[i], issub[i], base);
                            issub[limitNew] = false;
                        } else { //both are negative
                            List<List<DFEVar>> chunks = getBitChunks128(vars[i], vars[i+1], false, true);
                            if (chunks != null) pipelineDelay[limitNew] += chunks.size() - 1;
                            vars[limitNew] = addExact(vars[i], vars[i+1], false, base);
                            issub[limitNew] = true;
                        }
                        i++;
                    }
                }
                limit = limitNew;
            }
            DFEVar[] varsCapture = Arrays.copyOf(vars, limit);
            int[] pipelineDelayCapture = Arrays.copyOf(pipelineDelay, limit);
            boolean[] issubCapture = Arrays.copyOf(issub, limit);
            int[] idxsort = IntStream.range(0, limit).mapToObj(Integer::valueOf).sorted((i, j) -> {
                int c = Integer.compare(pipelineDelayCapture[i], pipelineDelayCapture[j]);
                int s = Integer.compare(((DFEFix)varsCapture[i].getType()).getOffset(), ((DFEFix)varsCapture[j].getType()).getOffset());
                return c != 0 ? c : (s != 0 ? s : Integer.compare(i, j));
                }).mapToInt(Integer::intValue).toArray();
            vars = IntStream.of(idxsort).mapToObj(i -> varsCapture[i]).toArray(DFEVar[]::new);
            issub = booleanAdapter(IntStream.of(idxsort).mapToObj(i -> issubCapture[i]).toArray(Boolean[]::new));
            pipelineDelay = IntStream.of(idxsort).map(i -> pipelineDelayCapture[i]).toArray();
        }
        totalDelay[0] = pipelineDelay[0] + (issub[0] ? 1 : 0);
        return issub[0] ? -vars[0] : vars[0];
    }    
    public static DFEVar ternaryAdder(DFEVar[] vars, boolean[] issub, int[] pipelineDelay, Integer[] totalDelay, KernelBase<?> base) //O(ceil(log_3 n)) + additional 1 if negation occurs...at 6 additions this routine is more efficient than the default chaining
    {
        int limit = vars.length;
        while (limit > 1) {
            if (pipelineDelay[0] != pipelineDelay[1]) {
                vars[0] = base.optimization.pipeline(vars[0]);
                pipelineDelay[0]++;
            } else if (limit > 2 && pipelineDelay[0] == pipelineDelay[1] && pipelineDelay[1] != pipelineDelay[2]) { //maxcompiler will still infer tri-adds and disregard correct pipelining if doing binary additions before the end...
                vars[0] = base.optimization.pipeline(vars[0]);
                pipelineDelay[0]++;
                vars[1] = base.optimization.pipeline(vars[1]);
                pipelineDelay[1]++;
            } else {
                int limitNew, i;
                for (i = 0, limitNew = 0; i < limit; i++, limitNew++) {
                    if (i == 0 && (limit == 2)) {// || pipelineDelay[0] == pipelineDelay[1] && pipelineDelay[1] != pipelineDelay[2])) {
                        pipelineDelay[0]++;
                        if (!issub[0]) { //first argument must be positive for efficient binary arithmetic
                            List<List<DFEVar>> chunks = getBitChunks128(vars[0], vars[1], issub[1], true);
                            if (chunks != null) pipelineDelay[0] += chunks.size() - 1;
                            vars[0] = addExact(vars[0], vars[1], issub[1], base);
                            issub[0] = false;
                        } else if (!issub[1]) {
                            List<List<DFEVar>> chunks = getBitChunks128(vars[1], vars[0], issub[0], true);
                            if (chunks != null) pipelineDelay[0] += chunks.size() - 1;
                            vars[0] = addExact(vars[1], vars[0], issub[0], base);
                            issub[0] = false;
                        } else {
                            List<List<DFEVar>> chunks = getBitChunks128(vars[0], vars[1], false, true);
                            if (chunks != null) pipelineDelay[0] += chunks.size() - 1;
                            vars[0] = addExact(vars[0], vars[1], false, base);
                            issub[0] = true;
                        }
                        i++;
                    } else if (i >= limit-2 || pipelineDelay[i] != pipelineDelay[i+1] || pipelineDelay[i+1] != pipelineDelay[i+2]) {
                        pipelineDelay[limitNew] = pipelineDelay[i];
                        vars[limitNew] = vars[i];
                        issub[limitNew] = issub[i];                        
                    } else {
                        pipelineDelay[limitNew] = pipelineDelay[i] + 1;                        
                        if (!issub[i]) { //first argument must be positive for efficient TriArith
                            List<List<DFEVar>> chunks = getTriBitChunks128(vars[i], vars[i+1], vars[i+2], issub[i+1], issub[i+2], true);
                            if (chunks != null) pipelineDelay[limitNew] += chunks.size() - 1;
                            vars[limitNew] = triAddExact(vars[i], vars[i+1], vars[i+2], issub[i+1], issub[i+2], base);
                            issub[limitNew] = false;
                        } else if (!issub[i+1]) {
                            List<List<DFEVar>> chunks = getTriBitChunks128(vars[i+1], vars[i], vars[i+2], issub[i], issub[i+2], true);
                            if (chunks != null) pipelineDelay[limitNew] += chunks.size() - 1;
                            vars[limitNew] = triAddExact(vars[i+1], vars[i], vars[i+2], issub[i], issub[i+2], base);
                            issub[limitNew] = false;
                        } else if (!issub[i+2]) {
                            List<List<DFEVar>> chunks = getTriBitChunks128(vars[i+2], vars[i], vars[i+1], issub[i], issub[i+1], true);
                            if (chunks != null) pipelineDelay[limitNew] += chunks.size() - 1;
                            vars[limitNew] = triAddExact(vars[i+2], vars[i], vars[i+1], issub[i], issub[i+1], base);
                            issub[limitNew] = false;
                        } else { //all 3 are negative
                            List<List<DFEVar>> chunks = getTriBitChunks128(vars[i], vars[i+1], vars[i+2], false, false, true);
                            if (chunks != null) pipelineDelay[limitNew] += chunks.size() - 1;
                            vars[limitNew] = triAddExact(vars[i], vars[i+1], vars[i+2], false, false, base);
                            issub[limitNew] = true;
                        }
                        i+=2;
                    }
                }
                limit = limitNew;
            }
            DFEVar[] varsCapture = Arrays.copyOf(vars, limit);
            int[] pipelineDelayCapture = Arrays.copyOf(pipelineDelay, limit);
            boolean[] issubCapture = Arrays.copyOf(issub, limit);
            int[] idxsort = IntStream.range(0, limit).mapToObj(Integer::valueOf).sorted((i, j) -> {
                int c = Integer.compare(pipelineDelayCapture[i], pipelineDelayCapture[j]);
                int s = Integer.compare(((DFEFix)varsCapture[i].getType()).getOffset(), ((DFEFix)varsCapture[j].getType()).getOffset());
                return c != 0 ? c : (s != 0 ? s : Integer.compare(i, j));
                }).mapToInt(Integer::intValue).toArray();
            vars = IntStream.of(idxsort).mapToObj(i -> varsCapture[i]).toArray(DFEVar[]::new);
            issub = booleanAdapter(IntStream.of(idxsort).mapToObj(i -> issubCapture[i]).toArray(Boolean[]::new));
            pipelineDelay = IntStream.of(idxsort).map(i -> pipelineDelayCapture[i]).toArray();
        }
        totalDelay[0] = pipelineDelay[0] + (issub[0] ? 1 : 0);
        return issub[0] ? -vars[0] : vars[0];
    }
    public static Iterable<List<Integer>> getIntegerPartitions(int value)
    {
        return new Iterable<List<Integer>>() {
            public Iterator<List<Integer>> iterator() {
                return new Iterator<List<Integer>>() {
                    class PartStack {
                        int n, I, i;
                        List<Integer> p;
                        PartStack(int n, int I, int i, List<Integer> p) {
                            this.n = n; this.I = I; this.i = i; this.p = p;
                        }
                        public String toString() { return this.n + " " + this.I + " " + this.i + " " + this.p.toString(); }
                    }            
                    List<PartStack> stack = null;
                    public boolean hasNext() { return stack == null || stack.size() != 0; }
                    public List<Integer> next() {
                        List<Integer> nextList = null;
                        if (stack == null) {
                            stack = new ArrayList<>();
                            stack.add(new PartStack(value, 1, 0, new ArrayList<>()));
                        }
                        while (stack.size() != 0) {
                            PartStack ps = stack.get(stack.size()-1);
                            if (ps.i == 0) {
                                if (nextList != null) break;
                                nextList = new ArrayList<>(ps.p);
                                nextList.add(ps.n);
                                ps.i = ps.I;
                                ps.p.add(ps.i);
                            } else ps.p.set(ps.p.size()-1, ps.i);
                            if (ps.i > ps.n/2) {
                                stack.remove(stack.size()-1);
                                if (stack.size() != 0) {
                                    stack.get(stack.size()-1).i++;
                                }
                                continue;
                            }                            
                            stack.add(new PartStack(ps.n-ps.i, ps.i, 0, new ArrayList<>(ps.p)));
                        }
                        return nextList;
                    }
                };
            }
        };
    }
    public static <T> Iterable<List<T>> getPermutations(List<T> start) {
        return new Iterable<List<T>>() {
            public Iterator<List<T>> iterator() {
                return new Iterator<List<T>>() {
                    class PermStack {
                        int i;
                        List<T> lst;
                        List<T> p;
                        PermStack(int i, List<T> lst, List<T> p) {
                            this.i = i; this.lst = lst; this.p = p;
                        }
                        public String toString() { return this.i + " " + this.lst.toString() + " " + this.p.toString(); }
                    }
                    List<PermStack> stack = null;
                    public boolean hasNext() { return start.size() != 0 && (stack == null || stack.size() != 0); }
                    public List<T> next() {
                        List<T> nextList = null;
                        if (stack == null) {
                            stack = new ArrayList<>();
                            if (start.size() == 1) return start;
                            stack.add(new PermStack(-1, start, new ArrayList<>()));
                        }
                        while (stack.size() != 0) {
                            PermStack ps = stack.get(stack.size()-1);
                            if (ps.lst.size() == 0) {
                                if (nextList != null) break;
                                nextList = new ArrayList<>(ps.p);
                                ps.i++;
                            } else if (ps.i == -1) {
                                ps.i++;
                                ps.p.add(ps.lst[ps.i]);
                            }
                            if (ps.i >= ps.lst.size()) {
                                stack.remove(stack.size()-1);
                                if (stack.size() != 0) {
                                    stack.get(stack.size()-1).i++;
                                }
                                continue;
                            }
                            ps.p.set(ps.p.size()-1, ps.lst[ps.i]);                            
                            List<T> remLst = new ArrayList<>(ps.lst.subList(0, ps.i));
                            remLst.addAll(ps.lst.subList(ps.i+1, ps.lst.size()));
                            stack.add(new PermStack(-1, remLst, new ArrayList<>(ps.p)));
                        }
                        return nextList;
                    }
                };
            }
        };    
    }
    /*
def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p
def gen_gpc(m, n): #m>=3 for carry-save-adder e.g. LUT-6 with maximum of 3 outputs is m=6, n=3
    import itertools
    gpcs = []
    for k in range(3, m+1):
        for part in partitions(k):
            for perm in set(itertools.permutations(part)):
                s, outbits = 0, 0
                for x in perm:
                    s = (s + x) >> 1
                    if s == 0: outbits = 0; break #unnecessary additions removed
                    outbits += 1
                outbits += s.bit_length()
                if outbits != 0 and outbits <= n: gpcs.append((perm, outbits))
    return list(sorted(gpcs, key=lambda x: (sum(x[0])/x[1], x[0][0]), reverse=True))
def gpc_to_lut(gpc):
    l, luts = [], []
    for i in range(gpc[1]):
        if i < len(gpc[0]): l.extend([1<<i]*gpc[0][i])
        luts.append([(sum(l[q] for q in range(len(l)) if ((1<<q) & b) != 0)>>i) & 1 for b in range(1 << len(l))])
    return luts

print(gen_gpc(6, 3))
print([gpc_to_lut(x) for x in gen_gpc(6, 3)])
    */
    //https://www.epfl.ch/labs/lap/wp-content/uploads/2018/05/ParandehAfsharJan08_EfficientSynthesisOfCompressorTreesOnFpgas_ASPDAC08.pdf
    public static DFEVar compressionTree(DFEVar[] vars, boolean[] issub, int[] pipelineDelay, Integer[] totalDelay, KernelBase<?> base) {
        final int lut_size = 6, max_out_size = 3, const_lut_size = 7 + 1; //for constants we extend to 7 (8 always needs 4 output bits)
        List<Pair<List<Integer>, Integer>> gpcs = new ArrayList<>();
        for (int k = 3; k <= const_lut_size; k++) {
            for (List<Integer> part : getIntegerPartitions(k)) {
                HashSet<List<Integer>> perms = new HashSet<>();
                getPermutations(part).forEach(perms::add);
                for (List<Integer> perm : perms) {
                    int s = 0, outbits = 0;
                    for (Integer x : perm) {
                        s = (s + x) >> 1;
                        if (s == 0) { outbits = 0; break; } //unnecessary additions removed
                        outbits++;
                    }
                    outbits += Integer.highestOneBit(s);
                    if (outbits != 0 && outbits <= max_out_size) gpcs.add(new Pair<List<Integer>, Integer>(perm, outbits));
                }
            }
        }
        gpcs.sort((a, b) -> {
            double aCompress = a.first.stream().mapToInt(x -> x).sum()*b.second;
            double bCompress = b.first.stream().mapToInt(x -> x).sum()*a.second;
            if (aCompress == bCompress) {
                for (int i = 0; i < Math.max(a.first.size(), b.first.size()); i++) {
                    if (i >= a.first.size()) return 1;
                    if (i >= b.first.size()) return -1;
                    if (a.first.get(i) == b.first.get(i)) continue;
                    return a.first.get(i) < b.first.get(i) ? 1 : -1; 
                }
                return 0;
            }
            return aCompress < bCompress ? 1 : -1;
            });
        System.out.println(gpcs);
        BiFunction<Pair<List<Integer>, Integer>, List<Integer>, List<List<Integer>>> getBitmaps = (x, y) -> {
            List<Integer> l = new ArrayList<>();
            List<List<Integer>> luts = new ArrayList<>();
            int curidx = 0, constmask = 0;
            for (int i = 0; i < x.second; i++) {
                if (i < x.first.size()) {
                    for (int j = 0; j < x.first.get(i); j++) {
                        if (curidx == y.size()) return luts; 
                        l.add(1<<i);
                        if (y.get(curidx) == 2) constmask |= 1<<curidx;
                        curidx++;
                    }
                }
                final int fi = i, fconstmask = constmask;
                luts.add(IntStream.range(0, 1<<l.size()).filter(b -> (b & fconstmask) == fconstmask).map(b -> (IntStream.range(0, l.size()).filter(q -> y.get(q) == 1 ^ ((1<<q) & b) != 0).map(q -> l.get(q)).sum() >> fi) & 1).boxed().collect(Collectors.toList()));
            }
            return luts;
        };
        int idx = 0;
        int curDelay = pipelineDelay[idx];
        HashMap<Integer, List<Pair<Boolean, DFEVar>>> columns = new HashMap<>();
        int maxcol = Arrays.stream(vars).mapToInt(x -> -((DFEFix)x.getType()).getFractionBits()+x.getType().getTotalBits()).max().getAsInt();
        boolean isUnsigned = true;
        while (true) {
            while (idx != pipelineDelay.length && pipelineDelay[idx] == curDelay) {
                int offset = -((DFEFix)vars[idx].getType()).getFractionBits();
                for (int i = 0; i < vars[idx].getType().getTotalBits(); i++) { //unpack bits
                    if (!columns.containsKey(offset+i)) columns.put(offset+i, new ArrayList<>());
                    columns.get(offset+i).add(new Pair<Boolean, DFEVar>(issub[idx], vars[idx].get(i)));
                }
                if (((DFEFix)vars[idx].getType()).getSignMode() == SignMode.TWOSCOMPLEMENT) { //sign extension
                    isUnsigned = false;
                    for (int i = vars[idx].getType().getTotalBits(); i < maxcol - offset; i++) {
                        if (!columns.containsKey(offset+i)) columns.put(offset+i, new ArrayList<>());
                        columns.get(offset+i).add(new Pair<Boolean, DFEVar>(issub[idx], vars[idx].get(vars[idx].getType().getTotalBits()-1)));
                    }
                }
                if (issub[idx]) columns.get(offset).add(new Pair<Boolean, DFEVar>(false, null)); //null represents a constant one
                idx++;
            }
            List<Integer> sortedTallCols = columns.entrySet().stream().sorted((x, y) -> Integer.compare(y.getValue().size(), x.getValue().size())).map(x -> x.getKey()).collect(Collectors.toList());
            if (columns.get(sortedTallCols.get(0)).size() <= 3) { //repack bits for ternary addition
                totalDelay[0] = curDelay;
                List<List<DFEVar>> toSum = new ArrayList<>();
                for (int i = 0; i < columns.get(sortedTallCols.get(0)).size(); i++) toSum.add(new ArrayList<>());
                int startcol = columns.keySet().stream().mapToInt(x -> x).min().getAsInt();
                for (int i = startcol; i < maxcol; i++) {
                    for (int j = 0; j < toSum.size(); j++) {
                        if (j < columns.get(i).size()) {
                            if (columns.get(i).get(j).first) throw new IllegalArgumentException(); //subtraction inversions require one stage at least, this case can easily be handling by pre-detection and just using a ternary adder
                            toSum.get(j).add(columns.get(i).get(j).second == null ? base.constant.var(KernelBase.dfeBool(), 1) : columns.get(i).get(j).second);
                        } else {
                            toSum.get(j).add(base.constant.zero(KernelBase.dfeBool()));
                        }
                    }
                }                
                if (toSum.size() == 1) return Bitops.catLsbToMsb(toSum.get(0)).reinterpret(KernelBase.dfeFixOffset(maxcol - startcol, -startcol, isUnsigned ? SignMode.UNSIGNED : SignMode.TWOSCOMPLEMENT));
                else if (toSum.size() == 2) {
                    DFEVar add1 = Bitops.catLsbToMsb(toSum.get(0)).reinterpret(KernelBase.dfeFixOffset(maxcol - startcol, -startcol, isUnsigned ? SignMode.UNSIGNED : SignMode.TWOSCOMPLEMENT));
                    DFEVar add2 = Bitops.catLsbToMsb(toSum.get(1)).reinterpret(KernelBase.dfeFixOffset(maxcol - startcol, -startcol, isUnsigned ? SignMode.UNSIGNED : SignMode.TWOSCOMPLEMENT));
                    List<List<DFEVar>> chunks = getBitChunks128(add1, add2, false, true);
                    if (chunks != null) totalDelay[0] += chunks.size() - 1;
                    totalDelay[0]++;
                    return addExact(add1, add2, false, base);                    
                } else if (toSum.size() == 3) {
                    DFEVar add1 = Bitops.catLsbToMsb(toSum.get(0)).reinterpret(KernelBase.dfeFixOffset(maxcol - startcol, -startcol, isUnsigned ? SignMode.UNSIGNED : SignMode.TWOSCOMPLEMENT));
                    DFEVar add2 = Bitops.catLsbToMsb(toSum.get(1)).reinterpret(KernelBase.dfeFixOffset(maxcol - startcol, -startcol, isUnsigned ? SignMode.UNSIGNED : SignMode.TWOSCOMPLEMENT));
                    DFEVar add3 = Bitops.catLsbToMsb(toSum.get(2)).reinterpret(KernelBase.dfeFixOffset(maxcol - startcol, -startcol, isUnsigned ? SignMode.UNSIGNED : SignMode.TWOSCOMPLEMENT));
                    List<List<DFEVar>> chunks = getTriBitChunks128(add1, add2, add3, false, false, true);
                    if (chunks != null) totalDelay[0] += chunks.size() - 1;
                    totalDelay[0]++;
                    return triAddExact(add1, add2, add3, false, false, base);                    
                }
                return null; //toSum.size() == 0 implies invalid input anyway
            }
            HashMap<Integer, List<Pair<Boolean, DFEVar>>> nextcols = new HashMap<>();
            while (true) {
                boolean changed = false;
                for (int tallCol : sortedTallCols) {
                    for (Pair<List<Integer>, Integer> gpc : gpcs) {
                        boolean found = true;
                        int constAvail = 0;
                        for (int j = 0; j < gpc.first.size(); j++) {
                            if (!columns.containsKey(tallCol+j) || columns.get(tallCol+j).size() < gpc.first.get(j)) { found = false; break; }
                            constAvail += Math.min(gpc.first.get(j), columns.get(tallCol+j).stream().filter(x -> x.second == null).count());
                        }
                        int needConsts = gpc.first.stream().mapToInt(x -> x).sum() - lut_size;
                        if (needConsts > constAvail) found = false;
                        if (found) {
                            List<DFEVar> curvars = new ArrayList<>();
                            List<Integer> vartypes = new ArrayList<>();
                            for (int j = 0; j < gpc.second; j++) {
                                if (j < gpc.first.size()) {
                                    for (Pair<Boolean, DFEVar> entry : columns.get(tallCol+j).stream().sorted((x, y) -> Boolean.compare(x.second != null, y.second != null)).limit(gpc.first.get(j)).collect(Collectors.toList())) {
                                        if (entry.second == null) {
                                            vartypes.add(2);
                                        } else {
                                            vartypes.add(entry.first ? 1 : 0);
                                            curvars.add(entry.second);
                                        }
                                        columns.get(tallCol+j).remove(entry);
                                    }
                                }
                                if (tallCol + j >= maxcol) continue;
                                if (!nextcols.containsKey(tallCol+j)) nextcols.put(tallCol+j, new ArrayList<>());
                                nextcols.get(tallCol+j).add(new Pair<Boolean, DFEVar>(false, base.control.mux(Bitops.catLsbToMsb(curvars),
                                    getBitmaps.apply(gpc, vartypes).get(j).stream().map(x -> base.constant.var(KernelBase.dfeBool(), x)).collect(Collectors.toList()))));
                            }
                            changed = true;
                            break;
                        }
                    }
                    if (changed) break;
                }
                if (!changed) break;
                sortedTallCols = columns.entrySet().stream().sorted((x, y) -> Integer.compare(y.getValue().size(), x.getValue().size())).map(x -> x.getKey()).collect(Collectors.toList());
            }
            System.out.println("Total Cols MUXed: " + nextcols.size() + " Total MUX: " + nextcols.values().stream().mapToInt(x -> x.size()).sum());
            for (HashMap.Entry<Integer, List<Pair<Boolean, DFEVar>>> entry : columns.entrySet()) {
                if (!nextcols.containsKey(entry.getKey()) && entry.getValue().size() != 0) nextcols.put(entry.getKey(), new ArrayList<>());
                for (int i = 0; i < entry.getValue().size(); i++) {
                    if (entry.getValue().get(i).second == null ) {
                        nextcols.get(entry.getKey()).add(new Pair<Boolean, DFEVar>(false, null));
                    } else if (entry.getValue().get(i).first) {
                        nextcols.get(entry.getKey()).add(new Pair<Boolean, DFEVar>(false, ~entry.getValue().get(i).second));
                    } else {
                        nextcols.get(entry.getKey()).add(new Pair<Boolean, DFEVar>(false, base.optimization.pipeline(entry.getValue().get(i).second)));
                    }
                }
            }            
            columns = nextcols;
            curDelay++;
        }        
    }
    public static boolean[] booleanAdapter(Boolean[] l)
    {
        boolean[] res = new boolean[l.length];
        for (int i = 0; i < l.length; i++) res[i] = l[i];
        return res;
    }
    public static Object[] addPostMultiply(DFEVar[] vars, int[] shift, boolean[] addsub, int[] pipelineDelay, KernelBase<?> base) {
        int[] idxsort = IntStream.range(0, vars.length).mapToObj(Integer::valueOf).sorted((i, j) -> {
            int c = Integer.compare(pipelineDelay[i], pipelineDelay[j]);
            int s = Integer.compare(shift[i], shift[j]);
            return c != 0 ? c : (s != 0 ? s : Integer.compare(i, j));
            }).mapToInt(Integer::intValue).toArray();
        int[] orderedShifts = IntStream.of(idxsort).map(i -> shift[i]).toArray();
        DFEVar[] orderedVars = IntStream.of(idxsort).mapToObj(i -> vars[i]).toArray(DFEVar[]::new);
        boolean[] issub = addsub == null ? new boolean[vars.length] : booleanAdapter(IntStream.of(idxsort).mapToObj(i -> addsub[i]).toArray(Boolean[]::new));
        int[] orderedPipelineDelay = pipelineDelay == null ? new int[vars.length] : IntStream.of(idxsort).map(i -> pipelineDelay[i]).toArray();
        List<DFEVar> c = new ArrayList<>();
        List<Integer> newshift = new ArrayList<>();
        List<Boolean> newaddsub = new ArrayList<>();
        List<Integer> newPipelineDelay = new ArrayList<>();
        for (int i = 0; i < orderedVars.length-1; i++) {
            if (orderedPipelineDelay[i] == orderedPipelineDelay[i+1]) {
                if (orderedShifts[i] == orderedShifts[i+1] && orderedVars[i].getType().getTotalBits() == orderedVars[i+1].getType().getTotalBits()) {
                    System.out.println(orderedVars[i] + " " + orderedShifts[i]);
                    if (!issub[i]) {
                        c.add(addExact(orderedVars[i], orderedVars[i+1], issub[i+1], base));
                        newaddsub.add(false);
                    } else if (!issub[i+1]) {
                        c.add(addExact(orderedVars[i+1], orderedVars[i], issub[i], base));
                        newaddsub.add(false);
                    } else {
                        c.add(addExact(orderedVars[i], orderedVars[i+1], false, base));
                        newaddsub.add(true);
                    }
                    newPipelineDelay.add(orderedPipelineDelay[i]+1);
                    newshift.add(orderedShifts[i]);
                    orderedVars[i] = null; orderedVars[i+1] = null; i++;             
                }
            }
        }
        for (int i = 0; i < orderedVars.length; i++) {
            if (orderedVars[i] != null) {
                c.add(orderedVars[i]);
                newshift.add(orderedShifts[i]);
                newaddsub.add(issub[i]);
                newPipelineDelay.add(orderedPipelineDelay[i]);                
            }
        }
        return new Object[] { c.stream().toArray(DFEVar[]::new), newshift.stream().mapToInt(Integer::valueOf).toArray(),
            booleanAdapter(newaddsub.stream().toArray(Boolean[]::new)), newPipelineDelay.stream().mapToInt(Integer::valueOf).toArray() };
    }
    public static DFEVar addShifter(DFEVar[] vars, int[] shift, boolean[] addsub, int[] pipelineDelay, Integer[] totalDelay, boolean useCompressor, KernelBase<?> base) {
        int[] idxsort = IntStream.range(0, vars.length).mapToObj(Integer::valueOf).sorted((i, j) -> {
            int c = Integer.compare(pipelineDelay[i], pipelineDelay[j]);
            int s = Integer.compare(shift[i], shift[j]);
            return c != 0 ? c : (s != 0 ? s : Integer.compare(i, j));
            }).mapToInt(Integer::intValue).toArray();
        DFEVar[] orderedVars = IntStream.of(idxsort).mapToObj(i -> shiftLeftFix(vars[i], shift[i], base)).toArray(DFEVar[]::new);
        boolean[] issub = addsub == null ? new boolean[vars.length] : booleanAdapter(IntStream.of(idxsort).mapToObj(i -> addsub[i]).toArray(Boolean[]::new));
        int[] orderedPipelineDelay = pipelineDelay == null ? new int[vars.length] : IntStream.of(idxsort).map(i -> pipelineDelay[i]).toArray();
        
        if (useCompressor) {
            return compressionTree(orderedVars, issub, orderedPipelineDelay, totalDelay, base);
        } else {
            return ternaryAdder(orderedVars, issub, orderedPipelineDelay, totalDelay, base);
            //return binaryAdder(orderedVars, issub, orderedPipelineDelay, totalDelay, base);
        }
    }
    public static int getIntSplits(DFEFix aType, int bits, int signbits, KernelBase<?> base)
    {
        int num = aType.getSignMode() == SignMode.UNSIGNED ? (aType.getTotalBits()+bits-1)/bits : 1+(aType.getTotalBits()-signbits+bits-1)/bits;
        if (aType.getSignMode() == SignMode.TWOSCOMPLEMENT && ((aType.getTotalBits()-signbits)%bits == 1) && bits == signbits) num--; //cannot have sign bit isolated
        return num;
    }
    public static DFEVar[] splitInt(DFEVar a, int bits, int signbits, KernelBase<?> base)
    {
        DFEFix aType = (DFEFix) a.getType();        
        int num = getIntSplits(aType, bits, signbits, base);
        DFEVar[] vars = new DFEVar[num];
        for (int i = 0; i < num; i++) {
            vars[i] = a.slice(i * bits, i == num - 1 ? aType.getTotalBits() - i * bits : bits)
                .reinterpret(KernelBase.dfeFixOffset(i == num - 1 ? aType.getTotalBits() - i * bits : bits, 0, i == num - 1 ? aType.getSignMode() : SignMode.UNSIGNED));
        }
        return vars;
    }    
    public static int gcd(int x, int y) {
        return y == 0 ? x : gcd(y, x % y);
    }
    public static DFEVar bigCast(DFEVar a, DFEFix targetType, KernelBase<?> base)
    {
        DFEFix aType = (DFEFix) a.getType();
        if (aType.getTotalBits() <= 128) return a.cast(targetType);
        int guardBit = aType.getFractionBits() - targetType.getFractionBits() + 1; //guard bit is one past the round bit, we let the sticky handle with low casting 
        DFEVar lowCast = a.slice(0, guardBit).reinterpret(KernelBase.dfeFixOffset(guardBit, -guardBit+1, SignMode.UNSIGNED)).cast(KernelBase.dfeUInt(2));
        DFEVar s = base.optimization.pipeline(a.slice(guardBit, aType.getTotalBits() - guardBit).reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits() - guardBit, 0, aType.getSignMode())));
        DFEVar c = lowCast.get(1).reinterpret(KernelBase.dfeBool());
        List<List<DFEVar>> chunks = getBitChunks128(s, c, false, true);
        DFEVar result = addExact(s, c, false, base).cat(repeatPipeline(lowCast.get(0), 1 + (chunks != null ? chunks.size() - 1 : 0), base))
            .reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits() - guardBit + 1 + 1, -targetType.getFractionBits(), aType.getSignMode())).cast(targetType);
        //base.debug.simPrintf(result !== a.cast(targetType), "%f %f %f\n", a, a.cast(targetType), result);
        return result;
    }
    public static DFEVar mulKaratsubaRectangularExact(DFEVar a, DFEVar b, int bits, int intBits, boolean useCompressor, KernelBase<?> base)
    {
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType();
        //if (aType.getTotalBits() > 100) return mulExact(a, b, bits, intBits, base);
        Integer[] totalDelay = new Integer[1];
        DFEVar result = mulKaratsubaRectangular(a.reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits(), 0, aType.getSignMode())),
                                        b.reinterpret(KernelBase.dfeFixOffset(bType.getTotalBits(), 0, bType.getSignMode())), 0, totalDelay, useCompressor, base); //aType.getTotalBits() <= 112 && bType.getTotalBits() <= 120 ? 0 : 2
        result = result
            .reinterpret(KernelBase.dfeFixOffset(getMulOutpBits(aType, bType), -(aType.getFractionBits() + bType.getFractionBits()), ((DFEFix)result.getType()).getSignMode()));
        result = bigCast(result, KernelBase.dfeFixOffset(bits, -bits+intBits, ((DFEFix)result.getType()).getSignMode()), base);
        //result = result.cast(KernelBase.dfeFixOffset(bits, -bits+intBits, ((DFEFix)result.getType()).getSignMode()));
        DFEVar verify = mulExact(a, b, bits, intBits, base);
        base.debug.simPrintf(result !== verify, "%d %X %X %X %X\n", aType.getTotalBits(), result.reinterpret(KernelBase.dfeUInt(result.getType().getTotalBits())), verify.reinterpret(KernelBase.dfeUInt(result.getType().getTotalBits())), a.reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits(), 0, aType.getSignMode())), b.reinterpret(KernelBase.dfeFixOffset(bType.getTotalBits(), 0, bType.getSignMode()))); //mulKaratsubaExact(a, b, bits, intBits).reinterpret(KernelBase.dfeUInt(result.getType().getTotalBits())),
        return result;
    }
    public static DFEVar signedMulKaratsubaRectangular(DFEVar a, DFEVar b, int tileMode, Integer[] totalDelay, KernelBase<?> base)
    {
        DFEVar aSign = getSignBit(a, base), bSign = getSignBit(b, base);
        DFEVar resSign = aSign ^ bSign;
        DFEVar result = uToSigned(mulKaratsubaRectangular(sToUnsigned(aSign ? -a : a), sToUnsigned(bSign ? -b : b), tileMode, totalDelay, false, base), base);
        for (int i = 0; i < totalDelay[0] + 1 + (tileMode == 2 ? 1 : 0); i++) resSign = base.optimization.pipeline(resSign);
        totalDelay[0] += 2;
        return resSign ? -result : result;
    }
    public static int getMulInpBits(DFEFix type) { return type.getTotalBits() + (type.getSignMode() == SignMode.UNSIGNED ? 1 : 0); } //effective bits for a signed multiplier
    public static int getMulOutpBits(DFEFix aType, DFEFix bType) { //actual output bits, save one sign bit on 2 argument signed multiplication
        if (aType.getTotalBits() == 1) return bType.getTotalBits();
        if (bType.getTotalBits() == 1) return aType.getTotalBits();
        return aType.getTotalBits() + bType.getTotalBits(); //the smallest signed times smallest signed is a special case making no optimization possible 
            //- ((aType.getSignMode() == SignMode.TWOSCOMPLEMENT && bType.getSignMode() == SignMode.TWOSCOMPLEMENT) ? 1 : 0);
    }
    public static void latencyDump(DFEVar a, DFEVar b, KernelBase<?> base)
    {
        final int DSP_X = 18, DSP_Y = 27; //DSP_Y = 25;
        for (int is = 0; is <= 1; is++) {
            for (int i = (is == 1 ? 2 : 1); i <= (is == 1 ? DSP_X : DSP_X-1); i++) {
                for (int js = 0; js <= 1; js++) {
                    for (int j = (js == 1 ? 2 : 1); j <= (js == 1 ? DSP_Y : DSP_Y-1); j++) {
                        base.debug.simPrintf("%X\n", mulExact(a.slice(0, i).reinterpret(KernelBase.dfeFixOffset(i, 0, is == 1 ? SignMode.TWOSCOMPLEMENT : SignMode.UNSIGNED)),
                                                b.slice(0, j).reinterpret(KernelBase.dfeFixOffset(j, 0, js == 1 ? SignMode.TWOSCOMPLEMENT : SignMode.UNSIGNED)), base));
                    }
                }
            }
        }
    } 
    public static int getMulPipelineDelay(DFEFix aType, DFEFix bType) //only for sizes up to 18x27 (DSP48E2) 18x25 (DSP48E1) signed/unsigned 
    {
        if ((aType.getTotalBits() == 1 || bType.getTotalBits() == 1) && aType.getSignMode() == SignMode.UNSIGNED && bType.getSignMode() == SignMode.UNSIGNED) return 1;
        //if (aType.getTotalBits() <= 10 || bType.getTotalBits() <= 10) return 3;         
        return ((getMulInpBits(aType) >= (aType.getSignMode() == SignMode.UNSIGNED ? 5 : 6) &&
                getMulInpBits(bType) >= (bType.getSignMode() == SignMode.UNSIGNED ? 22 : 23) ||
                getMulInpBits(bType) >= (bType.getSignMode() == SignMode.UNSIGNED ? 5 : 6) &&
                getMulInpBits(aType) >= (aType.getSignMode() == SignMode.UNSIGNED ? 22 : 23)) &&
                (aType.getSignMode() == SignMode.TWOSCOMPLEMENT || bType.getSignMode() == SignMode.TWOSCOMPLEMENT)) ? 4 : 3;
    }
    //https://hal.inria.fr/hal-01773447/document Karatsuba with Rectangular Multipliers for FPGAs
    public static Object[] mulKaratsubaRectangularSums(DFEVar a, DFEVar b, int tileMode, boolean useCompressor, KernelBase<?> base) //tileMode is 0 for 16x24, 1 for 17x24, 2 for 18x24
    {
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType();
        if (aType.getTotalBits() > bType.getTotalBits()) return mulKaratsubaRectangularSums(b, a, tileMode, useCompressor, base);
        int tileX = tileMode + 16, tileY = 24;
        int W = gcd(tileX, tileY);
        int M = tileX / W;
        int N = tileY / W;
        final int DSP_X = 18, DSP_Y = 27; //DSP_Y = 25;
        /*int MNWa = M * N * W;
        int MNWb1 = (N + 1) * M * W, MNWb2 = (M + 1) * N * W; 
        int MNWc = 2 * MNWa; 
        int MNWd1 = (2*N + 1) * M * W, MNWd2 = (2 * M + 1) * N * W;
        if (aType.getTotalBits()-(aType.getSignMode() == SignMode.UNSIGNED ? 0 : 1) <= MNWa && bType.getTotalBits()-(bType.getSignMode() == SignMode.UNSIGNED ? 0 : 1) <= MNWa) { //16x24 tiling, 48x48        
        } else if (aType.getTotalBits()-(aType.getSignMode() == SignMode.UNSIGNED ? 0 : 1) <= MNWb1 && bType.getTotalBits()-(bType.getSignMode() == SignMode.UNSIGNED ? 0 : 1) <= MNWb2) { //16x24 tiling, 64x72
        } else if (aType.getTotalBits() <= MNWc && bType.getTotalBits() <= MNWc || //16x24 tiling, 96x96 breaks down to 3 48x48 multiplications, 95x95 signed...
                   aType.getTotalBits() <= MNWd1 && bType.getTotalBits() <= MNWd2) { //16x24 tiling, 112x120 breaks down to 48x48, 2 64x64 multiplications, 111x119 signed...
             //17x24 tiling does not work until 425x432...so use 18x24 technique with one line done using logic resources, 132x132
            int splitPoint = W * M * N - (aType.getSignMode() == SignMode.TWOSCOMPLEMENT && bType.getSignMode() == SignMode.TWOSCOMPLEMENT ? (tileMode == 2 ? 2 : 1) : 0);
            DFEVar[] aSplit = new DFEVar[] { a.slice(0, splitPoint).reinterpret(KernelBase.dfeUInt(splitPoint)), a.slice(splitPoint, aType.getTotalBits() - splitPoint).reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits() - splitPoint, 0, aType.getSignMode())) };
            DFEVar[] bSplit = new DFEVar[] { b.slice(0, splitPoint).reinterpret(KernelBase.dfeUInt(splitPoint)), b.slice(splitPoint, bType.getTotalBits() - splitPoint).reinterpret(KernelBase.dfeFixOffset(bType.getTotalBits() - splitPoint, 0, bType.getSignMode())) };
            Integer[] delay0 = new Integer[1], delay1 = new Integer[1], delay2 = new Integer[1];
            DFEVar z0 = mulKaratsubaRectangular(aSplit[0], bSplit[0], tileMode, delay0, useCompressor, base);
            DFEVar z1 = signedMulKaratsubaRectangular(addExact(aSplit[0], aSplit[1], true, base), addExact(bSplit[1], bSplit[0], true, base), tileMode, delay1, useCompressor, base); //take care for the mistake in the paper on subtracting high-low vs low-high
            DFEVar z2 = mulKaratsubaRectangular(aSplit[1], bSplit[1], tileMode, delay2, useCompressor, base);
            return new Object[] { new DFEVar[] { z0, z0, z2, z1, z2, }, new int[] { 0, splitPoint, splitPoint, splitPoint, 2 * splitPoint },
                null, new int[] { delay0[0], delay0[0], delay2[0], delay1[0]+1, delay2[0] } };
        }*/
        DFEVar[] aSplit = splitInt(a, tileX, DSP_X, base);
        DFEVar[] bSplit = splitInt(b, tileY, DSP_Y, base);

        HashMap<Integer, List<Object>> sums = new HashMap<>();
        HashSet<Pair<Integer, Integer>> subs = new HashSet<>();
        HashSet<Object> prods = new HashSet<>(); 
        for (int i = 0; i < aSplit.length; i++) { //rectangular tiled grade school multiplication
            for (int j = 0; j < bSplit.length; j++) {
                if (!sums.containsKey(i * M + j * N)) sums.put(i * M + j * N, new ArrayList<Object>());
                sums.get(i * M + j * N).add(new Pair<Integer, Integer>(i, j));
            }
        }
        for (Integer key : sums.keySet()) {
            List<Object> l = sums.get(key);
            if (l == null || l.size() == 2) continue;
            for (int i = 0; i < l.size(); i++)
                prods.add(l[i]);
        }
        for (Integer key : sums.keySet()) {
            List<Object> l = sums.get(key);
            if (l == null || l.size() != 2) continue;
            Object o1 = l[0];
            Object o2 = l[1];
            if (o1 instanceof Pair<?, ?> && o2 instanceof Pair<?, ?>) {
                Pair<?, ?> left = (Pair<?, ?>)o1;
                Pair<?, ?> right = (Pair<?, ?>)o2;
                if (prods.contains(left)) {
                    prods.add(right); continue;
                } else if (prods.contains(right)) {
                    prods.add(left); continue;
                }
                Object l1 = left.getFirst(), l2 = left.getSecond();
                Object r1 = right.getFirst(), r2 = right.getSecond();
                if (l1 instanceof Integer && l2 instanceof Integer && r1 instanceof Integer && r2 instanceof Integer) {
                    Integer il1 = (Integer)l1, il2 = (Integer)l2;
                    Integer ir1 = (Integer)r1, ir2 = (Integer)r2;
                    Pair<Integer, Integer> sub1 = new Pair<Integer, Integer>(il1, ir1);
                    Pair<Integer, Integer> sub2 = new Pair<Integer, Integer>(-il2, -ir2);
                    subs.add(sub1);
                    subs.add(sub2);
                    l.clear();
                    Pair<Pair<Integer, Integer>, Pair<Integer, Integer>> subprod = new Pair<Pair<Integer, Integer>, Pair<Integer, Integer>>(sub1, sub2);
                    l.add(subprod);
                    prods.add(subprod);
                    Pair<Integer, Integer> prod1 = new Pair<Integer, Integer>(il1, ir2);
                    Pair<Integer, Integer> prod2 = new Pair<Integer, Integer>(ir1, il2);
                    prods.add(prod1);
                    prods.add(prod2);
                    l.add(prod1);
                    l.add(prod2);
                } else throw new IllegalArgumentException();
            } else throw new IllegalArgumentException();
        }
        long rightSubs = subs.stream().filter(o -> o.getFirst() < 0 || o.getSecond() < 0).count();
        long prodcount = prods.stream().filter(o -> o instanceof Pair<?, ?> && !(((Pair<?, ?>)o).getFirst() instanceof Pair<?, ?>)).count();
        //System.out.println(rightSubs + " " + prodcount + " " + aSplit[aSplit.length-1].getType() + " " + bSplit[bSplit.length-1].getType());
        if (rightSubs*2 + prodcount > 32 || (subs.size()-rightSubs)*2+prodcount > 32) { //we are not accounting for 18-bit left and 26,28-bit right  
            for (int i = 0; i < aSplit.length; i++) aSplit[i] = base.optimization.pipeline(aSplit[i]);
            for (int i = 0; i < bSplit.length; i++) bSplit[i] = base.optimization.pipeline(bSplit[i]);
        }
        
        HashMap<Pair<Integer, Integer>, DFEVar> subMap = new HashMap<>();
        HashMap<Object, DFEVar> prodMap = new HashMap<>();
        HashMap<Object, DFEVar> extraProdMap = new HashMap<>(); //for 18 bit unsigned case
        HashMap<Object, Integer> prodMapDelay = new HashMap<>();
        for (Pair<Integer, Integer> sub : subs) {
            if (sub.getFirst() < 0 || sub.getSecond() < 0) { //for pre-adder inference use the larger b argument, convert to signed for maxcompiler recognition
                subMap.put(sub, addExact(bSplit[-sub.getFirst()], bSplit[-sub.getSecond()], true, base));
                //subMap.put(sub, addExact(bSplit[-sub.getFirst()], bSplit[-sub.getSecond()], true, base));
            } else {
                subMap.put(sub, addExact(aSplit[sub.getFirst()], aSplit[sub.getSecond()], true, base));
            }
            
        }        
        for (Object o : prods) {
            DFEVar left = null, right = null;
            if (o instanceof Pair<?, ?>) {
                Pair<?, ?> p = (Pair<?, ?>)o;
                Object ol = p.getFirst();
                Object or = p.getSecond();
                if (ol instanceof Pair<?, ?> && or instanceof Pair<?, ?>) {
                    Pair<?, ?> p1 = (Pair<?, ?>)ol;
                    Object o1 = p1.getFirst();
                    Object o2 = p1.getSecond();
                    if (o1 instanceof Integer && o2 instanceof Integer) {
                        left = subMap.get(new Pair<Integer, Integer>((Integer)o1, (Integer)o2));
                    } else throw new IllegalArgumentException();
                    Pair<?, ?> p2 = (Pair<?, ?>)or;
                    o1 = p2.getFirst();
                    o2 = p2.getSecond();
                    if (o1 instanceof Integer && o2 instanceof Integer) {
                        right = subMap.get(new Pair<Integer, Integer>((Integer)o1, (Integer)o2));                         
                    } else throw new IllegalArgumentException();
                } else if (ol instanceof Integer && or instanceof Integer) {
                    left = aSplit[(Integer)ol];
                    right = bSplit[(Integer)or];                    
                } else throw new IllegalArgumentException();
            } else throw new IllegalArgumentException();
            //if (tileMode == 2) System.out.println(left.getType().getTotalBits() + " " + right.getType().getTotalBits());
            if (left.getType().getTotalBits() == DSP_X && ((DFEFix)left.getType()).getSignMode() == SignMode.UNSIGNED && right.getType().getTotalBits() > DSP_X-1) {                
                extraProdMap.put(o, left.get(0) ? right : base.constant.zero(KernelBase.dfeFixOffset(right.getType().getTotalBits(), 0, ((DFEFix)right.getType()).getSignMode())));
                left = left.slice(1, DSP_X-1).reinterpret(KernelBase.dfeUInt(DSP_X-1));
            } else if (right.getType().getTotalBits() == DSP_Y+1 && ((DFEFix)right.getType()).getSignMode() == SignMode.TWOSCOMPLEMENT) {
                extraProdMap.put(o, right.get(0) ? left : base.constant.zero(KernelBase.dfeFixOffset(left.getType().getTotalBits(), 0, ((DFEFix)left.getType()).getSignMode())));
                right = right.slice(1, DSP_Y).reinterpret(KernelBase.dfeInt(DSP_Y));
            }
            if (!(getMulInpBits((DFEFix)left.getType()) <= DSP_X && getMulInpBits((DFEFix)right.getType()) <= DSP_Y)) throw new IllegalArgumentException("Size: " + aType.getTotalBits() + " " + bType.getTotalBits() + " " + left.getType() + " " + right.getType());
            if (left.getType().getTotalBits() == 1) { //must be unsigned...
                if (((DFEFix)left.getType()).getSignMode() != SignMode.UNSIGNED) throw new IllegalArgumentException();
                prodMap.put(o, left ? right : base.constant.zero(KernelBase.dfeFixOffset(right.getType().getTotalBits(), 0, ((DFEFix)right.getType()).getSignMode())));
                prodMapDelay.put(o, 1);
            } else if (right.getType().getTotalBits() == 1) { //must be unsigned...
                if (((DFEFix)right.getType()).getSignMode() != SignMode.UNSIGNED) throw new IllegalArgumentException();
                prodMap.put(o, right ? left : base.constant.zero(KernelBase.dfeFixOffset(left.getType().getTotalBits(), 0, ((DFEFix)left.getType()).getSignMode())));
                prodMapDelay.put(o, 1);
            } else {
                //for MaxCompiler to recognize DSPAddMultiplyAdd/DSPMultiplyAdd, must use signed datatypes but post-adders not usable with Karatsuba without a cascade rather than a tree
                //if (extraProdMap.containsKey(o)) {
                //    if (((DFEFix)left.getType()).getSignMode() == SignMode.UNSIGNED) left = uToSigned(left, base);
                //    if (((DFEFix)right.getType()).getSignMode() == SignMode.UNSIGNED) right = uToSigned(right, base);
                //}
                int delay = getMulPipelineDelay((DFEFix)left.getType(), (DFEFix)right.getType());
                if (delay == 4) base.optimization.pushNodePipelining(0.75, PipelinedOps.FIX_MUL);
                //base.optimization.pushRoundingMode(RoundingMode.TONEAR);
                prodMap.put(o, mulExact(left, right, base));
                //base.optimization.popRoundingMode();
                if (delay == 4) base.optimization.popNodePipelining(PipelinedOps.FIX_MUL);
                prodMapDelay.put(o, delay - (delay == 4 ? 1 : 0));
            }
        }
        List<DFEVar> c = new ArrayList<DFEVar>();
        List<Integer> shift = new ArrayList<Integer>();
        List<Integer> pipelineDelay = new ArrayList<Integer>();
        for (Integer key : sums.keySet()) {
            List<Object> l = sums.get(key);
            if (l == null) continue;
            for (Object o : l) {
                int subDelay = o instanceof Pair<?, ?> && ((Pair<?, ?>)o).getFirst() instanceof Pair<?, ?> ? 1 : 0; 
                c.add(prodMap.get(o));
                pipelineDelay.add(prodMapDelay.get(o) + subDelay);
                shift.add(key * W + (extraProdMap.containsKey(o) ? 1 : 0));
                if (extraProdMap.containsKey(o)) {
                    c.add(extraProdMap.get(o));
                    pipelineDelay.add(1 + subDelay);
                    shift.add(key * W);
                }
            }
        }
        //System.out.println(aType.getTotalBits() + " Products: " + prodMap.size() + " Sums: " + c.size()); 
        return new Object[] { c.stream().toArray(DFEVar[]::new), shift.stream().mapToInt(Integer::valueOf).toArray(), null, pipelineDelay.stream().mapToInt(Integer::valueOf).toArray() };
    }
    
    public static Object[] mulKaratsubaRectangularInner(DFEVar a, DFEVar b, int tileMode, KernelBase<?> owner) //tileMode is 0 for 16x24, 1 for 17x24, 2 for 18x24
    {
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType();
        if (aType.getTotalBits() > bType.getTotalBits()) return mulKaratsubaRectangularInner(b, a, tileMode, owner);
        int tileX = tileMode + 16, tileY = 24;
        int W = gcd(tileX, tileY);
        int M = tileX / W;
        int N = tileY / W;
        final int DSP_X = 18, DSP_Y = 27; //DSP_Y = 25;
        int MNWa = M * N * W;
        int MNWb1 = (N + 1) * M * W, MNWb2 = (M + 1) * N * W;
        int MNWc = 2 * MNWa;
        int MNWd1 = (2*N + 1) * M * W, MNWd2 = (2 * M + 1) * N * W;
        if (aType.getTotalBits()-(aType.getSignMode() == SignMode.UNSIGNED ? 0 : 1) <= MNWa && bType.getTotalBits()-(bType.getSignMode() == SignMode.UNSIGNED ? 0 : 1) <= MNWa) { //16x24 tiling, 48x48
            DFEVar[] aSplit = splitInt(a, tileX, DSP_X, owner);
            DFEVar[] bSplit = splitInt(b, tileY, DSP_Y, owner);
            List<DFEVar> c = new ArrayList<>();
            List<Integer> shift = new ArrayList<>();
            List<Boolean> issub = new ArrayList<>();
            List<Integer> pipelineDelay = new ArrayList<>();
            for (int i = 0; i < aSplit.length; i++) { //rectangular tiled grade school multiplication
                for (int j = 0; j < bSplit.length; j++) {
                    DFEVar aArg;
                    if (aSplit[i].getType().getTotalBits() == DSP_X && ((DFEFix)aSplit[i].getType()).getSignMode() == SignMode.UNSIGNED && bSplit[j].getType().getTotalBits() > DSP_X-1) {
                        aArg = aSplit[i].get(DSP_X-1).reinterpret(KernelBase.dfeBool());
                        c.add(aArg ? bSplit[j] : owner.constant.zero(KernelBase.dfeFixOffset(bSplit[j].getType().getTotalBits(), 0, ((DFEFix)bSplit[j].getType()).getSignMode())));
                        shift.add(W * (i * M + j * N) + DSP_X-1);
                        pipelineDelay.add(0);
                        issub.add(false);                        
                        aArg = aSplit[i].slice(0, DSP_X-1).reinterpret(KernelBase.dfeUInt(DSP_X-1));
                    } else aArg = aSplit[i]; //i*bSplit.length+j
                    c.add(mulExact(aArg, bSplit[j], owner));
                    shift.add(W * (i * M + j * N));
                    pipelineDelay.add(getMulPipelineDelay((DFEFix)aArg.getType(), (DFEFix)bSplit[j].getType()));
                    issub.add(false);
                }
            }
            return new Object[] { c.toArray(new DFEVar[c.size()]), shift.stream().mapToInt(Integer::intValue).toArray(), booleanAdapter(issub.toArray(new Boolean[issub.size()])), pipelineDelay.stream().mapToInt(Integer::intValue).toArray() };
        } else if (aType.getTotalBits()-(aType.getSignMode() == SignMode.UNSIGNED ? 0 : 1) <= MNWb1 && bType.getTotalBits()-(bType.getSignMode() == SignMode.UNSIGNED ? 0 : 1) <= MNWb2) { //16x24 tiling, 64x72
            DFEVar[] aSplit = splitInt(a, tileX, DSP_X, owner);
            DFEVar[] bSplit = splitInt(b, tileY, DSP_Y, owner);
            //for (int i = 0; i < aSplit.length; i++) aSplit[i] = owner.optimization.pipeline(aSplit[i]);
            //for (int i = 0; i < bSplit.length; i++) bSplit[i] = owner.optimization.pipeline(bSplit[i]);
            int extra = bType.getSignMode() == SignMode.TWOSCOMPLEMENT ? 1 : 0;
            DFEVar[] c = new DFEVar[(M+1)*(N+1)+1+extra];
            int[] shift = new int[(M+1)*(N+1)+1+extra]; //maximum M * N + M + N multiplications
            boolean[] issub = new boolean[(M+1)*(N+1)+1+extra];
            int[] pipelineDelay = new int[(M+1)*(N+1)+1+extra];
            Integer[][] poly = new Integer[(M+1)*(N+1)+1][2];
            for (int i = 0; i <= N; i++) {
                for (int j = 0; j <= M; j++) {
                    if (poly[i * M + j * N][0] == null) poly[i * M + j * N][0] = (i << 8) | j;         
                    else poly[i * M + j * N][1] = (i << 8) | j;
                }
            }
            for (int k = 0; k < poly.length; k++) {
                if (poly[k][0] == null) continue;
                if (poly[k][1] == null) {
                    int i = poly[k][0] >> 8, j = poly[k][0] & 0xFF; 
                    c[i * M + j * N] = mulExact(aSplit[i], bSplit[j], owner);
                    shift[i * M + j * N] = W * (i * M + j * N);
                    pipelineDelay[i * M + j * N] = getMulPipelineDelay((DFEFix)aSplit[i].getType(), (DFEFix)bSplit[j].getType());
                }
            }
            for (int k = 0; k < poly.length; k++) {
                if (poly[k][1] == null) continue;
                //this is not properly general but only works for the specific needed paper case - if 3 terms an error would need to be thrown, 2 terms cannot use empty spaces on such assumptions as +/- 1...
                int i = poly[k][0] >> 8, j = poly[k][0] & 0xFF;
                int m = poly[k][1] >> 8, n = poly[k][1] & 0xFF;
                DFEVar x = addExact(aSplit[i], aSplit[m], true, owner); //take care for the mistake in the paper on subtracting high-low vs low-high
                DFEVar y = addExact(bSplit[j], bSplit[n], true, owner);
                if (extra != 0) { //y.getType().getTotalBits() == 26,28
                    DFEVar ylow = y.get(0).reinterpret(KernelBase.dfeBool());
                    c[c.length-1] = ylow ? x : owner.constant.zero(KernelBase.dfeFixOffset(x.getType().getTotalBits(), 0, ((DFEFix)x.getType()).getSignMode()));
                    shift[shift.length-1] = W * (i * M + j * N);
                    pipelineDelay[pipelineDelay.length-1] = 1;
                    y = y.slice(1, DSP_Y).reinterpret(KernelBase.dfeFixOffset(DSP_Y, 0, ((DFEFix)y.getType()).getSignMode()));
                }
                c[i * M + j * N] = mulExact(x, y, owner);
                shift[i * M + j * N] = W * (i * M + j * N)+extra;
                pipelineDelay[i * M + j * N] = getMulPipelineDelay((DFEFix)x.getType(), (DFEFix)y.getType()) + 1;
                c[i * M + n * N + 1] = c[i * M + n * N];
                shift[i * M + n * N + 1] = W * (i * M + j * N);
                pipelineDelay[i * M + n * N + 1] = pipelineDelay[i * M + n * N];
                c[m * M + j * N - 1] = c[m * M + j * N];
                shift[m * M + j * N - 1] = W * (i * M + j * N);
                pipelineDelay[m * M + j * N - 1] = pipelineDelay[m * M + j * N];
            }
            return new Object[] { c, shift, issub, pipelineDelay };
        } else if (aType.getTotalBits() <= MNWc && bType.getTotalBits() <= MNWc ||  //16x24 tiling, 96x96 breaks down to 3 48x48 multiplications, 95x95 signed...
                   aType.getTotalBits() <= MNWd1 && bType.getTotalBits() <= MNWd2) { //16x24 tiling, 112x120 breaks down to 48x48, 2 64x64 multiplications, 111x119 signed...
            int splitPoint = W * M * N - (aType.getSignMode() == SignMode.TWOSCOMPLEMENT && bType.getSignMode() == SignMode.TWOSCOMPLEMENT ? (tileMode == 2 ? 2 : 1) : 0);
            DFEVar[] aSplit = new DFEVar[] { a.slice(0, splitPoint).reinterpret(KernelBase.dfeUInt(splitPoint)), a.slice(splitPoint, aType.getTotalBits() - splitPoint).reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits() - splitPoint, 0, aType.getSignMode())) };
            DFEVar[] bSplit = new DFEVar[] { b.slice(0, splitPoint).reinterpret(KernelBase.dfeUInt(splitPoint)), b.slice(splitPoint, bType.getTotalBits() - splitPoint).reinterpret(KernelBase.dfeFixOffset(bType.getTotalBits() - splitPoint, 0, bType.getSignMode())) };
            if (tileMode == 2) {
                for (int i = 0; i < aSplit.length; i++) aSplit[i] = owner.optimization.pipeline(aSplit[i]);
                for (int i = 0; i < bSplit.length; i++) bSplit[i] = owner.optimization.pipeline(bSplit[i]);
            }
            Object[] z0 = mulKaratsubaRectangularInner(aSplit[0], bSplit[0], tileMode, owner);
            DFEVar[] c0 = (DFEVar[])z0[0]; int[] shift0 = (int[])z0[1]; boolean[] issub0 = (boolean[])z0[2]; int[] pipelineDelay0 = (int[])z0[3]; 
            Object[] z1 = mulKaratsubaRectangularInner(addExact(aSplit[0], aSplit[1], true, owner), addExact(bSplit[1], bSplit[0], true, owner), tileMode, owner); //take care for the mistake in the paper on subtracting high-low vs low-high
            DFEVar[] c1 = (DFEVar[])z1[0]; int[] shift1 = (int[])z1[1]; boolean[] issub1 = (boolean[])z1[2]; int[] pipelineDelay1 = (int[])z1[3];
            Object[] z2 = mulKaratsubaRectangularInner(aSplit[1], bSplit[1], tileMode, owner);
            DFEVar[] c2 = (DFEVar[])z2[0]; int[] shift2 = (int[])z2[1]; boolean[] issub2 = (boolean[])z2[2]; int[] pipelineDelay2 = (int[])z2[3];
            DFEVar[] c = new DFEVar[c0.length * 2 + c1.length + c2.length * 2];
            int[] shift = new int[c0.length * 2 + c1.length + c2.length * 2];
            boolean[] issub = new boolean[c0.length * 2 + c1.length + c2.length * 2];
            int[] pipelineDelay = new int[c0.length * 2 + c1.length + c2.length * 2];
            int base = 0;
            for (int i = 0; i < c0.length; i++) {
                c[base + i] = c0[i]; shift[base + i] = shift0[i];
                issub[base + i] = issub0[i]; pipelineDelay[base + i] = pipelineDelay0[i]; 
            }
            base += c0.length;
            for (int i = 0; i < c0.length; i++) {
                c[base + i] = c0[i]; shift[base + i] = shift0[i] + splitPoint;
                issub[base + i] = issub0[i]; pipelineDelay[base + i] = pipelineDelay0[i]; 
            }
            base += c0.length;
            for (int i = 0; i < c1.length; i++) {
                c[base + i] = c1[i]; shift[base + i] = shift1[i] + splitPoint;
                issub[base + i] = issub1[i]; pipelineDelay[base + i] = pipelineDelay1[i]+1; 
            }
            base += c1.length;
            for (int i = 0; i < c2.length; i++) {
                c[base + i] = c2[i]; shift[base + i] = shift2[i] + splitPoint;
                issub[base + i] = issub2[i]; pipelineDelay[base + i] = pipelineDelay2[i]; 
            }
            base += c2.length;
            for (int i = 0; i < c2.length; i++) {
                c[base + i] = c2[i]; shift[base + i] = shift2[i] + 2 * splitPoint;
                issub[base + i] = issub2[i]; pipelineDelay[base + i] = pipelineDelay2[i]; 
            }
            return new Object[] { c, shift, issub, pipelineDelay };
        } else { //17x24 tiling does not work until 425x432...so use 18x24 technique with one line done using logic resources, 132x132
            throw new IllegalArgumentException(); //return null;
        }
    }
    public static DFEVar mulKaratsubaRectangular(DFEVar a, DFEVar b, int tileMode, Integer[] totalDelay, boolean useCompressor, KernelBase<?> base) //tileMode is 0 for 16x24, 1 for 17x24, 2 for 18x24
    {
        Object[] objs = mulKaratsubaRectangularSums(a, b, tileMode, useCompressor, base);
        DFEVar[] c = (DFEVar[])objs[0];
        int[] shift = (int[])objs[1];
        boolean[] issub = (boolean[])objs[2];
        int[] pipelineDelay = (int[])objs[3];
        //objs = addPostMultiply(c, shift, issub, pipelineDelay,  base);
        //c = (DFEVar[])objs[0]; shift = (int[])objs[1]; issub = (boolean[])objs[2]; pipelineDelay = (int[])objs[3];
        DFEVar result = addShifter(c, shift, issub, pipelineDelay, totalDelay, useCompressor, base);
        //System.out.println(a.getType().getTotalBits() + "x" + b.getType().getTotalBits() + "->" + result.getType().getTotalBits() + " Delay: " + totalDelay[0]);
        return result.cast(KernelBase.dfeFixOffset(getMulOutpBits((DFEFix)a.getType(), (DFEFix)b.getType()), 0, ((DFEFix)a.getType()).getSignMode() == SignMode.UNSIGNED && ((DFEFix)b.getType()).getSignMode() == SignMode.UNSIGNED ? SignMode.UNSIGNED : SignMode.TWOSCOMPLEMENT));
    }
    public static DFEVar mulKaratsubaExact(DFEVar a, DFEVar b, int bits, int intBits, KernelBase<?> base) {
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType();
        Integer[] delay = new Integer[1];
        DFEVar result = mulKaratsuba(a.reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits(), 0, aType.getSignMode())),
                                        b.reinterpret(KernelBase.dfeFixOffset(bType.getTotalBits(), 0, bType.getSignMode())), delay, base);
        result = result
            .reinterpret(KernelBase.dfeFixOffset(getMulOutpBits(aType, bType), -(aType.getFractionBits() + bType.getFractionBits()), ((DFEFix)result.getType()).getSignMode()))
            .cast(KernelBase.dfeFixOffset(bits, -bits+intBits, SignMode.TWOSCOMPLEMENT));
        return result;
    }
    public static Object[] mulKaratsubaInner(DFEVar a, DFEVar b, KernelBase<?> owner) {
        final int DSP_X = 18, DSP_Y = 27; //DSP_Y = 25;
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType();
        if ((aType.getTotalBits() % (DSP_Y-1) == 0 ? 0 : ((DSP_Y-1) - aType.getTotalBits() % (DSP_Y-1))) + (bType.getTotalBits() % (DSP_X-1) == 0 ? 0 : ((DSP_X-1) - bType.getTotalBits() % (DSP_X-1))) <
            (aType.getTotalBits() % (DSP_X-1) == 0 ? 0 : ((DSP_X-1) - aType.getTotalBits() % (DSP_X-1))) + (bType.getTotalBits() % (DSP_Y-1) == 0 ? 0 : ((DSP_Y-1) - bType.getTotalBits() % (DSP_Y-1))))
            return mulKaratsubaInner(b, a, owner);
        if (getDSPCount(aType.getTotalBits(), bType.getTotalBits()) <= 5) {
            return new Object[] { new DFEVar[] { mulExact(a, b, owner) }, new int[] { 0 }, new boolean[] { false }, new int[] { getMulPipelineDelay(aType, bType) }};
        }
        int aDiv = aType.getTotalBits() / 2, bDiv = bType.getTotalBits() / 2;
        aDiv = aDiv - aDiv % (DSP_X-1) + (aDiv % (DSP_X-1) > 8 ? (DSP_X-1) : 0); //ideally we want to round up/down to minimize abs(b-a)
        bDiv = bDiv - bDiv % (DSP_Y-1) + (bDiv % (DSP_Y-1) > 11 ? (DSP_Y-1) : 0); //divide closeness ranges 0-8, 9-17, 0-11, 12-23
        DFEVar high1 = a.slice(aDiv, aType.getTotalBits() - aDiv).reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits() - aDiv, 0, aType.getSignMode())), low1 = a.slice(0, aDiv).reinterpret(KernelBase.dfeUInt(aDiv));
        DFEVar high2 = b.slice(bDiv, bType.getTotalBits() - bDiv).reinterpret(KernelBase.dfeFixOffset(bType.getTotalBits() - bDiv, 0, bType.getSignMode())), low2 = b.slice(0, bDiv).reinterpret(KernelBase.dfeUInt(bDiv));
        Object[] z0 = mulKaratsubaInner(low1, low2, owner), z1;
        DFEVar[] c0 = (DFEVar[])z0[0]; int[] shift0 = (int[])z0[1]; boolean[] issub0 = (boolean[])z0[2]; int[] pipelineDelay0 = (int[])z0[3];
        //(x0+x1*2^a)(y0+y1*2^b)=x0y0+y0x1*2^a+x0y1*2^b+x1y1*2^(a+b) 
        //z0 = x0y0
        //z1 = x1y0*2^a + x0y1*2^b = 2^a(x1y0 + x0y1*2^(b-a)) if b>a
        //z2 = x1y1
        //result = z2*2^(a+b)+ z1 + z0
        //z1 = 2^a((x0-x1)(y1*2^(b-a)-y0) + z2*2^(b-a) + z0)
        int mn = Math.min(aDiv, bDiv);
        if (aDiv == bDiv)
            z1 = mulKaratsubaInner(addExact(low1, high1, true, owner), addExact(high2, low2, true, owner), owner);
        else if (aDiv < bDiv)
            z1 = mulKaratsubaInner(addExact(low1, high1, true, owner), addExact(shiftLeftFix(high2, bDiv - aDiv, owner), low2, true, owner), owner);
        else
            z1 = mulKaratsubaInner(addExact(shiftLeftFix(high1, aDiv - bDiv, owner), low1, true, owner), addExact(low2, high2, true, owner), owner);
        DFEVar[] c1 = (DFEVar[])z1[0]; int[] shift1 = (int[])z1[1]; boolean[] issub1 = (boolean[])z1[2]; int[] pipelineDelay1 = (int[])z1[3];
        Object[] z2 = mulKaratsubaInner(high1, high2, owner);
        DFEVar[] c2 = (DFEVar[])z2[0]; int[] shift2 = (int[])z2[1]; boolean[] issub2 = (boolean[])z2[2]; int[] pipelineDelay2 = (int[])z2[3];
        DFEVar[] c = new DFEVar[c0.length * 2 + c1.length + c2.length * 2];
        int[] shift = new int[c0.length * 2 + c1.length + c2.length * 2];
        boolean[] issub = new boolean[c0.length * 2 + c1.length + c2.length * 2];
        int[] pipelineDelay = new int[c0.length * 2 + c1.length + c2.length * 2];
        int base = 0;
        for (int i = 0; i < c0.length; i++) {
            c[base + i] = c0[i]; shift[base + i] = shift0[i];
            issub[base + i] = issub0[i]; pipelineDelay[base + i] = pipelineDelay0[i]; 
        }
        base += c0.length;
        for (int i = 0; i < c0.length; i++) {
            c[base + i] = c0[i]; shift[base + i] = shift0[i] + mn;
            issub[base + i] = issub0[i]; pipelineDelay[base + i] = pipelineDelay0[i]; 
        }
        base += c0.length;
        for (int i = 0; i < c1.length; i++) {
            c[base + i] = c1[i]; shift[base + i] = shift1[i] + mn;
            issub[base + i] = issub1[i]; pipelineDelay[base + i] = pipelineDelay1[i]+1; 
        }
        base += c1.length;
        for (int i = 0; i < c2.length; i++) {
            c[base + i] = c2[i]; shift[base + i] = shift2[i] + (bDiv > aDiv ? bDiv : aDiv);
            issub[base + i] = issub2[i]; pipelineDelay[base + i] = pipelineDelay2[i]; 
        }
        base += c2.length;
        for (int i = 0; i < c2.length; i++) {
            c[base + i] = c2[i]; shift[base + i] = shift2[i] + aDiv + bDiv;
            issub[base + i] = issub2[i]; pipelineDelay[base + i] = pipelineDelay2[i]; 
        }
        return new Object[] { c, shift, issub, pipelineDelay };        
    }
    public static DFEVar mulKaratsuba(DFEVar a, DFEVar b, Integer[] totalDelay, KernelBase<?> base)
    {
        Object[] objs = mulKaratsubaInner(a, b, base);
        DFEVar[] c = (DFEVar[])objs[0];
        int[] shift = (int[])objs[1];
        boolean[] issub = (boolean[])objs[2];
        int[] pipelineDelay = (int[])objs[3];
        DFEVar result = addShifter(c, shift, issub, pipelineDelay, totalDelay, false, base);
        return result.cast(KernelBase.dfeFixOffset(getMulOutpBits((DFEFix)a.getType(), (DFEFix)b.getType()), 0, ((DFEFix)result.getType()).getSignMode()));
    }    
    public static DFEVar mulKaratsubaSingleRoundExact(DFEVar a, DFEVar b, int bits, int intBits, KernelBase<?> base) {
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType();
        Integer[] delay = new Integer[1];
        DFEVar result = mulKaratsubaSingleRound(a.reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits(), 0, aType.getSignMode())),
                                        b.reinterpret(KernelBase.dfeFixOffset(bType.getTotalBits(), 0, bType.getSignMode())), delay, base);
        result = result
            .reinterpret(KernelBase.dfeFixOffset(getMulOutpBits(aType, bType), -(aType.getFractionBits() + bType.getFractionBits()), ((DFEFix)result.getType()).getSignMode()))
            .cast(KernelBase.dfeFixOffset(bits, -bits+intBits, SignMode.TWOSCOMPLEMENT));
        return result;
    }
    public static DFEVar mulKaratsubaSingleRound(DFEVar a, DFEVar b, Integer[] totalDelay, KernelBase<?> base)
    {
        DFEFix aType = (DFEFix)a.getType(), bType = (DFEFix)b.getType();
        if (aType.getTotalBits() <= 65 && bType.getTotalBits() <= 65) return mulExact(a, b, base);
        int aDiv = aType.getTotalBits() / 2;
        int bDiv = bType.getTotalBits() / 2;
        int mn = Math.min(aDiv, bDiv);
        DFEVar high1 = a.slice(aDiv, aType.getTotalBits() - aDiv).reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits() - aDiv, 0, aType.getSignMode())), low1 = a.slice(0, aDiv).reinterpret(KernelBase.dfeUInt(aDiv));
        DFEVar high2 = b.slice(bDiv, bType.getTotalBits() - bDiv).reinterpret(KernelBase.dfeFixOffset(bType.getTotalBits() - bDiv, 0, bType.getSignMode())), low2 = b.slice(0, bDiv).reinterpret(KernelBase.dfeUInt(bDiv));
        DFEVar z0 = sToUnsigned(mulExact(uToSigned(low1, base), uToSigned(low2, base), base)), //maxcompiler has issues with large-bit unsigned latency estimates
            z1, z2 = mulExact(high1, high2, base);
        if (aDiv == bDiv)
            z1 = mulExact(addExact(low1, high1, true, base), addExact(high2, low2, true, base), base);
        else if (aDiv < bDiv)
            z1 = mulExact(addExact(low1, high1, true, base), addExact(shiftLeftFix(high2, bDiv - aDiv, base), low2, true, base), base);
        else
            z1 = mulExact(addExact(shiftLeftFix(high1, aDiv - bDiv, base), low1, true, base), addExact(low2, high2, true, base), base);
        DFEVar result = addShifter(new DFEVar[] { z0, z0, z1, z2, z2 }, new int[] { 0, mn, mn, (bDiv > aDiv ? bDiv : aDiv), aDiv + bDiv},
            new boolean[] {false, false, false, false, false}, new int[] { 0+(aType.getTotalBits() == 127 ? 0 : 1), 0+(aType.getTotalBits() == 127 ? 0 : 1), 1+(aType.getTotalBits() == 128 ? 0 : 3), 0, 0 }, totalDelay, false, base);
        return result.cast(KernelBase.dfeFixOffset(getMulOutpBits((DFEFix)a.getType(), (DFEFix)b.getType()), 0, ((DFEFix)result.getType()).getSignMode()));
    }
    public static int getSignedMulKaratsubaDSPCount(int a, int b) {
        return getMulKaratsubaDSPCount(a-1, b-1)+1;
    }
    public static int getMulKaratsubaDSPCount(int a, int b) {
        final int DSP_X = 18, DSP_Y = 27; //DSP_Y = 25;
        if ((a % (DSP_Y-1) == 0 ? 0 : ((DSP_Y-1) - a % (DSP_Y-1))) + (b % (DSP_X-1) == 0 ? 0 : ((DSP_X-1) - b % (DSP_X-1))) <
            (a % (DSP_X-1) == 0 ? 0 : ((DSP_X-1) - a % (DSP_X-1))) + (b % (DSP_Y-1) == 0 ? 0 : ((DSP_Y-1) - b % (DSP_Y-1))))
            return getMulKaratsubaDSPCount(b, a);
        if (getDSPCount(a, b) <= 5) return getDSPCount(a, b);
        int aDiv = a / 2, bDiv = b / 2;
        aDiv = aDiv - aDiv % (DSP_X-1) + (aDiv % (DSP_X-1) > 8 ? (DSP_X-1) : 0); //ideally we want to round up/down to minimize abs(b-a)
        bDiv = bDiv - bDiv % (DSP_Y-1) + (bDiv % (DSP_Y-1) > 11 ? (DSP_Y-1) : 0); //divide closeness ranges 0-8, 9-17, 0-11, 12-23
        int z0 = getMulKaratsubaDSPCount(aDiv, bDiv), z2 = getMulKaratsubaDSPCount(a-aDiv, b-bDiv);
        int z1;
        if (aDiv == bDiv) z1 = getSignedMulKaratsubaDSPCount(Math.max(aDiv, a-aDiv)+1, Math.max(bDiv, b-bDiv)+1);
        else if (aDiv < bDiv) z1 = getSignedMulKaratsubaDSPCount(Math.max(aDiv, a-aDiv)+1, Math.max(bDiv+bDiv-aDiv, b-bDiv)+1);
        else z1 = getSignedMulKaratsubaDSPCount(Math.max(aDiv+aDiv-bDiv, a-aDiv)+1, Math.max(bDiv, b-bDiv)+1);
        return z0 + z1 + z2;        
    }
    public static DFEVar uToSigned(DFEVar a, KernelBase<?> base) {
        //KernelBase.dfeInt(a.getType().getTotalBits() + 1);
        if (((DFEFix)a.getType()).getSignMode() == SignMode.TWOSCOMPLEMENT) return a;
        return base.constant.zero(KernelBase.dfeBool()).cat(a).reinterpret(KernelBase.dfeFixOffset(a.getType().getTotalBits() + 1, -((DFEFix)a.getType()).getFractionBits(), SignMode.TWOSCOMPLEMENT));
    }
    public static DFEVar sToUnsigned(DFEVar a) {
        //KernelBase.dfeUInt(a.getType().getTotalBits() - 1)
        if (((DFEFix)a.getType()).getSignMode() == SignMode.UNSIGNED) return a;
        return a.slice(0, a.getType().getTotalBits() - 1).reinterpret(KernelBase.dfeFixOffset(a.getType().getTotalBits() - 1, -((DFEFix)a.getType()).getFractionBits(), SignMode.UNSIGNED));
    }
    public static DFEVar shiftLeftFix(DFEVar a, int bits, KernelBase<?> base) {
        if (bits == 0) return a;
        DFEFix aType = (DFEFix)a.getType();
        //return a.cat(base.constant.zero(base.dfeUInt(bits))).reinterpret(KernelBase.dfeFixOffset(a.getType().getTotalBits() + bits, bits - aType.getFractionBits(), aType.getSignMode()));
        return a.reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits(), bits - aType.getFractionBits(), aType.getSignMode()));
    }
    public static Pair<Integer, Integer> intersectRange(int min1, int max1, int min2, int max2)
    {
        if (min2 > max1 || min1 > max2) return null;
        return new Pair<Integer, Integer>(Math.max(min1, min2), Math.min(max1, max2));
    }
    public static List<List<DFEVar>> getTriBitChunks128(DFEVar a, DFEVar b, DFEVar c, boolean isBSub, boolean isCSub, boolean sizeOnly)
    {
        final int CHUNKSIZE = 128;
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType(), cType = (DFEFix) c.getType();
        DFEFix resultType = getTriAddExactType(aType, bType, cType, isBSub, isCSub);
        if (resultType.getTotalBits() <= CHUNKSIZE) return null; //no chunking is needed
        int lsb; 
        DFEFix bcType = isCSub ? getAddExactType(bType, cType, isCSub) : getAddExactType(cType, bType, isBSub);
        DFEFix acType = getAddExactType(aType, cType, isCSub);
        DFEFix abType = getAddExactType(aType, bType, isBSub);
        if ((!isBSub || !isCSub) && -aType.getFractionBits() > bcType.getIntegerBits()) {
            return getBitChunks128(b, c, isCSub, true);
        } else if (!isBSub && -bType.getFractionBits() > acType.getIntegerBits()) {
            return getBitChunks128(a, c, isCSub, true);
        } else if (!isCSub && -cType.getFractionBits() > abType.getIntegerBits()) {
            return getBitChunks128(a, b, isBSub, true);
        }
        if (aType.getFractionBits() > bcType.getFractionBits()) { //if splicing did not yet occur we account for it anyway
            lsb = -bcType.getFractionBits();
        } else if (!isBSub && bType.getFractionBits() > acType.getFractionBits()) {
            lsb = -acType.getFractionBits();
        } else if (!isCSub && cType.getFractionBits() > abType.getFractionBits()) {
            lsb = -abType.getFractionBits();
        } else lsb = -Math.max(Math.max(aType.getFractionBits(), bType.getFractionBits()), cType.getFractionBits());        
        int msb = Math.max(Math.max(aType.getIntegerBits(), bType.getIntegerBits()), cType.getIntegerBits());
        List<List<DFEVar>> chunks = new ArrayList<>();
        for (int i = lsb; i < msb; ) { //going from least to most significant byte order might miss special optimizations for binary/tri-adder, very difficult to handle generally 
            int numbits = 127; //numbits will be 125, 126 or 127 depending on if 3 arguments, subtraction, signedness of result
            List<DFEFix> types = new ArrayList<>();
            do {
                int j = i + numbits;
                Pair<Integer, Integer> ap = intersectRange(-aType.getFractionBits(), aType.getIntegerBits(), i, j);
                Pair<Integer, Integer> bp = intersectRange(-bType.getFractionBits(), bType.getIntegerBits(), i, j);
                Pair<Integer, Integer> cp = intersectRange(-cType.getFractionBits(), cType.getIntegerBits(), i, j);
                if (ap != null) {
                    boolean consumed = aType.getIntegerBits() == ap.second;
                    types.add(KernelBase.dfeFixOffset(ap.second - ap.first, ap.first, consumed ? aType.getSignMode() : SignMode.UNSIGNED));
                } else types.add(null);
                if (bp != null) {
                    boolean consumed = bType.getIntegerBits() == bp.second;
                    types.add(KernelBase.dfeFixOffset(bp.second - bp.first, bp.first, consumed ? bType.getSignMode() : SignMode.UNSIGNED));
                } else types.add(null);
                if (cp != null) {
                    boolean consumed = cType.getIntegerBits() == cp.second;
                    types.add(KernelBase.dfeFixOffset(cp.second - cp.first, cp.first, consumed ? cType.getSignMode() : SignMode.UNSIGNED));
                } else types.add(null);
                if (types.get(0) != null && types.get(1) != null && types.get(2) != null) {
                    if (getTriAddExactType(types.get(0), types.get(1), types.get(2), isBSub, isCSub).getTotalBits() <= CHUNKSIZE) break;
                } else if (types.get(0) != null && types.get(1) != null) {
                    if (getAddExactType(types.get(0), types.get(1), isBSub).getTotalBits() <= CHUNKSIZE) break;
                } else if (types.get(0) != null && types.get(2) != null) {
                    if (getAddExactType(types.get(0), types.get(2), isCSub).getTotalBits() <= CHUNKSIZE) break;
                } else if (types.get(1) != null && types.get(2) != null) {
                    if (!isBSub) {
                        if (getAddExactType(types.get(1), types.get(2), isCSub).getTotalBits() <= CHUNKSIZE) break;
                    } else {
                        if (getAddExactType(types.get(2), types.get(1), !isCSub).getTotalBits() <= CHUNKSIZE) break;
                    }
                } else break;
                numbits--;
                types.clear();
            } while (true);
            List<DFEVar> vars = new ArrayList<>();
            if (!sizeOnly) {
                if (types.get(0) == null) vars.add(null);
                else vars.add(a.slice(aType.getFractionBits() - types.get(0).getFractionBits(), types.get(0).getTotalBits()).reinterpret(types.get(0)));
                if (types.get(1) == null) vars.add(null);
                else vars.add(b.slice(bType.getFractionBits() - types.get(1).getFractionBits(), types.get(1).getTotalBits()).reinterpret(types.get(1)));
                if (types.get(2) == null) vars.add(null);
                else vars.add(c.slice(cType.getFractionBits() - types.get(2).getFractionBits(), types.get(2).getTotalBits()).reinterpret(types.get(2)));
            }
            chunks.add(vars);
            i += numbits;
        }
        return chunks;
    }
    public static void verifyTriAddExact(DFEVar verify, DFEVar a, DFEVar b, DFEVar c, boolean isBSub, boolean isCSub, KernelBase<?> base) {
        DFEFix resultType = getTriAddExactType((DFEFix) a.getType(), (DFEFix) b.getType(), (DFEFix) c.getType(), isBSub, isCSub);
        base.optimization.pushFixOpMode(Optimization.bitSizeExact(resultType.getTotalBits()),
            Optimization.offsetExact(-resultType.getFractionBits()), MathOps.ADD_SUB);
        //base.optimization.pushEnableBitGrowth(true);
        DFEVar result;
        if (isBSub) result = isCSub ? (a - b - c) : (a - b + c);
        else result = isCSub ? (a + b - c) : (a + b + c);
        base.optimization.popFixOpMode(MathOps.ADD_SUB);
        //base.optimization.popEnableBitGrowth();
        result = (isBSub || isCSub) && ((DFEFix)result.getType()).getSignMode() == SignMode.UNSIGNED ? result.reinterpret(resultType) : result;
        base.debug.simPrintf(verify.cast(result.getType()) !== result, "%X %X " + isBSub + " " + isCSub + "\n", verify.reinterpret(KernelBase.dfeUInt(verify.getType().getTotalBits())), result.reinterpret(KernelBase.dfeUInt(result.getType().getTotalBits())));
    }    
    public static DFEVar triAddExact(DFEVar a, DFEVar b, DFEVar c, boolean isBSub, boolean isCSub, KernelBase<?> base) {
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType(), cType = (DFEFix) c.getType();
        DFEFix resultType = getTriAddExactType(aType, bType, cType, isBSub, isCSub);        
        DFEFix bcType = isCSub ? getAddExactType(bType, cType, isCSub) : getAddExactType(cType, bType, isBSub);
        DFEFix acType = getAddExactType(aType, cType, isCSub);
        DFEFix abType = getAddExactType(aType, bType, isBSub);
        if (bType.getSignMode() == SignMode.UNSIGNED && cType.getSignMode() == SignMode.UNSIGNED && !isBSub && !isCSub && -aType.getFractionBits() > bcType.getIntegerBits()) {
            List<List<DFEVar>> chunks = getBitChunks128(b, c, isCSub, true);
            resultType = KernelBase.dfeFixOffset(bcType.getTotalBits()+aType.getTotalBits()-aType.getFractionBits() - bcType.getIntegerBits(), -bcType.getFractionBits(), resultType.getSignMode());
            DFEVar result = repeatPipeline(a, 1 + (chunks != null ? chunks.size() - 1 : 0), base).cat(base.constant.zero(KernelBase.dfeUInt(-aType.getFractionBits() - bcType.getIntegerBits()))).cat(addExact(b, c, isCSub, base)).reinterpret(resultType);
            //verifyTriAddExact(result, a, b, c, isBSub, isCSub, base);
            return result;
        } else if (aType.getSignMode() == SignMode.UNSIGNED && cType.getSignMode() == SignMode.UNSIGNED && !isCSub && -bType.getFractionBits() > acType.getIntegerBits()) {
            List<List<DFEVar>> chunks = getBitChunks128(a, c, isCSub, true);
            resultType = KernelBase.dfeFixOffset(acType.getTotalBits()+bType.getTotalBits()-bType.getFractionBits() - acType.getIntegerBits(), -acType.getFractionBits(), resultType.getSignMode());
            DFEVar result = (isBSub ? -b : repeatPipeline(b, 1 + (chunks != null ? chunks.size() - 1 : 0), base)).cat(base.constant.zero(KernelBase.dfeUInt(-bType.getFractionBits() - acType.getIntegerBits()))).cat(addExact(a, c, isCSub, base)).reinterpret(resultType);
            //verifyTriAddExact(result, a, b, c, isBSub, isCSub, base);
            return result;
        } else if (aType.getSignMode() == SignMode.UNSIGNED && bType.getSignMode() == SignMode.UNSIGNED && !isBSub && -cType.getFractionBits() > abType.getIntegerBits()) {
            List<List<DFEVar>> chunks = getBitChunks128(a, b, isBSub, true);
            resultType = KernelBase.dfeFixOffset(abType.getTotalBits()+cType.getTotalBits()-cType.getFractionBits() - abType.getIntegerBits(), -abType.getFractionBits(), resultType.getSignMode());
            DFEVar result = (isCSub ? -c : repeatPipeline(c, 1 + (chunks != null ? chunks.size() - 1 : 0), base)).cat(base.constant.zero(KernelBase.dfeUInt(-cType.getFractionBits() - abType.getIntegerBits()))).cat(addExact(a, b, isBSub, base)).reinterpret(resultType);
            //verifyTriAddExact(result, a, b, c, isBSub, isCSub, base);
            return result;
        }
        //first we splice off any trailing bits
        if (aType.getFractionBits() > bcType.getFractionBits()) { //isBSub isCSub in above calculation not relevant to fraction bits so we reuse...
            int bitDiff = aType.getFractionBits() - bcType.getFractionBits();
            //DFEVar aOrig = a;
            if (aType.getSignMode() == SignMode.TWOSCOMPLEMENT && aType.getTotalBits() <= bitDiff+1) { //remembering that the smallest twos complement number is 2 bits
                aType = KernelBase.dfeFixOffset(aType.getTotalBits()+bitDiff-aType.getTotalBits()+2, -aType.getFractionBits(), aType.getSignMode());
                a = a.cast(aType);
            }
            DFEVar asplice = bitDiff > aType.getTotalBits() ? base.constant.zero(KernelBase.dfeUInt(bitDiff - aType.getTotalBits())).cat(a) : a.slice(0, bitDiff);
            List<List<DFEVar>> chunks;
            DFEVar result;
            if (aType.getTotalBits() <= bitDiff) {
                if (isBSub && isCSub) {
                    a = base.constant.zero(KernelBase.dfeBool());
                    chunks = getTriBitChunks128(a, b, c, isBSub, isCSub, true);
                    result = doTriAddExact(a, b, c, isBSub, isCSub, base);                    
                } else if (isCSub) {
                    chunks = getBitChunks128(b, c, isCSub, true);
                    result = doAddExact(b, c, isCSub, base);
                    resultType = KernelBase.dfeFixOffset(resultType.getTotalBits()-1, resultType.getOffset(), resultType.getSignMode());
                } else {
                    chunks = getBitChunks128(c, b, isBSub, true);
                    result = doAddExact(c, b, isBSub, base);
                    resultType = KernelBase.dfeFixOffset(resultType.getTotalBits()-1, resultType.getOffset(), resultType.getSignMode());
                }
            } else {
                a = a.slice(bitDiff, aType.getTotalBits() - bitDiff).reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits() - bitDiff, -(aType.getFractionBits() - bitDiff), aType.getSignMode()));
                chunks = getTriBitChunks128(a, b, c, isBSub, isCSub, true);
                result = doTriAddExact(a, b, c, isBSub, isCSub, base);
            }
            result = result.cat(repeatPipeline(asplice, 1 + (chunks != null ? chunks.size()-1 : 0), base)).reinterpret(resultType);
            //verifyTriAddExact(result, aOrig, b, c, isBSub, isCSub, base);
            return result;            
        } else if (!isBSub && bType.getFractionBits() > acType.getFractionBits()) {
            int bitDiff = bType.getFractionBits() - acType.getFractionBits();
            //DFEVar bOrig = b;
            if (bitDiff > bType.getTotalBits()) throw new IllegalArgumentException(); //not yet handled case as above
            DFEVar bsplice = b.slice(0, bitDiff);
            b = b.slice(bitDiff, bType.getTotalBits() - bitDiff).reinterpret(KernelBase.dfeFixOffset(bType.getTotalBits() - bitDiff, -(bType.getFractionBits() - bitDiff), bType.getSignMode()));
            List<List<DFEVar>> chunks = getTriBitChunks128(a, b, c, isBSub, isCSub, true);
            DFEVar result = doTriAddExact(a, b, c, isBSub, isCSub, base);
            result = result.cat(repeatPipeline(bsplice, 1 + (chunks != null ? chunks.size()-1 : 0), base)).reinterpret(resultType);
            //verifyTriAddExact(result, a, bOrig, c, isBSub, isCSub, base);
            return result;            
        } else if (!isCSub && cType.getFractionBits() > abType.getFractionBits()) {
            int bitDiff = cType.getFractionBits() - abType.getFractionBits();
            //DFEVar cOrig = c;
            if (bitDiff > cType.getTotalBits()) throw new IllegalArgumentException(); //not yet handled case as above
            DFEVar csplice = c.slice(0, bitDiff);
            c = c.slice(bitDiff, cType.getTotalBits() - bitDiff).reinterpret(KernelBase.dfeFixOffset(cType.getTotalBits() - bitDiff, -(cType.getFractionBits() - bitDiff), cType.getSignMode()));
            List<List<DFEVar>> chunks = getTriBitChunks128(a, b, c, isBSub, isCSub, true);
            DFEVar result = doTriAddExact(a, b, c, isBSub, isCSub, base);
            result = result.cat(repeatPipeline(csplice, 1 + (chunks != null ? chunks.size()-1 : 0), base)).reinterpret(resultType);
            //verifyTriAddExact(result, a, b, cOrig, isBSub, isCSub, base);
	        return result;
        } else {
            DFEVar result = doTriAddExact(a, b, c, isBSub, isCSub, base);
            //verifyTriAddExact(result, a, b, c, isBSub, isCSub, base);
            return result;
        }
    }
    public static DFEFix getTriAddExactType(DFEFix aType, DFEFix bType, DFEFix cType, boolean isBSub, boolean isCSub) {
        int fracBits = Math.max(Math.max(aType.getFractionBits(), bType.getFractionBits()), cType.getFractionBits());
        boolean growSign = aType.getSignMode() == SignMode.UNSIGNED && (bType.getSignMode() == SignMode.TWOSCOMPLEMENT || cType.getSignMode() == SignMode.TWOSCOMPLEMENT) &&
            aType.getIntegerBits() >= bType.getIntegerBits() && aType.getIntegerBits() >= cType.getIntegerBits() ||
                            bType.getSignMode() == SignMode.UNSIGNED && (aType.getSignMode() == SignMode.TWOSCOMPLEMENT || cType.getSignMode() == SignMode.TWOSCOMPLEMENT) &&
            aType.getIntegerBits() <= bType.getIntegerBits() && bType.getIntegerBits() >= cType.getIntegerBits() ||
                            cType.getSignMode() == SignMode.UNSIGNED && (aType.getSignMode() == SignMode.TWOSCOMPLEMENT || bType.getSignMode() == SignMode.TWOSCOMPLEMENT) &&
            aType.getIntegerBits() <= cType.getIntegerBits() && bType.getIntegerBits() <= cType.getIntegerBits();
        return KernelBase.dfeFixOffset(1 + 1 + (growSign ? 1 : 0) + Math.max(Math.max(aType.getIntegerBits(), bType.getIntegerBits()), cType.getIntegerBits()) +
                fracBits, -fracBits, isBSub || isCSub || aType.getSignMode() == SignMode.TWOSCOMPLEMENT || bType.getSignMode() == SignMode.TWOSCOMPLEMENT || cType.getSignMode() == SignMode.TWOSCOMPLEMENT ? SignMode.TWOSCOMPLEMENT : SignMode.UNSIGNED);
    }
    public static DFEVar doTriAddExact(DFEVar a, DFEVar b, DFEVar c, boolean isBSub, boolean isCSub, KernelBase<?> base) {
        DFEFix resultType = getTriAddExactType((DFEFix) a.getType(), (DFEFix) b.getType(), (DFEFix) c.getType(), isBSub, isCSub);
        List<List<DFEVar>> chunks = getTriBitChunks128(a, b, c, isBSub, isCSub, false);
        if (chunks != null) {
            List<DFEVar> results = new ArrayList<>();
            for (List<DFEVar> batch : chunks) {
                if (batch.get(0) != null && batch.get(1) != null && batch.get(2) != null) {
                    results.add(doTriAddExact(batch.get(0), batch.get(1), batch.get(2), isBSub, isCSub, base));
                } else if (batch.get(0) != null && batch.get(1) != null) {
                    results.add(doAddExact(batch.get(0), batch.get(1), isBSub, base));
                } else if (batch.get(0) != null && batch.get(2) != null) {
                    results.add(doAddExact(batch.get(0), batch.get(2), isCSub, base));
                } else if (batch.get(1) != null && batch.get(2) != null) {
                    if (!isBSub)
                        results.add(doAddExact(batch.get(1), batch.get(2), isCSub, base));
                    else results.add(doAddExact(batch.get(2), batch.get(1), !isCSub, base));
                } else if (batch.get(0) != null) results.add(base.optimization.pipeline(batch.get(0)));
                else if (batch.get(1) != null) results.add(base.optimization.pipeline(batch.get(1)));
                else if (batch.get(2) != null) results.add(base.optimization.pipeline(batch.get(2)));
            }
            DFEVar result = results.get(0);
            int i;
            for (i = 1; i < results.size(); i++) result = addExact(repeatPipeline(results.get(i), i-1, base), result,
                chunks.get(i-1).get(0) == null && chunks.get(i-1).get(1) != null && chunks.get(i-1).get(2) == null && isBSub ||
                chunks.get(i-1).get(0) == null && chunks.get(i-1).get(1) == null && chunks.get(i-1).get(2) != null && isCSub ||
                chunks.get(i-1).get(0) == null && chunks.get(i-1).get(1) != null && chunks.get(i-1).get(2) != null && isBSub && isCSub, base);
            if (chunks.get(i-1).get(0) == null && chunks.get(i-1).get(1) != null && chunks.get(i-1).get(2) == null && isBSub ||
                chunks.get(i-1).get(0) == null && chunks.get(i-1).get(1) == null && chunks.get(i-1).get(2) != null && isCSub ||
                chunks.get(i-1).get(0) == null && chunks.get(i-1).get(1) != null && chunks.get(i-1).get(2) != null && isBSub && isCSub) result = -result;
            //DFEVar realResult = doTriAddExact(a, b, c, isBSub, isCSub, base, false);
            //base.debug.simPrintf(result.cast(resultType) !== realResult, "%d %d %X %X %X %X %X\n", isBSub ? 1 : 0, isCSub ? 1 : 0, a.reinterpret(KernelBase.dfeUInt(a.getType().getTotalBits())), b.reinterpret(KernelBase.dfeUInt(b.getType().getTotalBits())), c.reinterpret(KernelBase.dfeUInt(c.getType().getTotalBits())), result.reinterpret(KernelBase.dfeUInt(result.getType().getTotalBits())), realResult.reinterpret(KernelBase.dfeUInt(realResult.getType().getTotalBits())));
            result = result.cast(resultType); //an extra bit can grow in chained operations, so it is eliminated here as we know the carry only occurred in the chunk add or chain stage, not both
            //verifyTriAddExact(result, a, b, c, isBSub, isCSub, base);
            return result; 
        }
        base.optimization.pushFixOpMode(Optimization.bitSizeExact(resultType.getTotalBits()),
            Optimization.offsetExact(-resultType.getFractionBits()), MathOps.ADD_SUB);
        //base.optimization.pushEnableBitGrowth(true);
        DFEVar result;
        if (isBSub) result = isCSub ? (a - b - c) : (a - b + c);
        else result = isCSub ? (a + b - c) : (a + b + c);
        base.optimization.popFixOpMode(MathOps.ADD_SUB);
        //base.optimization.popEnableBitGrowth();
        return (isBSub || isCSub) && ((DFEFix)result.getType()).getSignMode() == SignMode.UNSIGNED ? result.reinterpret(resultType) : result;
    }
    public static DFEVar mulFrac(DFEVar a, DFEVar b, KernelBase<?> base) //a is between -1 and 1, b is between -2 and 2
    {
        DFEType t = a.getType();
        int fracBits = t.getTotalBits() - 3;
        DFEVar asign = a.get(fracBits + 2);
        DFEVar bsign = b.get(fracBits + 2);
        a = asign ? -a : a;
        b = bsign ? -b : b;
        DFEVar aIntPart = a.get(fracBits);
        DFEVar bIntPart = b.slice(fracBits, 2);
        DFEType mulType = KernelBase.dfeFixOffset(fracBits, -fracBits, SignMode.UNSIGNED);
        a = a.slice(0, fracBits).reinterpret(mulType);
        b = b.slice(0, fracBits).reinterpret(mulType);
        //(a+b)(c+d)=ac+ad+bc+bd
        DFEVar at = a.cast(t);
        DFEVar bt = b.cast(t);
        DFEVar oneat = base.constant.var(KernelBase.dfeUInt(3), 1).cat(a).reinterpret(t);
        DFEVar twobt = bt << 1;
        DFEVar zt = base.constant.zero(t);
        DFEVar result = base.control.mux(bIntPart.cat(aIntPart), zt, bt, at, oneat + bt, twobt, oneat + twobt, zt, zt) + (a * b).cast(t);
        return (asign ^ bsign) ? -result : result;
    }
    public static DFEVar getSignBit(DFEVar x, KernelBase<?> base) {
        return ((DFEFix)x.getType()).getSignMode() == SignMode.UNSIGNED ? base.constant.zero(KernelBase.dfeBool()) : x.get(x.getType().getTotalBits() - 1);
    }
    public static DFEVar getIsOneSet(DFEVar x, KernelBase<?> base) {
        return getSignBit(x, base) ^ x.get(x.getType().getTotalBits() - 2);
    }
    public static List<List<DFEVar>> getBitChunks128(DFEVar a, DFEVar b, boolean isSub, boolean sizeOnly)
    {
        final int CHUNKSIZE = 128;
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType();
        DFEFix resultType = getAddExactType(aType, bType, isSub);
        if (resultType.getTotalBits() <= CHUNKSIZE) return null; //no chunking is needed       
        int lsb = -Math.min(aType.getFractionBits(), bType.getFractionBits()); //minimum to account for splicing that should already have occurred
        int msb = Math.max(aType.getIntegerBits(), bType.getIntegerBits());
        List<List<DFEVar>> chunks = new ArrayList<>();
        for (int i = lsb; i < msb; ) { //going from least to most significant byte order might miss special optimizations for binary/tri-adder, very difficult to handle generally 
            int numbits = 127; //numbits will be 126 or 127 depending on subtraction, signedness of result
            List<DFEFix> types = new ArrayList<>();
            do {
                int j = i + numbits;
                Pair<Integer, Integer> ap = intersectRange(-aType.getFractionBits(), aType.getIntegerBits(), i, j);
                Pair<Integer, Integer> bp = intersectRange(-bType.getFractionBits(), bType.getIntegerBits(), i, j);
                if (ap != null) {
                    boolean consumed = aType.getIntegerBits() == ap.second;
                    types.add(KernelBase.dfeFixOffset(ap.second - ap.first, ap.first, consumed ? aType.getSignMode() : SignMode.UNSIGNED));
                } else types.add(null);
                if (bp != null) {
                    boolean consumed = bType.getIntegerBits() == bp.second;
                    types.add(KernelBase.dfeFixOffset(bp.second - bp.first, bp.first, consumed ? bType.getSignMode() : SignMode.UNSIGNED));
                } else types.add(null);
                if (types.get(0) != null && types.get(1) != null) {
                    if (getAddExactType(types.get(0), types.get(1), isSub).getTotalBits() <= CHUNKSIZE) break;
                } else break;
                numbits--;
                types.clear();
            } while (true);
            List<DFEVar> vars = new ArrayList<>();
            if (!sizeOnly) {
                if (types.get(0) == null) vars.add(null);
                else vars.add(a.slice(aType.getFractionBits() - types.get(0).getFractionBits(), types.get(0).getTotalBits()).reinterpret(types.get(0)));
                if (types.get(1) == null) vars.add(null);
                else vars.add(b.slice(bType.getFractionBits() - types.get(1).getFractionBits(), types.get(1).getTotalBits()).reinterpret(types.get(1)));
            }
            chunks.add(vars);
            i += numbits;
        }
        return chunks;
    }
    public static void verifyAddExact(DFEVar verify, DFEVar a, DFEVar b, boolean isSub, KernelBase<?> base) {
        DFEFix resultType = getAddExactType((DFEFix) a.getType(), (DFEFix) b.getType(), isSub);
        base.optimization.pushFixOpMode(Optimization.bitSizeExact(resultType.getTotalBits()),
            Optimization.offsetExact(-resultType.getFractionBits()), MathOps.ADD_SUB);
        //base.optimization.pushEnableBitGrowth(true);
        DFEVar result = isSub ? (a - b) : (a + b);
        base.optimization.popFixOpMode(MathOps.ADD_SUB);
        //base.optimization.popEnableBitGrowth();
        result = isSub && ((DFEFix)result.getType()).getSignMode() == SignMode.UNSIGNED ? result.reinterpret(resultType) : result;
        base.debug.simPrintf(verify.cast(result.getType()) !== result, "%X %X " + isSub + "\n", verify.reinterpret(KernelBase.dfeUInt(verify.getType().getTotalBits())), result.reinterpret(KernelBase.dfeUInt(result.getType().getTotalBits())));
    }
    public static DFEVar addExact(DFEVar a, DFEVar b, boolean isSub, KernelBase<?> base) {
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType();
        DFEFix resultType = getAddExactType(aType, bType, isSub);
        //first non-overlapping variables are handled
        if (-aType.getFractionBits() > bType.getIntegerBits() && !isSub && bType.getSignMode() == SignMode.UNSIGNED) {
            DFEVar result = a.cat(base.constant.zero(KernelBase.dfeUInt(-aType.getFractionBits() - bType.getIntegerBits()))).cat(b).reinterpret(resultType);
            return result;
        } else if (-bType.getFractionBits() > aType.getIntegerBits() && !isSub && aType.getSignMode() == SignMode.UNSIGNED) {
            DFEVar result = b.cat(base.constant.zero(KernelBase.dfeUInt(-bType.getFractionBits() - aType.getIntegerBits()))).cat(a).reinterpret(resultType);
            return result;
        }
        //first we splice off any trailing bits
        int bitDiff = aType.getFractionBits() - bType.getFractionBits();
        if (bitDiff > 0) {
            if (aType.getTotalBits() <= bitDiff) throw new IllegalArgumentException();
            DFEVar asplice = a.slice(0, bitDiff);
            //DFEVar aOrig = a;
            a = a.slice(bitDiff, aType.getTotalBits() - bitDiff).reinterpret(KernelBase.dfeFixOffset(aType.getTotalBits() - bitDiff, -(aType.getFractionBits() - bitDiff), aType.getSignMode()));
            List<List<DFEVar>> chunks = getBitChunks128(a, b, isSub, true);
            DFEVar result = doAddExact(a, b, isSub, base);
            result = result.cat(repeatPipeline(asplice, 1 + (chunks != null ? chunks.size() - 1 : 0), base)).reinterpret(resultType);
            //verifyAddExact(result, aOrig, b, isSub, base);
            return result;
        } else if (bitDiff < 0 && !isSub) {
            if (bType.getTotalBits() <= -bitDiff) throw new IllegalArgumentException();
            DFEVar bsplice = b.slice(0, -bitDiff);
            //DFEVar bOrig = b;
            b = b.slice(-bitDiff, bType.getTotalBits() + bitDiff).reinterpret(KernelBase.dfeFixOffset(bType.getTotalBits() + bitDiff, -(bType.getFractionBits() + bitDiff), bType.getSignMode()));            
            List<List<DFEVar>> chunks = getBitChunks128(a, b, isSub, true);
            DFEVar result = doAddExact(a, b, isSub, base);
            result = result.cat(repeatPipeline(bsplice, 1 + (chunks != null ? chunks.size() - 1 : 0), base)).reinterpret(resultType);
            //verifyAddExact(result, a, bOrig, isSub, base);
            return result;
        } else {
            DFEVar result = doAddExact(a, b, isSub, base);
            //verifyAddExact(result, a, b, isSub, base);
            return result;
        }
    }
    public static DFEFix getAddExactType(DFEFix aType, DFEFix bType, boolean isSub) {
        int fracBits = Math.max(aType.getFractionBits(), bType.getFractionBits());
        //n-bit unsigned 0 to 2^n -- m-1 bit signed -2^m to 2^m-1 - carry plus sign if n>m in all forms of addition and subtraction
        //addition range: -2^m to 2^n+2^m-1
        //subtraction range unsigned-signed: -2^m+1 to 2^n+2^m
        //subtraction range signed-unsigned: -2^m-2^n to 2^m-1 
        boolean growSign = aType.getSignMode() == SignMode.UNSIGNED && bType.getSignMode() == SignMode.TWOSCOMPLEMENT && aType.getIntegerBits() >= bType.getIntegerBits() ||
                            bType.getSignMode() == SignMode.UNSIGNED && aType.getSignMode() == SignMode.TWOSCOMPLEMENT && aType.getIntegerBits() <= bType.getIntegerBits();
        return KernelBase.dfeFixOffset(1 + (growSign ? 1 : 0) + Math.max(aType.getIntegerBits(), bType.getIntegerBits()) +
                fracBits, -fracBits, isSub || aType.getSignMode() == SignMode.TWOSCOMPLEMENT || bType.getSignMode() == SignMode.TWOSCOMPLEMENT ? SignMode.TWOSCOMPLEMENT : SignMode.UNSIGNED);
    }
    public static DFEVar doAddExact(DFEVar a, DFEVar b, boolean isSub, KernelBase<?> base) {
        DFEFix resultType = getAddExactType((DFEFix) a.getType(), (DFEFix) b.getType(), isSub);
        List<List<DFEVar>> chunks = getBitChunks128(a, b, isSub, false);
        if (chunks != null) {
            //by reinterpreting at least one of the inner add/subs and the outer add/sub, the tri-arith optimization should be avoided 
            List<DFEVar> results = new ArrayList<>();
            for (List<DFEVar> batch : chunks) {
                if (batch.get(0) != null && batch.get(1) != null) {
                    results.add(doAddExact(repeatPipeline(batch.get(0), batch == chunks.get(1) ? 1 : 0, base), repeatPipeline(batch.get(1), batch == chunks.get(1) ? 1 : 0, base), isSub, base)); 
                } else if (batch.get(0) != null) results.add(base.optimization.pipeline(batch.get(0)));
                else if (batch.get(1) != null) results.add(base.optimization.pipeline(batch.get(1)));
            }
            DFEVar result = results.get(0);
            int i;
            for (i = 1; i < results.size(); i++) result = addExact(repeatPipeline(results.get(i), i-1, base), result, false, base);
            result = result.cast(resultType); //an extra bit will occur in chained operations, so it is eliminated here as we know the carry only occurred in the chunk add or chain stage, not both
            //verifyAddExact(result, a, b, isSub, base);
            return result; 
        }
        base.optimization.pushFixOpMode(Optimization.bitSizeExact(resultType.getTotalBits()),
            Optimization.offsetExact(-resultType.getFractionBits()), MathOps.ADD_SUB);
        //base.optimization.pushEnableBitGrowth(true);
        if (isSub && ((DFEFix)a.getType()).getSignMode() == SignMode.UNSIGNED && ((DFEFix)b.getType()).getSignMode() == SignMode.UNSIGNED) {
            a = uToSigned(a, base);
            b = uToSigned(b, base);
        }
        DFEVar result = isSub ? (a - b) : (a + b);
        base.optimization.popFixOpMode(MathOps.ADD_SUB);
        //base.optimization.popEnableBitGrowth();
        return result; //isSub && ((DFEFix)result.getType()).getSignMode() == SignMode.UNSIGNED ? result.reinterpret(resultType) : result;
    }
    public static DFEVar doAddExact(DFEVar a, DFEVar b, DFEVar isSub, KernelBase<?> base) {
        DFEFix resultType = getAddExactType((DFEFix) a.getType(), (DFEFix) b.getType(), true);
        if (((DFEFix)b.getType()).getSignMode() == SignMode.UNSIGNED) b = uToSigned(b, base);
        DFEVar result = triAddExact(a, 
            b ^ Bitops.catLsbToMsb(Collections.nCopies(b.getType().getTotalBits(), isSub)).reinterpret(b.getType()),
            isSub.reinterpret(KernelBase.dfeFixOffset(1, -((DFEFix)b.getType()).getFractionBits(), SignMode.UNSIGNED)), false, false, base).cast(resultType);
        /*
        base.optimization.pushFixOpMode(Optimization.bitSizeExact(resultType.getTotalBits()),
            Optimization.offsetExact(-resultType.getFractionBits()), MathOps.ADD_SUB);
        //base.optimization.pushEnableBitGrowth(true);
        if (((DFEFix)a.getType()).getSignMode() == SignMode.UNSIGNED) a = uToSigned(a, base);
        if (((DFEFix)b.getType()).getSignMode() == SignMode.UNSIGNED) b = uToSigned(b, base);
        DFEVar result = (isSub ? base.optimization.pipeline(a-b) : base.optimization.pipeline(a+b)); //NodeCondAddSub
        base.optimization.popFixOpMode(MathOps.ADD_SUB);*/
        //base.optimization.popEnableBitGrowth();
        return result; //((DFEFix)result.getType()).getSignMode() == SignMode.UNSIGNED ? result.reinterpret(resultType) : result;
    }
    public static DFEVar mulExact(DFEVar a, DFEVar b, KernelBase<?> base) {
        DFEFix aType = (DFEFix) a.getType(), bType = (DFEFix) b.getType();    
        return mulExact(a, b, getMulOutpBits(aType, bType), getMulOutpBits(aType, bType) - aType.getFractionBits() - bType.getFractionBits(), base);
    }
    public static DFEVar mulExact(DFEVar a, DFEVar b, int bits, int intBits, KernelBase<?> base) {
        base.optimization.pushFixOpMode(Optimization.bitSizeExact(bits),
            Optimization.offsetExact(-bits + intBits), MathOps.MUL);
        //System.out.println(a.getType().getTotalBits() + "x" + b.getType().getTotalBits() + "=" + getDSPCount(a.getType().getTotalBits(), b.getType().getTotalBits()) + 
        //    " " + "Karatsuba=" + getSignedMulKaratsubaDSPCount(a.getType().getTotalBits(), b.getType().getTotalBits()));
        DFEVar result = a * b;
        base.optimization.popFixOpMode(MathOps.MUL);
        return result;
    }
    public static DFEVar doShift(DFEVar value, DFEVar shift, int maxShift, int limit, boolean isLeft, boolean isOneHot, KernelBase<?> base) //limit==32, at 1024 we would need 2 stage pipeline tree...
    {
        int bits = value.getType().getTotalBits();
        if (!isOneHot)
            shift = oneHotEncode(shift, maxShift, limit, base).reinterpret(KernelBase.dfeUInt(maxShift));
        DFEVar res = null;
        for (int j = 0; j < (bits+limit-1) / limit; j++) {
            int totbits = bits<(j+1)*limit ? bits : (j+1)*limit;
            int curbits = bits-j*limit<limit ? bits-j*limit : limit;
            int b = totbits > maxShift ? totbits-maxShift : 0;
            totbits = totbits > maxShift ? maxShift : totbits;
            DFEVar[] valueFan = totbits <= limit ? new DFEVar[] { base.optimization.pipeline(value.slice(isLeft ? b : bits-b-totbits, totbits))} : parallelFanout(value.slice(isLeft ? b : bits-b-totbits, totbits), totbits, limit, base);
            List<DFEVar> vars = new ArrayList<DFEVar>();
            for (int i = 0; i < totbits; i++) {
                if (isLeft) {
                    base.optimization.pushRoundingMode(RoundingMode.TRUNCATE);
                    vars.add(valueFan[i/limit].reinterpret(KernelBase.dfeFixOffset(totbits, i-j*limit, SignMode.UNSIGNED)).cast(KernelBase.dfeUInt(curbits)));
                    base.optimization.popRoundingMode(); 
                } else {
                    vars.add(valueFan[i/limit].slice(i, totbits - i).reinterpret(KernelBase.dfeUInt(totbits - i)).cast(KernelBase.dfeUInt(curbits)));
                }
            }
            DFEVar val = base.control.oneHotMux(shift.slice(0, totbits), vars);
            if (j == 0) { res = val;
            } else if (isLeft) { res = val.cat(res);
            } else { res = res.cat(val); }
        }
        return res;
    }
    public static DFEVar barrelShifter(DFEVar value, DFEVar shift, boolean isLeft, KernelBase<?> base)
    {
        final int muxPipeliningLimit = 3;
        value = base.optimization.pipeline(value);
        //LUT6 allows 4:1 MUX, LUT6_2 allows 2-bit 2:1 MUX
        int tot = shift.getType().getTotalBits();
        for (int i = 0; i < tot; i+=2) {
            if (i % (muxPipeliningLimit * 2) == (muxPipeliningLimit * 2 - 2) || i+2 >= tot) base.optimization.pushNoPipelining();
            if (i == tot-1) {
                value = base.control.mux(shift.get(i), value, isLeft ? value << (1<<i) : value >> (1<<i)); //1+2=LUT3 or 1+2+1+2=LUT6_2
            } else {
                value = base.control.mux(shift.slice(i, 2), value, isLeft ? value << (1<<i) : value >> (1<<i),
                    isLeft ? value << (1<<(i+1)) : value >> (1<<(i+1)), isLeft ? value << ((1<<i)+(1<<(i+1))) : value >> ((1<<i)+(1<<(i+1)))); //2+4=LUT6
            }
            if (i % (muxPipeliningLimit * 2) == (muxPipeliningLimit * 2 - 2) || i+2 >= tot) base.optimization.popNoPipelining();
        }
        return value;
    }
    public static DFEVar trailing1mask(DFEVar var, KernelBase<?> base)
    {
        //trailing1detect(x) = -x & x = (~x+1) & x
        //trailing1mask(x) = -x | x = (~x+1) | x
        //base.optimization.pushNoPipelining();
        DFEVar negVar = (~var+1);
        //base.optimization.popNoPipelining();
        int bits = var.getType().getTotalBits();
        //return negVar | var;
        return (negVar.slice(bits/2, bits-bits/2) | var.slice(bits/2, bits-bits/2)).cat(negVar.slice(0, bits/2) | var.slice(0, bits/2));
    }
    public static class NodeKeep extends Node implements OptionallyInlinedLogicNode, ConstantFold
    {
        public NodeKeep(final PhotonDesignData photonDesignData, final String[] array) {
            super(photonDesignData, array);
            this.addInput("a");
            this.addOutput("result");            
        }

        @Override
        public Optimization.PipelinedOps getNodeType() {
            return Optimization.PipelinedOps.MISC;
        }

        @Override
        public VarTyped<NodeKeep> connectOutput(final String s) {
            return this._connectOutput(this, s);
        }
        
        @Override
        protected void assignOutputTypes() {
            final InputDesc inputDesc = this.getInputDesc("a");
            final HWType hwType = inputDesc.getSrcType();
            this.setOutputDesc("result", hwType, this.isResultRegistered() ? 1 : 0);
        }

        @Override
        public Set<ControlSignal> getControlSignals() {
            if (!this.isResultRegistered()) return Collections.emptySet();
            if (this.getOutputDesc("result").getVar().getResetValue() != null) {
                return EnumSet.of(ControlSignal.CLOCK, ControlSignal.CLOCK_ENABLE, ControlSignal.RESET);
            }
            return EnumSet.of(ControlSignal.CLOCK, ControlSignal.CLOCK_ENABLE);
        }
        
        @Override
        public Entity make() {
            throw new PhotonException("make() not implemented", new Object[0]);
        }
        
        @Override
        public void makeSimCode(final CodeContext codeContext) {
            final ExpInput expInput = codeContext.getInputVar("a");
            final COutput cOutput = codeContext.getOutputVar("result");
            final HWType hwType = cOutput.getType().getHWType();
            Expression expression = expInput;
            if (!expInput.getType().isSoftwareType() && !expInput.getType().getHWType().equals(hwType)) {
                expression = expInput.cast(new SimCodeType(hwType));
            }
            final Bits bits = this.getOutputDesc("result").getVar().getResetValue();
            if (bits != null) {
                final CodeBlockRoot codeBlockRoot = codeContext.getResetBlock();
                codeBlockRoot.output(cOutput, codeBlockRoot.constant(bits, cOutput.getType()));
            }
            final CodeBlockRoot codeBlockRoot2 = codeContext.getExecuteBlock();
            codeBlockRoot2.output(cOutput, codeBlockRoot2.eval(expression));
        }
        
        @Override
        protected EntityResourceUsage getEstimatedResourceCount() {
            final int totalBits = this.getInputDesc("a").getType().getTotalBits();
            return new EntityResourceUsage(totalBits, this.isResultRegistered() ? totalBits : 0, 0, 0);
        }
        
        private boolean isResultRegistered() {
            return this.getOperatorSupplier().getSquashFactor(Optimization.PipelinedOps.MISC) == 0.0;
        }
        
        @Override
        public boolean canResetOutput(final OutputDesc outputDesc) {
            return outputDesc == this.getOutputDesc("result") && this.isResultRegistered();
        }
        
        @Override
        public Map<String, LogicSource> makeInlineLogic(final Map<String, LogicSource> map, final EntityStructural entityStructural) {
            Signal signal = entityStructural.signal(map.get("a"));  
            LogicSource logicSource = null;
            if (this.isResultRegistered()) {
                final Reg reg = entityStructural.reg(signal);
                reg.setKeepSynthesis(); //Vivado KEEP
                //reg.setKeepImplementation(); //Vivado DONT_TOUCH prevents LUTNMs!
                final Bits bits = this.getOutputDesc("result").getVar().getResetValue();
                if (bits != null) {
                    reg.setResetValue(entityStructural.constant(bits));
                }
                logicSource = reg;
            } else {
                signal.setKeepSynthesis(); //Vivado KEEP
                //signal.setKeepImplementation(); //Vivado DONT_TOUCH prevents LUTNMs!
                logicSource = signal;
            }
            entityStructural.setKeepHierarchy(KeepHierarchy.FALSE);
            return Collections.singletonMap("result", logicSource);
        }
        
        @Override
        public String getUserSignature() {
            return "PhotonKeep_" + this.getInputDesc("a").getType().getTotalBits() + ((this.getOperatorSupplier().getSquashFactor(this.getNodeType()) == 0.0) ? "pipe" : "nopipe");
        }
        
        @Override
        public void visitKernelGraph(final KernelGraphVisitor kernelGraphVisitor) {
            kernelGraphVisitor.visitKernelGraph(this);
        }
        
        @Override
        public <R extends Enum<R> & ResourceComponent> void visitKernelUsageEstimation(final KernelEstimatorVisitor<R> kernelEstimatorVisitor) {
            kernelEstimatorVisitor.visit(this);
        }
       
        @Override
        public NodeConstant foldOperation() {
            final NodeConstant nodeConstant = this.tryReplaceByConstant();
            if (nodeConstant != null) {
                return nodeConstant;
            }
            final ConstantInput constantInput = this.findSingleConstantInput();
            if (constantInput == null) {
                return null;
            }
            final Bits bits = constantInput.node.getValueAsBits(constantInput.type);
            if (bits.isAllZeros()) {
                return this.replaceWithConstant(bits);
            }
            return null;
        }
               
        @Override
        public String toString() {
            return "KEEP";
        }
        
        protected NodeConstant tryReplaceByConstant() {
            final NodeConstant nodeConstant = this.getInputAsConstant("a");
            if (nodeConstant == null) {
                return null;
            }
            return this.replaceWithConstant(nodeConstant.getValueAsBits(this.getInputDesc("a").getType()));
        }        
        protected ConstantInput findSingleConstantInput() {
            String s = "a";
            NodeConstant nodeConstant = this.getInputAsConstant("a");
            HWType hwType = this.getInputDesc("a").getType();
            if (nodeConstant == null) {
                return null;
            }
            return new ConstantInput(s, nodeConstant, hwType);
        }
        protected NodeConstant replaceWithConstant(final Bits bits) {
            return this.replaceWithConstant("result", bits);
        }

        protected static class ConstantInput
        {
            public final String input;
            public final NodeConstant node;
            public final HWType type;
            
            public ConstantInput(final String input, final NodeConstant node, final HWType type) {
                this.input = input;
                this.node = node;
                this.type = type;
            }
        }        
    }
    public static DFEVar setKeep(DFEVar var, boolean keepEnabled)
    {
        if (keepEnabled) {
            NodeKeep node = new NodeKeep(var.getOwner().getPhotonDesignData(), var.getOwner().getPhotonDesignData().getGroupPath());
            try {
                Field f = var.getClass().getDeclaredField("m_imp");
                f.setAccessible(true);
                node.connectInput("a", (Var)f.get(var));
            } catch (NoSuchFieldException|IllegalAccessException e) { }
            return new DFEVar(var.getOwner(), node.connectOutput("result"));
        } else {
            return var;
        }
    }
    //https://e-archivo.uc3m.es/bitstream/handle/10016/34413/efficient_IEEE-ESL_2022_ps.pdf
    public static Pair<DFEVar, DFEVar> leading0count(DFEVar var, KernelBase<?> base, boolean oldMethod, boolean keep, boolean altMethod)
    {
        int size = var.getType().getTotalBits();
        //construct 8-bit LZC
        if (oldMethod) keep = true;
        List<DFEVar> LP3s = new ArrayList<>(), LP2s = new ArrayList<>(), LP1_ints = new ArrayList<>();
        List<DFEVar> LP1s = new ArrayList<>(), LP4s = new ArrayList<>();
        base.optimization.pushNoPipelining();
        for (int i = size; i > 0; i -= 8) {
            int curSize = Math.min(i, 8);
            DFEVar x = var.slice(Math.max(0, i-8), curSize);
            if (size <= 8 || oldMethod || altMethod || (LP4s.size() & 1) == 0 || curSize < 7)
                LP3s.add(curSize <= 2 ? null : setKeep((curSize >= 4 ? x.slice(curSize-4, 4) : x) === 0, keep));
            LP2s.add(curSize <= 1 ? null :
                setKeep((curSize <= 4 ? x.slice(curSize-2, 2) === 0 :
                x.slice(curSize-2, 2) === 0 &
                    (x.slice(curSize-4, 2) !== 0 | (curSize >= 6 ? x.slice(curSize-6, 2) === 0 : ~x.get(curSize-5)))), keep));
            if (size > 8 && !oldMethod && !altMethod && ((LP1_ints.size() & 1) != 0) && curSize >= 7) {
                LP1_ints.add(setKeep(
                    ~var.get(i+1) & (var.get(i) | ~x.get(curSize-1) & (x.get(curSize-2) | ~x.get(curSize-3))), keep)); // & x.get(curSize-4)
            } else {
                LP1_ints.add(setKeep(
                    curSize > 6 && (size <= 8 || oldMethod || altMethod) ? ~x.get(curSize-1) & (x.get(curSize-2) | ~x.get(curSize-3) & (x.get(curSize-4) | ~x.get(curSize-5) & x.get(curSize-6))) :
                    curSize >= 5 ? ~x.get(curSize-1) & (x.get(curSize-2) | ~x.get(curSize-3) & (x.get(curSize-4) | ~x.get(curSize-5))) :
                    curSize >= 3 ? ~x.get(curSize-1) & (x.get(curSize-2) | ~x.get(curSize-3)) :
                    ~x.get(curSize-1), keep));
            }
            if (size <= 8 || oldMethod) {
                LP4s.add(curSize <= 4 ? null :
                    setKeep(curSize <= 6 ? x === 0 :
                    LP3s.get(LP3s.size()-1) & LP2s.get(LP2s.size()-1) & ~LP1_ints.get(LP1_ints.size()-1) & (curSize <= 7 ? ~x.get(curSize-7) : x.slice(curSize-8, 2) === 0), size > 8 && keep));
                LP1s.add(curSize <= 6 ? LP1_ints.get(LP1_ints.size()-1) :
                    setKeep(LP1_ints.get(LP1_ints.size()-1) | LP3s.get(LP3s.size()-1) & LP2s.get(LP2s.size()-1) & ~x.get(curSize-7), size > 8 && keep));
                if (size <= 8) {
                    base.optimization.popNoPipelining();
                    return new Pair<DFEVar, DFEVar>(base.optimization.limitFanout(
                        size <= 1 ? LP1s.get(0).reinterpret(KernelBase.dfeBool()) :
                        size <= 2 ? LP2s.get(0).reinterpret(KernelBase.dfeBool()) :
                        size <= 4 ? LP3s.get(0).reinterpret(KernelBase.dfeBool()) : LP4s.get(0).reinterpret(KernelBase.dfeBool()), 32),
                        size <= 1 ? null : base.optimization.limitFanout(
                        size <= 2 ? LP1s.get(0).reinterpret(KernelBase.dfeBool()) :
                        size <= 4 ? LP2s.get(0).cat(LP1s.get(0)).reinterpret(KernelBase.dfeUInt(2)) :
                        LP3s.get(0).cat(LP2s.get(0)).cat(LP1s.get(0)).reinterpret(KernelBase.dfeUInt(3)), 32));
                }
            } else {
                if (!altMethod && (LP4s.size() & 1) != 0 && curSize >= 7) { //low part of intermediate requires shifted calculation or Z0s requires 7 bits
                    LP4s.add(setKeep(var.slice(i, 2).cat(x.slice(curSize-3, 3)) === 0, keep));
                    LP1s.add(x.slice(0, curSize-3));
                    LP3s.add(LP4s.get(LP4s.size()-1) & ~x.get(curSize-4));
                } else {
                    LP4s.add(setKeep((curSize >= 6 ? x.slice(curSize-6, 6) : x) === 0, keep));
                    LP1s.add(curSize <= 6 ? null : curSize <= 7 ? x.get(curSize-7) : x.slice(curSize-8, 2));
                }
            }
        }
        List<DFEVar> V = new ArrayList<>(), Z0s = new ArrayList<>(), Z1s = new ArrayList<>(), Z2s = new ArrayList<>(), Z3s = new ArrayList<>();
        for (int i = 0; i < LP4s.size(); i += 2) {
            DFEVar VH;
            if (oldMethod) {
                if (size <= 16) keep = false;
                VH = LP4s.get(i) == null ? (LP3s.get(i) == null ? (LP2s.get(i) == null ? LP1s.get(i).reinterpret(KernelBase.dfeBool()) : LP2s.get(i)) : LP3s.get(i)) : LP4s.get(i);
                V.add(i+1==LP4s.size() ? VH : setKeep(VH & (LP4s.get(i+1) == null ? (LP3s.get(i+1) == null ? (LP2s.get(i+1) == null ? LP1s.get(i+1) : LP2s.get(i+1)) : LP3s.get(i+1)) : LP4s.get(i+1)), keep));
                Z0s.add(setKeep(VH ? (i+1==LP1s.size() ? base.constant.var(KernelBase.dfeBool(), 1) : LP1s.get(i+1).reinterpret(KernelBase.dfeBool())) : LP1s.get(i).reinterpret(KernelBase.dfeBool()), keep));
                Z1s.add(setKeep(VH ? (i+1==LP2s.size() ? base.constant.var(KernelBase.dfeBool(), 1) : (LP2s.get(i+1) == null ? LP1s.get(i+1) : LP2s.get(i+1)).reinterpret(KernelBase.dfeBool())) : (LP2s.get(i) == null ? LP1s.get(i) : LP2s.get(i)).reinterpret(KernelBase.dfeBool()), keep));
                Z2s.add(setKeep(VH ? (i+1==LP3s.size() ? base.constant.var(KernelBase.dfeBool(), 1) : (LP3s.get(i+1) == null ? (LP2s.get(i+1) == null ? LP1s.get(i+1) : LP2s.get(i+1)) : LP3s.get(i+1)).reinterpret(KernelBase.dfeBool())) : (LP3s.get(i) == null ? (LP2s.get(i) == null ? LP1s.get(i) : LP2s.get(i)) : LP3s.get(i)).reinterpret(KernelBase.dfeBool()), keep));
            } else {
                VH = LP1s.get(i) == null ? LP4s.get(i) : LP4s.get(i) & LP1s.get(i) === 0;
                boolean defaultLogic = i+1==LP4s.size() || LP1s.get(i+1) == null || LP1s.get(i+1).getType().getTotalBits() < 3;
                if (defaultLogic) {
                    if (size <= 16) keep = false;
                    V.add(i+1==LP4s.size() ? VH : setKeep(LP4s.get(i) & LP4s.get(i+1) & (LP1s.get(i+1) == null ? LP1s.get(i) : LP1s.get(i).cat(LP1s.get(i+1))) === 0, keep));
                    if (size <= 14) Z0s.add(setKeep(VH ? (i+1==LP4s.size() ? base.constant.var(KernelBase.dfeBool(), 1) : (LP1s.get(i+1) == null ? LP1_ints.get(i+1) : LP1_ints.get(i+1) | LP4s.get(i+1)).reinterpret(KernelBase.dfeBool())) : (LP1s.get(i) == null ? LP1_ints.get(i) : LP1_ints.get(i) & ~LP4s.get(i) | LP4s.get(i) & ~LP1s.get(i).get(LP1s.get(i).getType().getTotalBits()-1)).reinterpret(KernelBase.dfeBool()), keep));
                    else Z0s.add(setKeep(VH ? (i+1==LP4s.size() ? base.constant.var(KernelBase.dfeBool(), 1) : (LP1s.get(i+1) == null ? LP1_ints.get(i+1) : LP1_ints.get(i+1) | LP4s.get(i+1) & ~LP1s.get(i+1).get(LP1s.get(i+1).getType().getTotalBits()-1)).reinterpret(KernelBase.dfeBool())) : (LP1s.get(i) == null ? LP1_ints.get(i) : LP1_ints.get(i) & ~LP4s.get(i) | LP4s.get(i) & ~LP1s.get(i).get(LP1s.get(i).getType().getTotalBits()-1)).reinterpret(KernelBase.dfeBool()), keep));
                } else {
                    DFEVar allZero = setKeep(LP1s.get(i+1) === 0, keep);
                    int l = LP1s.get(i+1).getType().getTotalBits();
                    DFEVar parity = setKeep(LP1s.get(i+1).get(l-1) | ~LP1s.get(i+1).get(l-2) & (LP1s.get(i+1).get(l-3) | ~LP1s.get(i+1).get(l-4)), keep);
                    if (size <= 16) keep = false;
                    V.add(i+1==LP4s.size() ? VH : setKeep(LP1s.get(i+1) == null ? LP4s.get(i) & LP4s.get(i+1) : LP4s.get(i) & LP4s.get(i+1) & allZero, keep));
                    Z0s.add(setKeep((LP4s.get(i) ? (LP4s.get(i+1) ? parity : LP1_ints.get(i+1)) : LP1_ints.get(i)).reinterpret(KernelBase.dfeBool()), keep));
                }
                Z1s.add(setKeep(VH ? (i+1==LP4s.size() ? base.constant.var(KernelBase.dfeBool(), 1) : (LP2s.get(i+1) == null ? LP1_ints.get(i+1).reinterpret(KernelBase.dfeBool()) : LP2s.get(i+1))) :
                    (LP2s.get(i) == null ? LP1_ints.get(i).reinterpret(KernelBase.dfeBool()) : LP2s.get(i)), keep));
                Z2s.add(setKeep(VH ? (i+1==LP4s.size() ? base.constant.var(KernelBase.dfeBool(), 1) : (LP3s.get(i+1) == null ? (LP2s.get(i+1) == null ? LP1_ints.get(i+1).reinterpret(KernelBase.dfeBool()) : LP2s.get(i+1)) : LP3s.get(i+1))) :
                    (LP3s.get(i) == null ? (LP2s.get(i) == null ? LP1_ints.get(i).reinterpret(KernelBase.dfeBool()) : LP2s.get(i)) : LP3s.get(i)), keep));
                if (LP1s.get(i) != null) VH = setKeep(VH, keep);
            }
            Z3s.add(VH);
        }
        List<List<DFEVar>> Zs = new ArrayList<>();
        int total = MathUtils.ceilLog2(size);
        for (int c = 0; c < total-4; c++) {
            if (c == total-4-1) keep = false;
            Zs.add(new ArrayList<>());
            for (int i = 0; i < V.size(); i++) {
                Z0s.set(i, setKeep(V.get(i) ? (i+1==V.size() ? base.constant.var(KernelBase.dfeBool(), 1) : Z0s.get(i+1)) : Z0s.get(i), keep));
                Z1s.set(i, setKeep(V.get(i) ? (i+1==V.size() ? base.constant.var(KernelBase.dfeBool(), 1) : Z1s.get(i+1)) : Z1s.get(i), keep));
                Z2s.set(i, setKeep(V.get(i) ? (i+1==V.size() ? base.constant.var(KernelBase.dfeBool(), 1) : Z2s.get(i+1)) : Z2s.get(i), keep));
                Z3s.set(i, setKeep(V.get(i) ? (i+1==V.size() ? base.constant.var(KernelBase.dfeBool(), 1) : Z3s.get(i+1)) : Z3s.get(i), keep));
                for (int j = 0; j < c; j++) {
                    Zs.get(j).set(i, setKeep(V.get(i) ? (i+1==V.size() ? base.constant.var(KernelBase.dfeBool(), 1) : Zs.get(j).get(i+1)) : Zs.get(j).get(i), keep));
                }
                Zs.get(c).add(V.get(i));
                V.set(i, i+1==V.size() ? V.get(i) : setKeep(V.get(i) & V.get(i+1), keep));
                if (i+1!=V.size()) { V.remove(i+1); Z0s.remove(i+1); Z1s.remove(i+1); Z2s.remove(i+1); Z3s.remove(i+1);
                    for (int j = 0; j < c; j++) { Zs.get(j).remove(i+1); }
                }
            }
        }
        base.optimization.popNoPipelining();
        return new Pair<DFEVar, DFEVar>(base.optimization.limitFanout(V.get(0), 32),
            base.optimization.limitFanout(
                (Zs.size() != 0 ? Bitops.catLsbToMsb(Zs.stream().map(x -> x.get(0)).collect(Collectors.toList())).cat(
                    Z3s.get(0)) : Z3s.get(0)).cat(Z2s.get(0)).cat(Z1s.get(0)).cat(Z0s.get(0)).reinterpret(KernelBase.dfeUInt(total)), 32));
    }
    //static boolean floatDelay = true;
    public static DFEVector<DFEComplex> doFloatAdd(DFEVector<DFEComplex> vec1, DFEVector<DFEComplex> vec2, boolean isSub, KernelBase<?> base)
    {
        DFEVector<DFEComplex> res = vec1.getType().newInstance(base);
        for (int i = 0; i < vec1.getSize(); i++) {
            res[i] <== doFloatAdd(vec1[i], vec2[i], isSub, base);
        }
        return res;
    }
    public static DFEComplex doFloatAdd(DFEComplex n1, DFEComplex n2, boolean isSub, KernelBase<?> base)
    {
        return DFEComplexType.newInstance(base, doFloatAdd(n1.getReal(), n2.getReal(), isSub, base),
            doFloatAdd(n1.getImaginary(), n2.getImaginary(), isSub, base));
    }
    public static DFEVar doFloatAdd(DFEVar n1, DFEVar n2, boolean isSub, KernelBase<?> base)
    {
        //return isSub ? n1 - n2 : n1 + n2;
        //initialization: splice apart mantissa, exponent and sign
        DFEFloat n1type = (DFEFloat)n1.getType(), n2type = (DFEFloat)n2.getType();
        int n1exp = n1type.getExponentBits(), n2exp = n2type.getExponentBits();
        int n1mant = n1type.getMantissaBits(), n2mant = n2type.getMantissaBits();
        int bigMant = Math.max(n1mant, n2mant);
        DFEVar n1e = n1.slice(n1mant - 1, n1exp).reinterpret(KernelBase.dfeUInt(n1exp)), n2e = n2.slice(n2mant - 1, n2exp).reinterpret(KernelBase.dfeUInt(n2exp));
        DFEVar n1sign = n1.get(n1type.getTotalBits()-1), n2sign = n2.get(n2type.getTotalBits()-1);
        
        //pipeline stage 1: comparison, sign parity and final sign
        base.optimization.pushNoPipelining();
        DFEVar swap = n1.slice(0, n1exp + n1mant - 1).reinterpret(KernelBase.dfeUInt(n1exp+n1mant-1)) < n2.slice(0, n2exp + n2mant - 1).reinterpret(KernelBase.dfeUInt(n2exp+n2mant-1));
        base.optimization.popNoPipelining();
        base.optimization.pushFanoutLimit(Integer.MAX_VALUE, FanoutLimitType.TREE);
        swap = base.optimization.limitFanout(swap, 32);
        base.optimization.popFanoutLimit();
        DFEVar signParity = isSub ? n1sign === n2sign : n1sign !== n2sign;

        //pipeline stage 2: swapping and exponent difference, zero, inf/NaN detection
        DFEVar sign = swap ? (isSub ? ~n2sign : n2sign) : n1sign;
        DFEVar n1s = swap ? base.optimization.pipeline(n2.slice(0, n1mant - 1)) : base.optimization.pipeline(n1.slice(0, n1mant - 1));
        DFEVar n2s = swap ? base.optimization.pipeline(n1.slice(0, n1mant - 1)) : base.optimization.pipeline(n2.slice(0, n1mant - 1));
        DFEVar n1m = base.constant.var(KernelBase.dfeBool(), 1).cat(n1s).reinterpret(KernelBase.dfeFixOffset(n1mant, -n1mant+1, SignMode.UNSIGNED)).cast(KernelBase.dfeFixOffset(bigMant+2, -bigMant-2+1, SignMode.UNSIGNED)),
               n2m = base.constant.var(KernelBase.dfeBool(), 1).cat(n2s).reinterpret(KernelBase.dfeFixOffset(n2mant, -n2mant+1, SignMode.UNSIGNED)).cast(KernelBase.dfeFixOffset(bigMant+2, -bigMant-2+1, SignMode.UNSIGNED));
        //r = (v + mask) ^ mask;
        base.optimization.pushNoPipelining();
        DFEVar expDiff = (n1e ^ Bitops.catLsbToMsb(Collections.nCopies(n1exp, swap)).reinterpret(KernelBase.dfeUInt(n1exp))) -
                         (n2e ^ Bitops.catLsbToMsb(Collections.nCopies(n2exp, swap)).reinterpret(KernelBase.dfeUInt(n2exp))); //swap ? n2e-n1e : n1e-n2e
        base.optimization.popNoPipelining();
        expDiff = base.optimization.limitFanout(expDiff, 32);
        DFEVar expZero1 = n1e === 0, expZero2 = n2e === 0;
        DFEVar expInf1 = n1e === (1<<n1exp)-1, expInf2 = n2e === (1<<n2exp)-1;
        DFEVar resExp = swap ? base.optimization.pipeline(n2e) : base.optimization.pipeline(n1e);
        
        //pipeline stage 3: add guard, round bits, shifting smaller argument right, compression for "sticky" bit
        DFEVar expDiffRedux = expDiff.slice(0, MathUtils.bitsToAddress(bigMant+2)).reinterpret(KernelBase.dfeUInt(MathUtils.bitsToAddress(bigMant+2)));
        DFEVar shifted = barrelShifter(n2m, expDiffRedux.slice(0, MathUtils.bitsToAddress(bigMant)).reinterpret(KernelBase.dfeUInt(MathUtils.bitsToAddress(bigMant))), false, base);
        DFEVar zeroMant = expZero1 | expZero2 | expDiff > bigMant+1 | expInf1 | expInf2;
        DFEVar shifts[] = new DFEVar[] { base.optimization.pipeline(expDiffRedux), bigMant < 32 ? null : base.optimization.pipeline(expDiffRedux) };
        DFEVector<DFEVar> enc = new DFEVectorType<DFEVar>(KernelBase.dfeBool(), bigMant-1).newInstance(base);
        for (int i = 0; i < bigMant-1; i++) {
            enc[i] <== shifts[i / 32] > i+2;
        }
        DFEVar nearOverflow = resExp === (1<<Math.max(n1exp, n2exp))-2;
        
        //pipeline stage 4: handling zero mantissa, computing sticky bit
        //base.optimization.pushNoPipelining();
        DFEVar n2mshifted =
            (MathUtils.bitsToAddress(bigMant) != MathUtils.bitsToAddress(bigMant+2)) ?
            base.control.mux(zeroMant.cat(expDiff.get(MathUtils.bitsToAddress(bigMant+2)-1)), 
                shifted, base.optimization.pipeline(n2m>>bigMant), base.constant.zero(n2m.getType()), base.constant.zero(n2m.getType())) :  
            (zeroMant ? base.constant.zero(n2m.getType()) : shifted);
        DFEVar sticky2 = (n2m.slice(2, bigMant-1) & enc.reinterpret(KernelBase.dfeRawBits(bigMant-1))) !== 0;
        //base.optimization.popNoPipelining();
        DFEVar sticky1 = base.constant.zero(KernelBase.dfeBool());
        n1m = n1m.cat(sticky1).reinterpret(KernelBase.dfeFixOffset(bigMant+3, -bigMant-2, SignMode.UNSIGNED));
        n2mshifted = n2mshifted.cat(sticky2).reinterpret(KernelBase.dfeFixOffset(bigMant+3, -bigMant-2, SignMode.UNSIGNED)); 
        base.optimization.pushFanoutLimit(Integer.MAX_VALUE, FanoutLimitType.TREE);
        signParity = base.optimization.limitFanout(signParity, 32);
        base.optimization.popFanoutLimit();
        
        //pipeline stage 5: conditional addition/subtraction
        base.optimization.pushFanoutLimit(Integer.MAX_VALUE, FanoutLimitType.TREE);
        DFEVar sum = doAddExact(n1m, n2mshifted, signParity, base);
        base.optimization.popFanoutLimit();
            //base.control.oneHotMux((signParity & ~unevenSign).cat(signParity & unevenSign).cat(~signParity), sum, sub1, sub2);
            
        //pipeline stage 6: check zero result, count of leading zeros, rounding
        //DFEVar sumIsZero = sum === 0;
        //DFEVar oneHotShift = Bitops.trailing1Detect(Bitops.bitreverse(sum.slice(2, bigMant+2)).reinterpret(KernelBase.dfeUInt(bigMant+2))); //sum === 0
        //DFEVar expAdjust = Bitops.onehotDecode(oneHotShift); //.slice(1, bigMant+1)
        Pair<DFEVar, DFEVar> isZeroExpAdjust = leading0count(sum.slice(2, bigMant+2), base, false, false, true);
        DFEVar sumIsZero = isZeroExpAdjust.first, expAdjust = isZeroExpAdjust.second;        
        //Rounding to nearest even: with Enable, Guard, Round Sticky, g.rs-> round when 1.1x 0.11 e&r&(g|s)
        base.optimization.pushNoPipelining();
        DFEVar roundAdjust = (sum.get(bigMant+3) & (sum.get(3) & (sum.get(4) | sum.slice(0, 3)!==0))).cat(
                ~sum.get(bigMant+3) & sum.get(bigMant+2) & (sum.get(2) & (sum.get(3) | sum.slice(0, 2)!==0))).cat(
                ~sum.get(bigMant+3) & ~sum.get(bigMant+2) & sum.get(bigMant+1) & (sum.get(1) & (sum.get(2) | sum.get(0)))).reinterpret(KernelBase.dfeFixOffset(3, -bigMant, SignMode.UNSIGNED));
        base.optimization.popNoPipelining();
        DFEVar roundSum = doAddExact(sum.reinterpret(KernelBase.dfeFixOffset(bigMant+4, -bigMant-2, SignMode.UNSIGNED)),
                roundAdjust, false, base);
        base.optimization.pushNoPipelining();
        DFEVar roundOverflow = base.optimization.pipeline((sum.get(bigMant+3) & roundSum.get(bigMant+4) | ~sum.get(bigMant+3) & sum.get(bigMant+2) & roundSum.get(bigMant+3) | ~sum.get(bigMant+3) & ~sum.get(bigMant+2) & sum.get(bigMant+1) & roundSum.get(bigMant+2)).reinterpret(KernelBase.dfeBool()));
        base.optimization.popNoPipelining();
        sum = roundSum.slice(0, bigMant+4);
        /*for (int i = 2; i <= bigMant+2; i++) {
            DFEVar check = sum.slice(2, i);
            DFEVar z = check === 0;
            DFEVar ea = Bitops.onehotDecode(Bitops.trailing1Detect(Bitops.bitreverse(check).reinterpret(KernelBase.dfeUInt(i))));
            ea = ea - (z ? base.constant.var(ea.getType(), 1) : base.constant.zero(ea.getType()));
            Pair<DFEVar, DFEVar> isZeroExpAdjust = leading0count(check, base, false, false, true);
            base.debug.simPrintf(z !== isZeroExpAdjust.first | ea !== isZeroExpAdjust.second, "%d %X %X %X %X %X\n", i, check, z, ea, isZeroExpAdjust.first, isZeroExpAdjust.second);
        }*/
        //rounding is only needed if the high bit is set for full rounding, or next to high bit is set (round becomes guard, sticky becomes round bit and sticky is 0)
        //DFEVar sumshifted = oneHotShift.get(0) ? sum.reinterpret(KernelBase.dfeUInt(sum.getType().getTotalBits())) : doShift(sum << 1, oneHotShift.slice(1, bigMant), bigMant, 32, true, true, base); //0 to bigMant inclusive shift amounts, since addition adds 1, while subtraction can make the bigMant bit the highest bit set, otherwise it must be zero
        
        //pipeline stage 7: shift left, underflow detect
        DFEVar underflow = expAdjust.cast(resExp.getType()) > resExp;
        //DFEVar sumshifted = barrelShifter(sum, expAdjust, true, base);
        DFEVar sumshifted = barrelShifter(sum, expAdjust.slice(0, MathUtils.bitsToAddress(bigMant)).reinterpret(KernelBase.dfeUInt(MathUtils.bitsToAddress(bigMant))), true, base);
        base.optimization.pushNoPipelining();
        DFEVar isNaN = base.optimization.pipeline((sumIsZero | signParity) & expInf1 & expInf2);
        DFEVar zeroExp = base.optimization.pipeline(sumIsZero & ~expInf1 & ~expInf2 | underflow);
        DFEVar zeroResult = base.optimization.pipeline(underflow | nearOverflow & (roundOverflow | expAdjust===0));
        base.optimization.popNoPipelining();
        DFEVar sel = (MathUtils.bitsToAddress(bigMant) != MathUtils.bitsToAddress(bigMant+2)) ?
            (zeroResult | expAdjust.get(MathUtils.bitsToAddress(bigMant+2)-1)).cat(isNaN | expAdjust.get(MathUtils.bitsToAddress(bigMant+2)-1)) : zeroResult.cat(isNaN);
        
        //pipeline stage 8: exponent adjust and mantissa zero/NaN or big shift adjustments       
        //sumshifted = zeroResult ? base.constant.zero(KernelBase.dfeFixOffset(bigMant-1, -bigMant+1, SignMode.UNSIGNED)) : sumshifted.cast(KernelBase.dfeFixOffset(bigMant-1, -bigMant+1, SignMode.UNSIGNED));
        sumshifted = (MathUtils.bitsToAddress(bigMant) != MathUtils.bitsToAddress(bigMant+2)) ?
            base.control.mux(sel,
                sumshifted.slice(4, bigMant-1).cast(KernelBase.dfeFixOffset(bigMant-1, -bigMant+1, SignMode.UNSIGNED)),
                base.constant.var(KernelBase.dfeBool(), 1).cat(base.constant.zero(KernelBase.dfeUInt(bigMant-2))).reinterpret(KernelBase.dfeFixOffset(bigMant-1, -bigMant+1, SignMode.UNSIGNED)),                
                base.constant.zero(KernelBase.dfeFixOffset(bigMant-1, -bigMant+1, SignMode.UNSIGNED)),
                (sumshifted<<bigMant).slice(4, bigMant-1).cast(KernelBase.dfeFixOffset(bigMant-1, -bigMant+1, SignMode.UNSIGNED))) :
            base.control.mux(sel,
                sumshifted.slice(4, bigMant-1).cast(KernelBase.dfeFixOffset(bigMant-1, -bigMant+1, SignMode.UNSIGNED)),
                base.constant.var(KernelBase.dfeBool(), 1).cat(base.constant.zero(KernelBase.dfeUInt(bigMant-2))).reinterpret(KernelBase.dfeFixOffset(bigMant-1, -bigMant+1, SignMode.UNSIGNED)),
                base.constant.zero(KernelBase.dfeFixOffset(bigMant-1, -bigMant+1, SignMode.UNSIGNED)));
        resExp = zeroExp ? base.constant.zero(resExp.getType()) : (base.optimization.pipeline(resExp)+roundOverflow.cat(~roundOverflow).reinterpret(KernelBase.dfeUInt(2)).cast(resExp.getType()) - expAdjust.cast(resExp.getType()));        
        //finalization: recombine mantissa, exponent and sign
        DFEVar result = sign.cat(resExp).cat(sumshifted).reinterpret(KernelBase.dfeFloat(Math.max(n1exp, n2exp), bigMant));
        //if (floatDelay) { base.debug.simPrintf("%d %d\n", base.stream.measureDistance("floatAddDelay1", n1, result).getDFEVar(base, KernelBase.dfeUInt(4)), base.stream.measureDistance("floatAddDelay2", n2, result).getDFEVar(base, KernelBase.dfeUInt(4))); floatDelay=false; }
        /*if (isSub)
            base.debug.simPrintf(result !== n1 - n2, "sub %f %f %f %f %X %X %X %X %X %X %d %d %d %d %d\n", n1, n2, n1 - n2, result, n1.reinterpret(KernelBase.dfeUInt(79)), n2.reinterpret(KernelBase.dfeUInt(79)), (n1 - n2).reinterpret(KernelBase.dfeUInt(79)), result.reinterpret(KernelBase.dfeUInt(79)), n1mshifted.reinterpret(KernelBase.dfeUInt(bigMant+3)), n2mshifted.reinterpret(KernelBase.dfeUInt(bigMant+3)), expDiff, resExp, expAdjust, sticky1, sticky2);
        else
            base.debug.simPrintf(result !== n1 + n2, "add %f %f %f %f %X %X %X %X %X %X %d %d %d %d %d\n", n1, n2, n1 + n2, result, n1.reinterpret(KernelBase.dfeUInt(79)), n2.reinterpret(KernelBase.dfeUInt(79)), (n1 + n2).reinterpret(KernelBase.dfeUInt(79)), result.reinterpret(KernelBase.dfeUInt(79)), n1mshifted.reinterpret(KernelBase.dfeUInt(bigMant+3)), n2mshifted.reinterpret(KernelBase.dfeUInt(bigMant+3)), expDiff, resExp, expAdjust, sticky1, sticky2);*/
        //hex(int.from_bytes((np.fromstring(((0x5FFCC<<61)|0x4762E060FCDF98E5).to_bytes(length=16, byteorder='little'), dtype=np.longdouble)-np.fromstring(((0x5FFE4<<61)|0x0055CB372292309D).to_bytes(length=16, byteorder='little'), dtype=np.longdouble))[0].tobytes(), byteorder='little'))
        return result;
    }
    public static DFEVar doFloatMult(DFEVar n1, DFEVar n2, KernelBase<?> base) //default long double multiplication latency is 22, we achieve 7 and less DSPs
    {
        //initialization: splice apart mantissa, exponent and sign
        DFEFloat n1type = (DFEFloat)n1.getType(), n2type = (DFEFloat)n2.getType();
        int n1exp = n1type.getExponentBits(), n2exp = n2type.getExponentBits();
        int n1mant = n1type.getMantissaBits(), n2mant = n2type.getMantissaBits();
        int bigMant = Math.max(n1mant, n2mant);
        DFEVar n1sign = n1.get(n1type.getTotalBits()-1), n2sign = n2.get(n2type.getTotalBits()-1);
        DFEVar n1m = base.constant.var(KernelBase.dfeBool(), 1).cat(n1.slice(0, n1mant - 1)).reinterpret(KernelBase.dfeFixOffset(n1mant, -n1mant+1, SignMode.UNSIGNED)),
               n2m = base.constant.var(KernelBase.dfeBool(), 1).cat(n2.slice(0, n2mant - 1)).reinterpret(KernelBase.dfeFixOffset(n2mant, -n2mant+1, SignMode.UNSIGNED));
        DFEVar n1e = n1.slice(n1mant - 1, n1exp).reinterpret(KernelBase.dfeUInt(n1exp)), n2e = n2.slice(n2mant - 1, n2exp).reinterpret(KernelBase.dfeUInt(n2exp));
        DFEVar adjustExp = base.constant.var(KernelBase.dfeUInt(Math.min(n1exp, n2exp)-1), (1 << (Math.min(n1exp, n2exp)-1))-1);
        
        //multi-cycle pipeline stage 1: zero, inf/NaN detection, final sign, multiply, detect explicit NaN, simple underflow, overflow
        int totalDelay = n1mant >= 64 || n2mant >= 64 ? 6 : (n1mant >= 53 || n2mant >= 53 ? 5 : 4); //for long double, 24x24 (2 DSPs) has 4 delay, 53x53 has 5? delay, 64x64 has 6 delay 
        DFEVar resultMant = mulKaratsubaRectangularExact(n1m, n2m, n1mant + n2mant, 2, false, base);
        //pipeline stage 1A:
        DFEVar expInf1 = n1e === (1<<n1exp)-1, expInf2 = n2e === (1<<n2exp)-1; 
        DFEVar isZero = n1e === 0 | n2e === 0; //2 delay for comparison and bitwise or
        DFEVar sign = n1sign ^ n2sign;
        DFEVar expNormal = triAddExact(n1e, n2e, adjustExp, false, true, base);
        DFEVar mant1nz = n1m.slice(0, n1mant-1) !== 0, mant2nz = n2m.slice(0, n2mant-1) !== 0;
        //pipeline stage 1B:
        base.optimization.pushNoPipelining();
        DFEVar isNaN = base.optimization.pipeline(isZero & (expInf1 | expInf2) | mant1nz & expInf1 | mant2nz & expInf2);
        DFEVar underflow = base.optimization.pipeline(isZero & ~expInf1 & ~expInf2 | expNormal < 0);
        DFEVar overflow = base.optimization.pipeline(expInf1 | expInf2 | expNormal >= (1<<Math.max(n1exp, n2exp))-1);
        base.optimization.popNoPipelining();
        DFEVar nearUnderflow = expNormal === 0;
        DFEVar nearOverflow = expNormal === (1<<Math.max(n1exp, n2exp))-2;
        //pipeline stage 1C:
        DFEVar expUpdate = base.control.mux(underflow.cat(overflow), expNormal,
            base.constant.var(KernelBase.dfeInt(Math.max(n1exp, n2exp)+1+1), (1<<Math.max(n1exp, n2exp))-1),
            base.constant.zero(KernelBase.dfeInt(Math.max(n1exp, n2exp)+1+1)));
        DFEVar flowCond = underflow | overflow;
        
        //pipeline stage 2: check result size, including boundary overflow and underflow, shift left by zero/one or set zero/NaN, calculate sticky 
        DFEVar incExp = resultMant.get(n1mant + n2mant - 1).reinterpret(KernelBase.dfeBool());
        DFEVar roundOverflow = resultMant.slice(bigMant-2, n1mant+n2mant-(bigMant-1)) === base.constant.var(KernelBase.dfeUInt(bigMant), (1<<bigMant)-1).cat(~nearUnderflow);
        DFEVar sticky = resultMant.slice(0, n1mant+n2mant-(bigMant+1)) !== base.constant.zero(KernelBase.dfeUInt(n1mant+n2mant-(bigMant+2))).cat(~incExp);
        DFEVar z = base.constant.zero(KernelBase.dfeBool()).cat(isNaN).cat(base.constant.zero(KernelBase.dfeFixOffset(bigMant-1, -bigMant-1, SignMode.UNSIGNED)));
        DFEVar mult = base.control.mux(
            flowCond.cat(nearUnderflow).cat(nearOverflow).cat(incExp),
            resultMant.slice(bigMant-2, n1mant+n2mant-(bigMant-1)),
            resultMant.slice(bigMant-1, n1mant+n2mant-(bigMant-1)),
            resultMant.slice(bigMant-2, n1mant+n2mant-(bigMant-1)), z,
            z, resultMant.slice(bigMant-1, n1mant+n2mant-(bigMant-1)),
            z, z,
            z, z, z, z, z, z, z, z
        );
        incExp = incExp & ~flowCond;
        
        //pipeline stage 3: rounding, final exponent computation
        mult = mult.cat(sticky).reinterpret(KernelBase.dfeFixOffset(bigMant+2, -bigMant-1, SignMode.UNSIGNED)).cast(KernelBase.dfeFixOffset(bigMant-1, -bigMant+1, SignMode.UNSIGNED));
        DFEVar newExp = triAddExact(expUpdate, incExp, roundOverflow, false, false, base).cast(KernelBase.dfeUInt(Math.max(n1exp, n2exp)));
        
        //finalization: recombine mantissa, exponent and sign
        DFEVar result = repeatPipeline(sign, totalDelay, base).cat(newExp).cat(mult).reinterpret(KernelBase.dfeFloat(Math.max(n1exp, n2exp), bigMant));
        //base.debug.simPrintf(result !== n1 * n2, "%f %f %f %f %f %f %f\n", n1, n2, n1 * n2, result, n1m, n2m, mulKaratsubaRectangularExact(n1m, n2m, n1mant + n2mant, 2, false, base));
        return result; //return n1 * n2;
    }
    public static DFEVar maxIntegerBits(DFEVar a, int bits)
    {
        DFEFix aType = (DFEFix)a.getType();
        return a.cast(KernelBase.dfeFixOffset(bits + aType.getFractionBits(), -aType.getFractionBits(), aType.getSignMode()));
    }
    public static DFEComplex doMult(DFEComplex n1, DFEComplex n2, int bits, boolean useUngar, boolean useFloat, KernelBase <?> owner) {
        //(a+bi)(c+di)=(ac-bd)+(bc+ad)i 4 multiplications 2 additions 4M+2A
        //vs. 3 multiplications 5 additions 3M+5A
        //Knuth: (c*(a+b)-b*(c+d))+(c*(a+b)+a*(d-c))i
        //Ungar: (ac - bd)+((a+b)*(c+d)-ac-bd)
        //Ungar is more numerically stable than Knuth, even if dataflow more even in Knuth!
        //if M=3A then 4M+2A=12A+2A=14A, 3M+5A=9A+5A=14A
        //if (useFloat) return n1 * n2;
        DFEVar a = n1.getReal();
        DFEVar c = n2.getReal();
        DFEVar b = n1.getImaginary();
        DFEVar d = n2.getImaginary();
        //DFEVar cab = mulFrac(c, a + b);
        //return DFEComplexType.newInstance(owner, cab-mulFrac(b, c+d), cab + mulFrac(a, d-c), false);
        if (useFloat) {
            //since addition is more expensive for floating point, the 4M+2A formula is optimal!
            //return n1 * n2;
            return DFEComplexType.newInstance(owner, doFloatAdd(doFloatMult(a, c, owner), doFloatMult(b, d, owner), true, owner), doFloatAdd(doFloatMult(c, b, owner), doFloatMult(d, a, owner), false, owner));
            /*if (useUngar) {
                DFEVar ac = a * c;
                DFEVar bd = b * d;
                return DFEComplexType.newInstance(owner, ac - bd, (a + b) * (c + d) - (ac + bd));
            } else {
                DFEVar cab = repeatPipeline(c, 14, owner) * (a+b);
                return DFEComplexType.newInstance(owner, cab - repeatPipeline(b, 14, owner) * (c + d), cab + repeatPipeline(a, 14, owner) * (d - c));
            }*/
        } else { //3M+5A formulas for complex fixed point is optimal DSP usage wise
            //return n1.cast(new DFEComplexType(KernelBase.dfeFixOffset(bits, -bits + 2, SignMode.TWOSCOMPLEMENT))) * n2.cast(new DFEComplexType(KernelBase.dfeFixOffset(bits, -bits + 2, SignMode.TWOSCOMPLEMENT))); 
            if (useUngar) {
                DFEVar ac = mulExact(a, c, bits, 2, owner);
                DFEVar bd = mulExact(b, d, bits, 2, owner);
                return DFEComplexType.newInstance(owner, addExact(ac, bd, true, owner), addExact(mulExact(addExact(a, b, false, owner), addExact(c, d, false, owner), bits, 4, owner), addExact(ac, bd, false, owner), true, owner).cast(KernelBase.dfeFixOffset(bits, -bits + 2, SignMode.TWOSCOMPLEMENT)));
            } else {
                List<List<DFEVar>> chunks = getBitChunks128(a, b, false, true);
                List<List<DFEVar>> chunkscd = getBitChunks128(c, d, false, true);
                List<List<DFEVar>> chunksdc = getBitChunks128(d, c, true, true);
                int depth = Math.max(Math.max(1 + (chunks != null ? chunks.size()-1 : 0), 1 + (chunkscd != null ? chunkscd.size()-1 : 0)), 1 + (chunksdc != null ? chunksdc.size()-1 : 0));
                DFEVar cab = repeatPipeline(
                    mulKaratsubaRectangularExact(repeatPipeline(c, 1 + (chunks != null ? chunks.size()-1 : 0), owner), addExact(a, b, false, owner), bits, 3, false, owner), //2..-2 requires 3 integer bits
                    depth - (1 + (chunks != null ? chunks.size()-1 : 0)), owner);
                DFEComplex ret = DFEComplexType.newInstance(owner, maxIntegerBits(addExact(cab, repeatPipeline(
                        mulKaratsubaRectangularExact(repeatPipeline(b, 1 + (chunkscd != null ? chunkscd.size()-1 : 0), owner), addExact(c, d, false, owner), bits, 3, false, owner), depth - (1 + (chunkscd != null ? chunkscd.size()-1 : 0)), owner), true, owner), 2),
                    maxIntegerBits(addExact(cab, repeatPipeline(
                        mulKaratsubaRectangularExact(repeatPipeline(a, 1 + (chunksdc != null ? chunksdc.size()-1 : 0), owner), addExact(d, c, true, owner), bits, 3, false, owner), depth - (1 + (chunksdc != null ? chunksdc.size()-1 : 0)), owner), false, owner), 2));
                return ret;
            }
        }
    }
    public static int getDSPUnsignedCount(int x, int y) { return getDSPCount(x+1, y+1); }
    public static int getDSPCount(int x, int y)
    {   
        if (y < x) return getDSPCount(y, x);
        int z = ((x-18)%17==0 || (x-18)%17>7) ? 1 : 0;
        return 1-z+(1+(y-25+17-1)/17)*(z+(x-18+17-1)/17); //25x18 signed tiles Virtex 5 DSP48E 24x17 unsigned
    }
    public static DFEVar smallConstMul(DFEVar val, int arg, KernelBase<?> owner)
    {
        switch (arg) {
        case 0: return owner.constant.zero(val.getType());
        case 1: return val;
        case 2: return val << 1;
        case 3: return (val << 1) + val;
        case 4: return val << 2;
        case 5: return (val << 2) + val;
        case 6: return (val << 2) + (val << 1);
        case 7: return (val << 3) - val;
        case 8: return val << 3;
        case 9: return (val << 3) + val;
        case 10: return (val << 3) + (val << 1);
        case 11: return (val << 3) + (val << 1) + val;
        case 12: return (val << 4) - (val << 2);
        case 13: return (val << 3) + (val << 2) + val;
        case 14: return (val << 4) - (val << 1);
        case 15: return (val << 4) - val;
        case 16: return val << 4;
        case 17: return (val << 4) + val;
        case 18: return (val << 4) + (val << 1);
        case 19: return (val << 4) + (val << 1) + val;
        case 20: return (val << 4) + (val << 2);
        default: return null;
        }
    }

    public static class GrayCounter {
        public DFEVar changeValue;
        public DFEVar bitDiffMask;
        public DFEVar changePosition;

        GrayCounter(DFEVar inCounter, KernelBase<?> owner) {
            //grayCount = dfeUInt(64).newInstance(owner);
            //changeValue = dfeBool().newInstance(owner);
            //changePosition = dfeUInt(MathUtils.ceilLog2(64)).newInstance(owner);

            DFEVar grayCount = inCounter ^ (inCounter >> 1);
            DFEVar newGrayCount = owner.optimization.pipeline(grayCount);
            bitDiffMask = grayCount ^ owner.stream.offset(newGrayCount, -1);
            //bitDiffMask = Bitops.trailing1Detect(inCounter);

            changeValue = grayCount > owner.stream.offset(newGrayCount, -1); //(grayCount & bitDiffMask) !== 0;
            changePosition = Bitops.onehotDecode(bitDiffMask);
        }
    }    

    static int id = 0;    
    public static DFEVar[] makeCounterWithInitValue(int bit_width, DFEVar max, DFEVar enable, long inc, DFEVar initValue, DFEVar nextInitValue, DFEVar userReset, boolean needWrap, int loopLength, KernelLib owner)
    {
        int wrap_value = 0;
        SMIO smio = owner.addStateMachine("CounterWithInitValue_" + bit_width + "_NUMERIC_INCREMENTING_" + inc + "_" + wrap_value + "_" + (needWrap ? 1 : 0) + "_COUNT_LT_MAX_THEN_WRAP_" + (bit_width+1) + "_" + loopLength + "_" + id++,
            new CounterWithInitValue(owner, bit_width, max.getType().isConcreteType() ? max.getType().getTotalBits() : bit_width + 1, wrap_value, inc, needWrap, loopLength));
        smio.connectInput("initValue", initValue);
        smio.connectInput("nextInitValue", nextInitValue);
        smio.connectInput("preWrap", nextInitValue >= max);
        smio.connectInput("max", max);
        if (userReset != null) smio.connectInput("userReset", userReset);
        smio.connectInput("enable", enable);
        return new DFEVar[] { smio.getOutput("count"), needWrap ? smio.getOutput("wrap") : null };
    }
    
    public static class PassThroughKernelLite extends KernelLite {
        public PassThroughKernelLite(ManagerKernelBase owner, String name, KernelConfiguration config, int vecsize, int bits) {
            super(owner, name, config);
            if (vecsize == 1) {
                DFEType packetType = dfeUInt(bits);
                PushInput<DFEVar> inputPacket = io.pushInput("input", packetType, io.stallLatencyTo("input"));
                //PullInput<DFEVar> inputPacket = io.pullInput("input", packetType, io.almostEmptyLatencyTo("input")); //almostEmptyLatency=1
                PushOutput<DFEVar> output =
                io.pushOutput("output", packetType, inputPacket.getStallLatency() + stream.ioDistance("input", "output"));
                //io.pushOutput("output", packetType, inputPacket.getAlmostEmptyLatency() + stream.ioDistance("input", "output"));
                //DFEVar outputStall = dfeBool().newInstance(this);
                //outputStall <== output.stall;
                //pushStall(outputStall); popStall();
                //io.scalarOutput("stall", output.stall.getType()) <== output.stall;
		        output.valid <== inputPacket.valid; //inputPacket.empty;
		        output.data <== inputPacket.data;
                inputPacket.stall <== output.stall; //inputPacket.read <== constant.var(true);
                
            } else {
                DFEVectorType<DFEVar> packetType = new DFEVectorType<DFEVar>(dfeUInt(bits), vecsize);
                PushInput<DFEVector<DFEVar>> inputPacket = io.pushInput("input", packetType, io.stallLatencyTo("input"));
                //PullInput<DFEVector<DFEVar>> inputPacket = io.pullInput("input", packetType, io.almostEmptyLatencyTo("input")); //almostEmptyLatency=1
                PushOutput<DFEVector<DFEVar>> output =
                io.pushOutput("output", packetType, inputPacket.getStallLatency() + stream.ioDistance("input", "output"));
                //io.pushOutput("output", packetType, inputPacket.getAlmostEmptyLatency() + stream.ioDistance("input", "output"));
                //DFEVar outputStall = dfeBool().newInstance(this);
                //outputStall <== output.stall;
                //pushStall(outputStall); popStall();
                //io.scalarOutput("stall", output.stall.getType()) <== output.stall;
		        output.valid <== inputPacket.valid; //inputPacket.empty;
		        output.data <== inputPacket.data;
                inputPacket.stall <== output.stall; //inputPacket.read <== constant.var(true);
            }
            
        }
    }

    public static class CounterWithInitValue extends KernelStateMachine //requires a reset signal for initialization, reset happens on current tick, not next tick
    {
        private final int m_bit_width;
        private final long m_inc;
        private final long m_wrap_value;
        private final boolean m_needWrap;
        private final int m_loopLength;
        private final DFEsmValueType m_count_type;
        private final DFEsmValueType m_count_type_int;
        private final DFEsmOutput m_count;
        private final DFEsmOutput m_wrap;
        private final DFEsmStateValue[] m_counter_values;
        private final DFEsmStateValue[] m_counter_next_values;
        private final DFEsmStateValue[] m_counter_onext_values;
        private final DFEsmStateValue[] m_has_wrappeds;
        private final DFEsmStateValue[] m_next_wrap;
        private DFEsmInput m_user_enable;
        private DFEsmInput m_user_reset;
        private DFEsmInput m_prewrap;
        private final DFEsmValue m_limit;
        private final DFEsmValue m_init_value;
        private final DFEsmValue m_next_init_value;
        private DFEsmValue wrap;
        public CounterWithInitValue(KernelLib owner, int bit_width, int max_width, long wrap_value, long inc, boolean needWrap, int loopLength)
        {
            super(owner);
            m_bit_width = bit_width;
            m_wrap_value = wrap_value;
            m_inc = inc;
            m_needWrap = needWrap;
            m_loopLength = loopLength;
            m_user_enable = null;
            m_user_reset = null;
            wrap = null;
            m_count_type = StateMachineLib.dfeUInt(m_bit_width);
            m_count_type_int = StateMachineLib.dfeUInt(m_bit_width + 1);
            m_init_value = io.input("initValue", m_count_type).cast(m_count_type_int);
            m_next_init_value = io.input("nextInitValue", m_count_type).cast(m_count_type_int);
            m_count = io.output("count", m_count_type);
            m_wrap = needWrap ? io.output("wrap", StateMachineLib.dfeBool()) : null;
            m_limit = io.input("max", StateMachineLib.dfeUInt(max_width)).cast(m_count_type_int);
            m_user_enable = io.input("enable", StateMachineLib.dfeBool());
            m_user_reset = io.input("userReset", StateMachineLib.dfeBool());
            m_prewrap = io.input("preWrap", StateMachineLib.dfeBool());
            m_counter_values = new DFEsmStateValue[loopLength];
            m_counter_next_values = new DFEsmStateValue[loopLength];
            m_counter_onext_values = new DFEsmStateValue[loopLength];
            m_has_wrappeds = new DFEsmStateValue[loopLength];
            m_next_wrap = new DFEsmStateValue[loopLength];
            for (int i = 0; i < loopLength; i++) {
                m_counter_values[i] = state.value(m_count_type_int);
                m_counter_next_values[i] = state.value(m_count_type_int);
                m_counter_onext_values[i] = state.value(m_count_type_int);
                m_has_wrappeds[i] = state.value(StateMachineLib.dfeBool()); //, false
                m_next_wrap[i] = state.value(StateMachineLib.dfeBool());
            }
            for (int i = 0; i < m_loopLength-1; i++) {
                m_counter_values[i+1].next.connect(m_counter_values[i]);
                m_counter_next_values[i+1].next.connect(m_counter_next_values[i]);
                m_counter_onext_values[i+1].next.connect(m_counter_onext_values[i]);
                m_has_wrappeds[i+1].next.connect(m_has_wrappeds[i]);
                m_next_wrap[i+1].next.connect(m_next_wrap[i]);
            }
            //wrap = m_counter_next_values[loopLength-1].gte(m_limit);
        }
        
        @Override
        protected void nextState() {
            if (m_user_reset != null) {
                _IF(m_user_reset);
                _IF(m_user_enable);
                m_counter_values[0].next.connect(m_next_init_value);
                DFEsmValue nextVal = m_next_init_value.add(m_inc);
                m_counter_next_values[0].next.connect(nextVal);
                m_counter_onext_values[0].next.connect(m_next_init_value.add(m_inc * 2));
                m_next_wrap[0].next.connect(nextVal.gte(m_limit));
                _ELSE();
                m_counter_values[0].next.connect(m_init_value);
                m_counter_next_values[0].next.connect(m_next_init_value);
                m_counter_onext_values[0].next.connect(m_next_init_value.add(m_inc));
                m_next_wrap[0].next.connect(m_next_init_value.gte(m_limit));
                _END_IF();
                m_has_wrappeds[0].next.connect(false);
                _ELSE();
                nextStateProcessEnable();
                _END_IF();
            }
            else {
                nextStateProcessEnable();
            }
        }
        
        void nextStateProcessEnable() {
            if (m_user_enable != null) {
                _IF(m_user_enable);
                genNextState();
                if (m_loopLength != 1) {
                    _ELSE();
                    m_counter_values[0].next.connect(m_counter_values[m_loopLength-1]);
                    m_counter_next_values[0].next.connect(m_counter_next_values[m_loopLength-1]);
                    m_counter_onext_values[0].next.connect(m_counter_onext_values[m_loopLength-1]);
                    m_has_wrappeds[0].next.connect(m_has_wrappeds[m_loopLength-1]);
                    m_next_wrap[0].next.connect(m_next_wrap[m_loopLength-1]);
                }
                _END_IF();
            }
            else {
                genNextState();
            }
        }
        
        private void genNextState() {
            m_counter_onext_values[0].next.connect((DFEsmExpr)m_counter_next_values[m_loopLength-1].add(2L * m_inc));
            //wrap = m_counter_next_values[m_loopLength-1].gte(m_limit);
            wrap = m_next_wrap[m_loopLength-1];
            _IF(wrap);
            m_has_wrappeds[0].next.connect(true);
            m_counter_values[0].next.connect(m_wrap_value);
            m_next_wrap[0].next.connect(m_limit.lte(m_wrap_value + m_inc));            
            m_counter_next_values[0].next.connect(m_wrap_value + m_inc);
            _ELSE();
            _IF(m_has_wrappeds[m_loopLength-1]);
            m_has_wrappeds[0].next.connect(false);
            m_counter_values[0].next.connect((DFEsmExpr)m_counter_next_values[m_loopLength-1]);
            m_next_wrap[0].next.connect(m_limit.lte(m_wrap_value + m_inc * 2));            
            m_counter_next_values[0].next.connect(m_wrap_value + m_inc * 2);
            _ELSE();
            if (m_loopLength != 1) m_has_wrappeds[0].next.connect(m_has_wrappeds[m_loopLength-1]);
            m_counter_values[0].next.connect((DFEsmExpr)m_counter_next_values[m_loopLength-1]);
            m_next_wrap[0].next.connect(m_counter_onext_values[m_loopLength-1].gte(m_limit));
            m_counter_next_values[0].next.connect(m_counter_onext_values[m_loopLength-1]);
            _END_IF();
            _END_IF();
        }
        
        @Override
        protected void outputFunction() {
            _IF(m_user_reset);
            m_count.connect((DFEsmExpr)m_init_value.cast(m_count_type));
            //if (m_needWrap) m_wrap.connect(m_next_init_value.gte(m_limit));
            _ELSE();
            m_count.connect((DFEsmExpr)m_counter_values[m_loopLength-1].slice(0, m_bit_width));
            //if (m_needWrap) m_wrap.connect((DFEsmExpr)wrap.and(m_user_enable));
            _END_IF();
            //formula: m_user_reset & m_prewrap | ~m_user_reset & m_user_enable & wrap
            //gen_truth_mux(lambda a, b, c, d: a & d | ~a & b & c, 4)
            if (m_needWrap) m_wrap.connect(control.mux(m_user_reset.cat(m_user_enable).cat(wrap).cat(m_prewrap), constant.value(false), constant.value(false), constant.value(false), constant.value(false), constant.value(false), constant.value(false), constant.value(true), constant.value(true), constant.value(false), constant.value(true), constant.value(false), constant.value(true), constant.value(false), constant.value(true), constant.value(false), constant.value(true)));
        }
    }
    
    public static DFEVar[] counterWithInit(int bit_width, DFEVar max, DFEVar enable, long inc, DFEVar initValue, DFEVar userReset, DFEVar nextUserReset, int loopLength, boolean needWrap, boolean pipelineMax, KernelBase<?> owner)
    {
        //return makeCounterWithInitValue(bit_width, max, enable, inc, initValue, initValue+inc, userReset, needWrap, owner);
        // max>0 required
        
        //Addition-based method
        DFEVar count = KernelBase.dfeUInt(bit_width).newInstance(owner);
        DFEVar wrap = KernelBase.dfeBool().newInstance(owner);
        DFEVar maxWrap = max-1;
        if (pipelineMax) maxWrap = owner.optimization.pipeline(maxWrap);
        DFEVar streamWrap = Utility.repeatPipeline(owner.stream.offset(wrap, -loopLength), loopLength - 4, owner);
        DFEVar wrapValue = needWrap ? owner.control.mux(enable.cat(nextUserReset), owner.constant.zero(KernelBase.dfeBool()), owner.constant.zero(KernelBase.dfeBool()), streamWrap, initValue === maxWrap) : null;
        DFEVar lastWrap = nextUserReset ? initValue === maxWrap : streamWrap; //lastCount 2 delay, streamWrap 1 delay
        //DFEVar lastCount = userReset ? initValue + ((enable & (initValue !== maxWrap)) ? owner.constant.var(KernelBase.dfeUInt(bit_width), inc) : owner.constant.zero(KernelBase.dfeUInt(bit_width))) : owner.stream.offset(count, -loopLength);
        DFEVar lastCount = userReset ? initValue : Utility.repeatPipeline(owner.stream.offset(count, -loopLength), loopLength - 4, owner);
        DFEVar countValue = lastCount;
        for (int i = 0; i < 2; i++) lastCount = owner.optimization.pipeline(lastCount);
        maxWrap = owner.optimization.pipeline(maxWrap);
        for (int i = 0; i < 2; i++) enable = owner.optimization.pipeline(enable);
        //owner.optimization.pushNoPipelining(); //conditional addition=1, double mux=1, >= is 1, so loop offset too 3 not 1...
        lastCount = lastCount + (enable ? (lastWrap ? ~maxWrap+1 : owner.constant.var(KernelBase.dfeUInt(bit_width), inc)) : owner.constant.zero(KernelBase.dfeUInt(bit_width))); //lastCount 1 delay, lastWrap 2 delay
        //owner.optimization.popNoPipelining();
        for (int i = 0; i < 3; i++) maxWrap = owner.optimization.pipeline(maxWrap);
        lastWrap = lastCount >= maxWrap; //1 delay
        //for (int i = 0; i < loopLength - 4; i++) lastCount = owner.optimization.pipeline(lastCount);
        //for (int i = 0; i < loopLength - 4; i++) lastWrap = owner.optimization.pipeline(lastWrap);
        count <== lastCount;
        wrap <== lastWrap;
        return new DFEVar[] {countValue, wrapValue};
        
        /*
        //State-machine based method
        //DFEVar userReset = owner.constant.zero(KernelBase.dfeBool());
        initValue = initValue.cast(KernelBase.dfeUInt(bit_width+1));
        max = max.cast(KernelBase.dfeUInt(bit_width+1));
        DFEVar wrapValue = owner.constant.zero(KernelBase.dfeUInt(bit_width+1));
        DFEVar count = KernelBase.dfeUInt(bit_width+1).newInstance(owner);
        DFEVar next_value = KernelBase.dfeUInt(bit_width+1).newInstance(owner);
        DFEVar onext_value = KernelBase.dfeUInt(bit_width+1).newInstance(owner);
        DFEVar has_wrapped = KernelBase.dfeBool().newInstance(owner);
        DFEVar nv = owner.stream.offset(next_value, -loopLength);
        DFEVar wrap = nv >= max | userReset;
        DFEVar lastCount = userReset ? initValue : owner.stream.offset(count, -loopLength);
        DFEVar lastHasWrapped = userReset ? false : owner.stream.offset(has_wrapped, -loopLength);
        DFEVar lastONextValue = userReset ? initValue+2*inc : owner.stream.offset(onext_value, -loopLength);
        count <== enable ? (wrap ? wrapValue : nv) : lastCount;
        //owner.optimization.pushNoPipelining();
        next_value <== control.mux(enable.cat(lastHasWrapped.cat(wrap)), lastONextValue, wrapValue+inc, wrapValue+2*inc, wrapValue+inc, nv, nv, nv, nv);
        //owner.optimization.popNoPipelining();
        onext_value <== enable ? nv + 2*inc : lastONextValue;
        has_wrapped <== enable ? wrap : lastHasWrapped;        
        return new DFEVar[] {lastCount.slice(0, bit_width).cast(KernelBase.dfeUInt(bit_width)), enable & wrap};*/
        
        /*
        //Counter-adjustment based method - not loopLength sensitive
        DFEVar initMax = KernelBase.dfeBool().newInstance(owner);
        DFEVar useMax = userReset ? false : owner.stream.offset(initMax, -1);
        Count.Counter counter = control.count.makeCounter(control.count.makeParams(bit_width).withMax(max - (useMax ? owner.constant.zero(KernelBase.dfeUInt(bit_width)) : initValue)).withInc(inc).withEnable(enable).withReset(userReset));
        DFEVar wrap = counter.getWrap();
        initMax <== useMax | wrap;
        return new DFEVar[] {counter.getCount() + (useMax ? owner.constant.zero(KernelBase.dfeUInt(bit_width)) : initValue), wrap};*/
        
        /*
        //Double counter method - not loopLength sensitive
        DFEVar afterInit = KernelBase.dfeBool().newInstance(owner);
        Count.Counter initCounter = control.count.makeCounter(control.count.makeParams(bit_width).withMax(max-initValue).withInc(inc).withEnable(enable).withReset(userReset));
        DFEVar wrap = initCounter.getWrap();
        afterInit <== userReset ? false : (owner.stream.offset(afterInit, -1) | owner.stream.offset(wrap, -1));
        Count.Counter counter = control.count.makeCounter(control.count.makeParams(bit_width).withMax(max).withInc(inc).withEnable(enable & afterInit));
        return new DFEVar[] { afterInit ? counter.getCount() : initCounter.getCount() + initValue, afterInit ? counter.getWrap() : wrap };*/
    }

    public static List<DFEVector<DFEVar>> counterChainWithInit(DFEVar enable, DFEVector<DFEVar> max, DFEVar initValue, DFEVar nextInitValue, DFEVector<DFEVar> userReset, int inc, int loopLength, boolean needWraps, KernelLib owner) //DFEVar[] initValue, 
    {
        final int counterChainWrapPipeliningLimit = 4;
        final int counterChainWrapPipelining = 1;
        DFEVector<DFEVar> counterChain = new DFEVectorType<DFEVar>(max[0].getType(), max.getType().getSize()).newInstance(owner);
        DFEVector<DFEVar> wraps = needWraps ? new DFEVectorType<DFEVar>(KernelBase.dfeBool(), max.getType().getSize()).newInstance(owner) : null;
        DFEVar wrap = null;
        int pipelineLimit = (max.getType().getSize()-1) / counterChainWrapPipeliningLimit;
        int currentPipelineOffset = 0;
        for (int i = 0; i < max.getType().getSize(); i++) {
            DFEVar[] counter = makeCounterWithInitValue(max[i].getType().getTotalBits(), Utility.repeatPipeline(max[i], currentPipelineOffset, owner), i == 0 ? enable : wrap, inc, initValue,
                    nextInitValue, Utility.repeatPipeline(userReset[i], currentPipelineOffset, owner), i != max.getType().getSize()-1 || needWraps, loopLength, owner);
            //DFEVar[] counter = counterWithInit(max[i].getType().getTotalBits(), Utility.repeatPipeline(max[i], currentPipelineOffset, owner), i == 0 ? enable : wrap, inc, initValue, userReset[i], userReset[i], loopLength, i != max.getType().getSize()-1 || needWraps, i==0, owner);            
            counterChain[i] <== Utility.repeatPipeline(counter[0], pipelineLimit - currentPipelineOffset, owner); //counter.getCount();
            if (i != max.getType().getSize()-1) {
                int newPipelining = i != 0 && (i % counterChainWrapPipeliningLimit == 0) ? counterChainWrapPipelining : 0;
                wrap = Utility.repeatPipeline(counter[1], newPipelining, owner); //counter.getWrap();
                initValue = Utility.repeatPipeline(initValue, newPipelining, owner);
                nextInitValue = Utility.repeatPipeline(nextInitValue, newPipelining, owner);
                currentPipelineOffset += newPipelining;
            } else if (needWraps) wrap = counter[1];
            if (needWraps) wraps[i] <== Utility.repeatPipeline(wrap, pipelineLimit - currentPipelineOffset, owner);
        }
        List<DFEVector<DFEVar>> ret = new ArrayList<DFEVector<DFEVar>>();
        ret.add(counterChain); ret.add(needWraps ? wraps : null);
        return ret;
    }
    


    public static DFEVar[] parallelFanout(DFEVar var, int count, int limit, KernelBase<?> base) //limit==32 is compiler default maximum fanout
    {
        if (count <= limit) return new DFEVar[] { var };
        DFEVar[] fanout = new DFEVar[(count+limit-1)/limit];
        for (int i = 0; i < (count+limit-1)/limit; i++) {
            fanout[i] = base.optimization.pipeline(var);
        }
        return fanout;
    }
    public static <T extends KernelObject<T>> T repeatPipeline(T var, int count, KernelBase<?> base)
    {
        for (int i = 0; i < count; i++) var = base.optimization.pipeline(var);
        return var;
    }
    public static DFEVar[] pipelineStaggered(DFEVar var, int[] counts, KernelBase<?> base)
    {
        DFEVar[] stagger = new DFEVar[counts.length];
        for (int i = 0; i < counts.length; i++) {
            var = repeatPipeline(var, counts[i], base);
            stagger[i] = var;
        }
        return stagger;
    }
    public static <T extends KernelObjectVectorizable<T,?>> T repeatVectorPipeline(T var, int count, int limit, KernelBase<?> base)
    {
        DFEVar v = var.reinterpret(KernelBase.dfeUInt(var.getType().getTotalBits()));
        int sz = ((DFEVectorType<?>)var.getType()).getContainedType().getTotalBits();        
        DFEVar out = null;
        for (int i = 0; i < 2; i++) {
            DFEVar chunk = repeatPipeline(v.slice(i*limit*sz, i == 0 ? limit*sz : var.getType().getTotalBits() - i*limit*sz), count, base);
            if (i == 0) out = chunk;
            else out = chunk.cat(out);
        }
        return out.reinterpret(var.getType());
    }
    public static DFEVar pipelineBits(DFEVar var, KernelBase<?> base)
    {
        int size = var.getType().getTotalBits();
        DFEVar res = base.optimization.pipeline(var.get(--size));
        while (size != 0) {
            res = res.cat(base.optimization.pipeline(var.get(--size)));
        }
        return res;
    }
    public static <T extends KernelObject<T>> T mux32to64(DFEVar sel, List<T> values, int split, KernelBase<?> base) //split==2 bits, 6-LUT 2 selection bits + 4 selections per Xilinx docs 
    {
        final int muxPipeliningLimit = split >= 4 ? 1 : (split == 3 ? 2 : 3);
        int limit = Math.max(32, 1 << split);
        int bitsToAddress = MathUtils.bitsToAddress(values.size()); //==6 bits...
        for (int bits = 0; bits < bitsToAddress; bits += split) {
            if (bits + split >= bitsToAddress) return base.control.mux(base.optimization.pipeline(sel), values); 
            DFEVar[] sels = parallelFanout(sel.slice(0, split), values.size() - (values.size() % (1 << split) == 1 ? 1 : 0), limit, base);
            List<T> nextValues = new ArrayList<>();
            boolean doPipelining = ((bits / split) % muxPipeliningLimit) == (muxPipeliningLimit-1);
            for (int i = 0; i < values.size(); i += (1 << split)) {
                List<T> curvals = values.subList(i, i+Math.min(1 << split, values.size() - i));
                if (curvals.size() == 1) {
                    nextValues.add(doPipelining ? base.optimization.pipeline(curvals.get(0)) : curvals.get(0));
                } else {
                    if (!doPipelining) base.optimization.pushNoPipelining();
                    nextValues.add(base.control.mux(sels[i / limit].slice(0, MathUtils.bitsToAddress(curvals.size())), curvals));
                    if (!doPipelining) base.optimization.popNoPipelining();
                }
            }
            values = nextValues;
            sel = sel.slice(split, bitsToAddress - bits - split);
            if (doPipelining) sel = base.optimization.pipeline(sel);
        }
        return null;
    }
    public static <T extends KernelObject<T>> T oneHotMux32to64(DFEVar sel, List<T> values, int split, KernelBase<?> base) //split==3 one hot bits, 6-LUT 3 selection bits + 3 selections per Xilinx docs
    {
        final int muxPipeliningLimit = 3;
        int iterations = 0;
        while (true) {
            if (split >= values.size()) return base.optimization.pipeline(base.control.oneHotMux(iterations == 0 ? base.optimization.pipeline(sel) : sel, values));
            List<T> nextValues = new ArrayList<>();
            DFEVar nextSel = null;
            boolean doPipelining = (iterations % muxPipeliningLimit) == (muxPipeliningLimit-1);
            for (int i = 0; i < values.size(); i += split) {
                DFEVar s = sel.slice(i, Math.min(split, values.size() - i));
                if (iterations == 0) s = base.optimization.pipeline(s);                
                if (!doPipelining) base.optimization.pushNoPipelining();
                if (nextSel == null) nextSel = s !== 0;
                else nextSel = (s !== 0).cat(nextSel);
                if (!doPipelining) base.optimization.popNoPipelining();
                T nextVal = base.control.oneHotMux(s, values.subList(i, i+Math.min(split, values.size() - i)));
                nextValues.add(doPipelining ? base.optimization.pipeline(nextVal) : nextVal);
            }
            values = nextValues;
            sel = nextSel;
            iterations++;
        }
    }
    public static <P extends KernelObjectVectorizable<P,?>,M extends DFEVectorBase<P,M,C,T>,C extends DFEVectorBase<DFEVar,C,?,?>,T extends DFEVectorTypeBase<P,M,C>,S extends DFEVectorBase<DFEVar,S,?,?>>
    M muxVector32to64(DFEVar sel, List<M> values, int split, KernelBase<?> base) //split==2 bits, 6-LUT 2 selection bits + 4 selections per Xilinx docs
    {
        int innerSize = values[0].getSize();
        M returnVal = values[0].getType().newInstance(base);
        for (int i = 0; i < innerSize; i++) {
            final int j = i;
            List<P> curVals = values.stream().map(x -> x[j]).collect(Collectors.toList());
            returnVal[i] <== mux32to64(base.optimization.pipeline(sel), curVals, split, base);
        }
        return returnVal;
    }    
    public static <P extends KernelObjectVectorizable<P,?>,M extends DFEVectorBase<P,M,C,T>,C extends DFEVectorBase<DFEVar,C,?,?>,T extends DFEVectorTypeBase<P,M,C>,S extends DFEVectorBase<DFEVar,S,?,?>>
    M oneHotMuxVector32to64(DFEVar sel, List<M> values, int split, KernelBase<?> base) //split==3 one hot bits, 6-LUT 3 selection bits + 3 selections per Xilinx docs
    {
        int innerSize = values[0].getSize();
        M returnVal = values[0].getType().newInstance(base);
        for (int i = 0; i < innerSize; i++) {
            final int j = i;
            List<P> curVals = values.stream().map(x -> x[j]).collect(Collectors.toList());
            returnVal[i] <== oneHotMux32to64(base.optimization.pipeline(sel), curVals, split, base);
        }
        return returnVal;
    }
    public static DFEVector<DFEVar> oneHotEncode(DFEVar shift, int bits, int limit, KernelBase<?> base) //limit==32, at 1024 we would need 2 stage pipeline tree...
    {
        DFEVar[] shifts = parallelFanout(shift, bits, limit, base); 
        DFEVector<DFEVar> res = new DFEVectorType<DFEVar>(KernelBase.dfeBool(), bits).newInstance(base);
        for (int i = 0; i < bits; i++) {
            res[i] <== shifts[i / limit] === i;
        }
        return res;
    }
    DFEVar shiftLeft(DFEVar value, DFEVar shift, int bits, int limit, KernelBase<?> base)
    {
        DFEVector<DFEVar> oneHot = oneHotEncode(shift, bits, limit, base);
        DFEVar[] valuesFanned = parallelFanout(value, bits, 32, base);
        List<DFEVar> values = new ArrayList<DFEVar>();
        for (int i = 0; i < bits; i++) {
            values.add((valuesFanned[i < limit ? 0 : 1] << i).cast(KernelBase.dfeUInt(bits)));
        } 
        return oneHotMux32to64(oneHot.reinterpret(KernelBase.dfeUInt(bits)), values, limit, base);
    }
    DFEVar shiftLeft(long value, DFEVar shift, int bits, int limit, KernelBase<?> base)
    {
        DFEVector<DFEVar> oneHot = oneHotEncode(shift, bits, limit, base);
        List<DFEVar> values = new ArrayList<DFEVar>();
        for (int i = 0; i < bits; i++) {
            values.add(base.constant.var(KernelBase.dfeUInt(bits), value << i));
        }
        return oneHotMux32to64(oneHot.reinterpret(KernelBase.dfeUInt(bits)), values, limit, base);
    }    
    public static void printKernelOptimizationDefaults(KernelBase<?> owner)
    {
        System.out.println("Graph Pipelining: " + owner.optimization.peekGraphPipelining(PipelinedOps.ALL)); //default 0.0
        System.out.println("Node Pipelining: " + owner.optimization.peekNodePipelining(PipelinedOps.ALL)); //default 1.0
        System.out.println("DSP Factor: " + owner.optimization.peekDSPFactor(MathOps.ALL)); //default 0.5
        System.out.println("Enable Growth: " + owner.optimization.getEnableBitGrowth()); //default false
        System.out.println("Enable Saturating Arithmetic: " + owner.optimization.peekEnableSaturatingArithmetic()); //default false
        System.out.println("Fanout Limit: " + owner.optimization.peekFanoutLimit()); //default 32
        System.out.println("Fanout Limit Type: " + owner.optimization.peekFanoutLimitType()); //default CHAIN
        System.out.println("Rounding Mode: " + owner.optimization.getRoundingMode()); //default TONEAREVEN
    }
    public static DFELink slrCrossingChain(String name, String suffix, DFELink input, int slrStart, int slrEnd, int regPerX, ManagerRouting owner, ManagerClock[] mc) //-1 for PCIE //ManagerKernel owner
    {
        if (owner != null) {
            for (int i = slrStart + (slrEnd >= slrStart ? 1 : -1); slrEnd >= slrStart ? i < slrEnd : i > slrEnd; ) {
                selectSLR(i, (XilinxAlveoU250Manager)owner);
                //KernelConfiguration config = new KernelConfiguration(owner.getCurrentKernelConfig());
                //config.optimization.setUseGlobalClockBuffer(false);
                //KernelBlock fo = owner.addKernel(new PassThroughKernelLite(owner, name + "Fanout" + suffix + i, config, 1, input.getSourceWidth()));
                Fanout fo = owner.fanout(name + "Fanout" + suffix + i);
                unselectSLR((XilinxAlveoU250Manager)owner);
                //fo.setClock(((ManagerKernel)owner).generateStaticClock(name + "Clock" + suffix + i, freq));
                if (mc[i] != null) fo.setClock(mc[i]);
                //DFELink inp = fo.getInput("input");
                DFELink inp = fo.getInput();
                inp.setAdditionalRegisters(regPerX);
                inp <== input;
                //input = fo.getOutput("output");
                input = fo.addOutput(name + "Chain" + suffix + i);
                input.setAdditionalRegisters(regPerX);
                if (slrEnd >= slrStart) i++; else i--;
            }
        } else
            input.setAdditionalRegisters(regPerX * ((slrEnd >= slrStart) ? slrEnd - slrStart : slrStart - slrEnd));
        return input;
    }
    public static Fanout[] slrCrossingFanoutChain(String name, String suffix, DFELink input, int slrStart, int slrEnd, int regPerX, ManagerRouting owner) //slrStart is inclusive here
    {
        Fanout fanouts[] = new Fanout[slrEnd - slrStart];
        for (int i = slrStart; i < slrEnd; i++) {
            selectSLR(i, (XilinxAlveoU250Manager)owner);
            fanouts[i] = owner.fanout(name + "Fanout" + suffix + i);
            unselectSLR((XilinxAlveoU250Manager)owner);
        }
        for (int i = slrStart; i < slrEnd; i++) {
            if (i == slrStart) {
                DFELink inp = fanouts[0].getInput();
                inp.setAdditionalRegisters(input.getAdditionalRegisters()); //PCIE->SLR0 - 1 crossing
                inp <== input;
            }
            if (i != slrEnd - 1) {
                DFELink chain = fanouts[i].addOutput(name + "Chain" + suffix + i);
                chain.setAdditionalRegisters(regPerX); //PCIE->SLR0 or SLR0->SLR1 or SLR1->SLR2- 1 crossing
                DFELink inp = fanouts[i+1].getInput();
                inp.setAdditionalRegisters(regPerX); //SLR0->SLR1 or SLR1->SLR2- 1 crossing
                inp <== chain;
            }             
        }
        return fanouts;
    }
    public static HashMap<String, ImplementationStrategy> getImplementationStrategies(boolean remap)
    { 
        //route options: RouteOpt.ALTERNATE_CLB_ROUTING, RouteOpt.EXPLORE, RouteOpt.createCustom(RouteOpt.ALTERNATE_CLB_ROUTING, true, false, ""), RouteOpt.createCustom(RouteOpt.EXPLORE, true, false, "")
        
        //You can also run phys_opt_design with the -sll_reg_hold_fix option after route_design to attempt fixing hold violations on these direct Laguna TX to RX paths with potentially long runtime.
        //PhysOpt.EXPLORE_WITH_HOLD_FIX but hold_fix and sll_reg_hold_fix are different options

        //boolean timingDriven, boolean timingSummary, boolean unplace, boolean postPlaceOpt, boolean fanoutOpt, boolean bufgOpt, java.lang.String additionalOptions
        //The directive option is incompatible with other options with the exception of -no_fanout_opt, -no_bufg_opt, -quiet, and -verbose but Vivado 2021.1 and later only
        //ImplementationStrategy.PlaceOpt PlaceOptAuto1 = ImplementationStrategy.PlaceOpt.createCustom(true, false, false, false, false, true, "-directive Auto_1"); //High performing predicted directive
        //ImplementationStrategy.PlaceOpt PlaceOptAuto2 = ImplementationStrategy.PlaceOpt.createCustom(true, false, false, false, false, true, "-directive Auto_2"); //Second best predicted directive
        //ImplementationStrategy.PlaceOpt PlaceOptAuto3 = ImplementationStrategy.PlaceOpt.createCustom(true, false, false, false, false, true, "-directive Auto_3"); //Third best predicted directive

        //phys_opt_design [-fanout_opt] [-placement_opt] [-routing_opt] [-slr_crossing_opt] [-rewire] [-insert_negative_edge_ffs] [-critical_cell_opt] [-dsp_register_opt] [-bram_register_opt] [-uram_register_opt] [-bram_enable_opt] [-shift_register_opt] [-hold_fix] [-aggressive_hold_fix] [-retime] [-force_replication_on_nets <args>] [-directive <arg>] [-critical_pin_opt] [-clock_opt] [-path_groups <args>] [-tns_cleanup] [-sll_reg_hold_fix] [-quiet] [-verbose]
        //boolean fanoutOpt, boolean placementOpt, boolean routingOpt, boolean slrCrossingOpt, boolean rewire, boolean criticalCellOpt, boolean dspRegisterOpt, boolean bramRegisterOpt, boolean uramRegisterOpt, boolean bramEnableOpt, boolean shiftRegisterOpt, boolean holdFix, boolean retime, java.lang.String forceReplicationOnNets, boolean criticalPinOpt, boolean clockOpt, java.lang.String pathGroups, java.lang.String additionalOptions
        //https://docs.xilinx.com/r/en-US/ug904-vivado-implementation/Available-Physical-Optimizations
        ImplementationStrategy.PhysOpt postPlaceOpt = ImplementationStrategy.PhysOpt.createCustom(true, true, false, true, true, true, true, true, true, true, true, true, true, "", true, false, "", "-insert_negative_edge_ffs -aggressive_hold_fix -sll_reg_hold_fix"); // -tns_cleanup
        ImplementationStrategy.PhysOpt postRouteOpt = ImplementationStrategy.PhysOpt.createCustom(false, true, true, true, true, true, false, false, false, false, false, true, true, "", true, true, "", "-aggressive_hold_fix -sll_reg_hold_fix"); // -tns_cleanup
        //boolean retarget, boolean propagateConst, boolean sweep, boolean bramPowerOpt, boolean remap, boolean resynthArea, boolean resynthSeqArea, boolean muxfRemap, int hierFanoutLimit, boolean bufgOpt, boolean shiftRegisterOpt, boolean controlSetMerge, boolean mergeEquivalentDrivers, boolean carryRemap, java.lang.String addionalOptions 
        int hierFanoutLimit = 512; //512 minimum
        ImplementationStrategy.NetlistOpt netlistOpt = ImplementationStrategy.NetlistOpt.createCustom(true, true, true, true, remap, false, false, false, hierFanoutLimit, true, true, false, false, false,
            "-aggressive_remap"); // -resynth_remap -dsp_register_opt
        ImplementationStrategy.NetlistOpt netlistOptDEFAULT = netlistOpt;
        ImplementationStrategy.NetlistOpt netlistOptEXPLORE = netlistOpt;
        ImplementationStrategy.NetlistOpt netlistOptEXPLORE_AREA = netlistOpt;
        ImplementationStrategy.NetlistOpt netlistOptEXPLORE_SEQUENTIAL_AREA = netlistOpt;
        ImplementationStrategy.NetlistOpt netlistOptEXPLORE_WITH_REMAP = netlistOpt;
        ImplementationStrategy.NetlistOpt netlistOptRUNTIME_OPTIMIZED = netlistOpt;
        ImplementationStrategy.NetlistOpt netlistOptNO_BRAM_POWER_OPT = netlistOpt;
        //"-directive ExploreWithAggressiveHoldFix"
        ImplementationStrategy.PhysOpt PhysOptDISABLED = ImplementationStrategy.PhysOpt.DISABLED;
        ImplementationStrategy.PhysOpt PhysOptDEFAULT = postPlaceOpt; //default with retiming
        ImplementationStrategy.PhysOpt PhysOptPostDEFAULT = postRouteOpt;
        ImplementationStrategy.PhysOpt PhysOptEXPLORE = postPlaceOpt; //Run different algorithms in multiple passes of optimization, including replication for very high fanout nets, SLR crossing optimization, and a final phase called Critical Path Optimization where a subset of physical optimizations are run on the top critical paths of all endpoint clocks, regardless of slack.
        ImplementationStrategy.PhysOpt PhysOptPostEXPLORE = postRouteOpt;
        ImplementationStrategy.PhysOpt PhysOptAGGRESSIVE_EXPLORE = postPlaceOpt; //Similar to Explore but with different optimization algorithms and more aggressive goals. Includes a SLR crossing optimization phase that is allowed to degrade WNS which should be regained in subsequent optimization algorithms. Also includes a hold violation fixing optimization.
        ImplementationStrategy.PhysOpt PhysOptPostAGGRESSIVE_EXPLORE = postRouteOpt;
        ImplementationStrategy.PhysOpt PhysOptAGGRESSIVE_FANOUT_OPT = postPlaceOpt; //Uses different algorithms for fanout-related optimizations with more aggressive goals.
        ImplementationStrategy.PhysOpt PhysOptALTERNATE_REPLICATION = postPlaceOpt; //Use different algorithms for performing critical cell replication.
        ImplementationStrategy.PhysOpt PhysOptALTERNATE_FLOW_WITH_RETIMING = postPlaceOpt; //Perform more aggressive replication and DSP and block RAM optimization, and enable register retiming.
        HashMap<String, ImplementationStrategy> strategies = new HashMap<>();
        strategies.put("VIVADO_DEFAULT", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptDISABLED, ImplementationStrategy.RouteOpt.DEFAULT, PhysOptDISABLED));
        strategies.put("PERFORMANCE_EXPLORE", ImplementationStrategy.createCustomStrategy(netlistOptEXPLORE, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXPLORE, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("PERFORMANCE_EXPLORE_POST_ROUTE_PHYS_OPT", ImplementationStrategy.createCustomStrategy(netlistOptEXPLORE, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXPLORE, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.createCustom(ImplementationStrategy.RouteOpt.EXPLORE, true, false, ""), PhysOptPostEXPLORE));
        strategies.put("PERFORMANCE_WL_BLOCK_PLACEMENT", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.WIRE_LENGTH_DRIVEN_BLOCK_PLACEMENT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptALTERNATE_REPLICATION, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("PERFORMANCE_WL_BLOCK_PLACEMENT_FANOUT_OPT", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.WIRE_LENGTH_DRIVEN_BLOCK_PLACEMENT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_FANOUT_OPT, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("PERFORMANCE_EARLY_BLOCK_PLACEMENT", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EARLY_BLOCK_PLACEMENT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptPostDEFAULT));
        strategies.put("PERFORMANCE_NET_DELAY_HIGH", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXTRA_NET_DELAY_HIGH, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("PERFORMANCE_NET_DELAY_LOW", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXTRA_NET_DELAY_LOW, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("PERFORMANCE_RETIMING", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXTRA_POST_PLACEMENT_OPT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptALTERNATE_FLOW_WITH_RETIMING, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("PERFORMANCE_EXTRA_TIMING_OPT", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXTRA_TIMING_OPT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("PERFORMANCE_REFINE_PLACEMENT", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXTRA_POST_PLACEMENT_OPT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("PERFORMANCE_REFINE_PLACEMENT_AGGEXPLORE_POST_ROUTE_PHYS_OPT", ImplementationStrategy.createCustomStrategy(netlistOptEXPLORE, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXTRA_POST_PLACEMENT_OPT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptPostEXPLORE));
        strategies.put("PERFORMANCE_SPREAD_SLLS", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_SPREAD_SLLS, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("PERFORMANCE_BALANCE_SLLS", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_BALANCE_SLLS, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("CONGESTION_SPREAD_LOGIC_HIGH", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.ALT_SPREAD_LOGIC_HIGH, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.ALTERNATE_CLB_ROUTING, PhysOptDISABLED));
        strategies.put("CONGESTION_SPREAD_LOGIC_MEDIUM", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.ALT_SPREAD_LOGIC_MEDIUM, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.ALTERNATE_CLB_ROUTING, PhysOptDISABLED));
        strategies.put("CONGESTION_SPREAD_LOGIC_LOW", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.ALT_SPREAD_LOGIC_LOW, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.ALTERNATE_CLB_ROUTING, PhysOptDISABLED));
        strategies.put("CONGESTION_SPREAD_LOGIC_EXPLORE", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.ALT_SPREAD_LOGIC_HIGH, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("CONGESTION_SSI_SPREAD_LOGIC_HIGH", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_SPREAD_LOGIC_HIGH, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.ALTERNATE_CLB_ROUTING, PhysOptDISABLED));
        strategies.put("CONGESTION_SSI_SPREAD_LOGIC_LOW", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_SPREAD_LOGIC_LOW, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.ALTERNATE_CLB_ROUTING, PhysOptDISABLED));
        strategies.put("CONGESTION_SSI_SPREAD_LOGIC_EXPLORE", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_SPREAD_LOGIC_HIGH, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("AREA_EXPLORE", ImplementationStrategy.createCustomStrategy(netlistOptEXPLORE_AREA, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptDISABLED, ImplementationStrategy.RouteOpt.DEFAULT, PhysOptDISABLED));
        strategies.put("AREA_EXPLORE_SEQUENTIAL", ImplementationStrategy.createCustomStrategy(netlistOptEXPLORE_SEQUENTIAL_AREA, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptDISABLED, ImplementationStrategy.RouteOpt.DEFAULT, PhysOptDISABLED));
        strategies.put("AREA_EXPLORE_WITH_REMAP", ImplementationStrategy.createCustomStrategy(netlistOptEXPLORE_WITH_REMAP, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptDISABLED, ImplementationStrategy.RouteOpt.DEFAULT, PhysOptDISABLED));
        strategies.put("POWER_DEFAULT_OPTS", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.ENABLED, ImplementationStrategy.PlaceOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptDISABLED, ImplementationStrategy.RouteOpt.DEFAULT, PhysOptDISABLED));
        strategies.put("POWER_EXPLORE_AREA", ImplementationStrategy.createCustomStrategy(netlistOptEXPLORE_SEQUENTIAL_AREA, ImplementationStrategy.PowerOpt.ENABLED, ImplementationStrategy.PlaceOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptDISABLED, ImplementationStrategy.RouteOpt.DEFAULT, PhysOptDISABLED));
        strategies.put("FLOW_RUN_PHYS_OPT", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptDEFAULT, ImplementationStrategy.RouteOpt.DEFAULT, PhysOptDISABLED));
        strategies.put("FLOW_RUN_POST_ROUTE_PHYS_OPT", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptDEFAULT, ImplementationStrategy.RouteOpt.createCustom(ImplementationStrategy.RouteOpt.DEFAULT, true, false, ""), PhysOptPostEXPLORE));
        strategies.put("FLOW_RUNTIME_OPTIMIZED", ImplementationStrategy.createCustomStrategy(netlistOptRUNTIME_OPTIMIZED, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.RUNTIME_OPTIMIZED, ImplementationStrategy.PowerOpt.DISABLED, PhysOptDISABLED, ImplementationStrategy.RouteOpt.RUNTIME_OPTIMIZED, PhysOptDISABLED));
        strategies.put("FLOW_QUICK", ImplementationStrategy.createCustomStrategy(netlistOptRUNTIME_OPTIMIZED, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.QUICK, ImplementationStrategy.PowerOpt.DISABLED, PhysOptDISABLED, ImplementationStrategy.RouteOpt.QUICK, PhysOptDISABLED));
        strategies.put("MAXELER1", ImplementationStrategy.createCustomStrategy(netlistOptNO_BRAM_POWER_OPT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXTRA_NET_DELAY_HIGH, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptPostAGGRESSIVE_EXPLORE));
        strategies.put("MAXELER2", ImplementationStrategy.createCustomStrategy(netlistOptNO_BRAM_POWER_OPT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXTRA_POST_PLACEMENT_OPT, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptPostAGGRESSIVE_EXPLORE));
        strategies.put("MAXELER3", ImplementationStrategy.createCustomStrategy(netlistOptEXPLORE, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.EXTRA_NET_DELAY_HIGH, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.createCustom(ImplementationStrategy.RouteOpt.ALTERNATE_CLB_ROUTING, true, false, ""), PhysOptPostAGGRESSIVE_EXPLORE));
        strategies.put("MAXELER4", ImplementationStrategy.createCustomStrategy(netlistOptEXPLORE, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_SPREAD_LOGIC_HIGH, ImplementationStrategy.PowerOpt.DISABLED, PhysOptAGGRESSIVE_EXPLORE, ImplementationStrategy.RouteOpt.createCustom(ImplementationStrategy.RouteOpt.ALTERNATE_CLB_ROUTING, true, false, ""), PhysOptPostAGGRESSIVE_EXPLORE));
        strategies.put("PERFORMANCE_BALANCE_SLRS", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_BALANCE_SLRS, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        strategies.put("PERFORMANCE_HIGH_UTIL_SLRS", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_HIGH_UTIL_SLRS, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        //strategies.put("AUTO_1", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PlaceOptAuto1, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        //strategies.put("AUTO_2", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PlaceOptAuto2, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        //strategies.put("AUTO_3", ImplementationStrategy.createCustomStrategy(netlistOptDEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PlaceOptAuto3, ImplementationStrategy.PowerOpt.DISABLED, PhysOptEXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, PhysOptDISABLED));
        
        return strategies;
    }
    public static ImplementationStrategy[] chooseImplementationStrategies(boolean customPhysOpt, boolean isProbe, boolean remap)
    {
        /*create_project p1 -force -part xcu250-figd2104-2L-e
        set steps [list opt place phys_opt route]
        set run [get_runs impl_1]
        join [list_property_value strategy [get_runs impl_1] ]
        foreach s $steps {
            puts "${s}_design Directives:"
            set dirs [list_property_value STEPS.${s}_DESIGN.ARGS.DIRECTIVE $run]
            set dirs [regsub -all {\s} $dirs \n]
            puts "$dirs\n"
        } 
        close_project -delete       
        */        
        if (!customPhysOpt) {
            //Auto_* do not exist until Vivado 2022.1
            //boolean timingDriven, boolean timingSummary, boolean unplace, boolean postPlaceOpt, boolean fanoutOpt, boolean bufgOpt, java.lang.String additionalOptions
            //The directive option is incompatible with other options with the exception of -no_fanout_opt, -no_bufg_opt, -quiet, and -verbose
            //ImplementationStrategy.PlaceOpt PlaceOptAuto1 = ImplementationStrategy.PlaceOpt.createCustom(true, false, false, false, false, true, "-directive Auto_1"); //High performing predicted directive
            //ImplementationStrategy.PlaceOpt PlaceOptAuto2 = ImplementationStrategy.PlaceOpt.createCustom(true, false, false, false, false, true, "-directive Auto_2"); //Second best predicted directive
            //ImplementationStrategy.PlaceOpt PlaceOptAuto3 = ImplementationStrategy.PlaceOpt.createCustom(true, false, false, false, false, true, "-directive Auto_3"); //Third best predicted directive
        
            //https://docs.xilinx.com/r/en-US/ug904-vivado-implementation/Directives-Used-by-opt_design-and-place_design-in-Implementation-Strategies
            //ImplementationStrategy PERFORMANCE_EXPLORE_POST_ROUTE_PHYS_OPT_CUSTOM = ImplementationStrategy.createCustomStrategy(ImplementationStrategy.NetlistOpt.EXPLORE, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_BALANCE_SLLS, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PhysOpt.EXPLORE, ImplementationStrategy.RouteOpt.createCustom(ImplementationStrategy.RouteOpt.EXPLORE, true, false, ""), ImplementationStrategy.PhysOpt.EXPLORE);
            ImplementationStrategy PERFORMANCE_BALANCE_SLRS = ImplementationStrategy.createCustomStrategy(ImplementationStrategy.NetlistOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_BALANCE_SLRS, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PhysOpt.EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, ImplementationStrategy.PhysOpt.DISABLED);
            ImplementationStrategy PERFORMANCE_HIGH_UTIL_SLRS = ImplementationStrategy.createCustomStrategy(ImplementationStrategy.NetlistOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PlaceOpt.SSI_HIGH_UTIL_SLRS, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PhysOpt.EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, ImplementationStrategy.PhysOpt.DISABLED);
            //ImplementationStrategy AUTO_1 = ImplementationStrategy.createCustomStrategy(ImplementationStrategy.NetlistOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PlaceOptAuto1, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PhysOpt.EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, ImplementationStrategy.PhysOpt.DISABLED);
            //ImplementationStrategy AUTO_2 = ImplementationStrategy.createCustomStrategy(ImplementationStrategy.NetlistOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PlaceOptAuto2, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PhysOpt.EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, ImplementationStrategy.PhysOpt.DISABLED);
            //ImplementationStrategy AUTO_3 = ImplementationStrategy.createCustomStrategy(ImplementationStrategy.NetlistOpt.DEFAULT, ImplementationStrategy.PowerOpt.DISABLED, PlaceOptAuto3, ImplementationStrategy.PowerOpt.DISABLED, ImplementationStrategy.PhysOpt.EXPLORE, ImplementationStrategy.RouteOpt.EXPLORE, ImplementationStrategy.PhysOpt.DISABLED);
        
            ImplementationStrategy[] probeStrategies = new ImplementationStrategy[] { //nearly all different placement strategies...
                //AUTO_1, AUTO_2, AUTO_3,
                //for congestion strategies, explore same placement as high, aggressive physical strategy
                ImplementationStrategy.CONGESTION_SPREAD_LOGIC_HIGH, ImplementationStrategy.CONGESTION_SPREAD_LOGIC_MEDIUM, ImplementationStrategy.CONGESTION_SPREAD_LOGIC_LOW, ImplementationStrategy.CONGESTION_SPREAD_LOGIC_EXPLORE,
                ImplementationStrategy.CONGESTION_SSI_SPREAD_LOGIC_HIGH, ImplementationStrategy.CONGESTION_SSI_SPREAD_LOGIC_LOW, ImplementationStrategy.CONGESTION_SSI_SPREAD_LOGIC_EXPLORE,
                PERFORMANCE_BALANCE_SLRS, PERFORMANCE_HIGH_UTIL_SLRS,
                ImplementationStrategy.PERFORMANCE_SPREAD_SLLS, ImplementationStrategy.PERFORMANCE_BALANCE_SLLS,
                ImplementationStrategy.PERFORMANCE_EXPLORE,
                ImplementationStrategy.PERFORMANCE_WL_BLOCK_PLACEMENT, ImplementationStrategy.PERFORMANCE_WL_BLOCK_PLACEMENT_FANOUT_OPT, //same placement strategy, different physical strategy
                ImplementationStrategy.PERFORMANCE_EARLY_BLOCK_PLACEMENT,
                ImplementationStrategy.PERFORMANCE_NET_DELAY_HIGH, ImplementationStrategy.PERFORMANCE_NET_DELAY_LOW,
                ImplementationStrategy.PERFORMANCE_RETIMING, ImplementationStrategy.PERFORMANCE_EXTRA_TIMING_OPT,
                ImplementationStrategy.PERFORMANCE_REFINE_PLACEMENT,
                ImplementationStrategy.VIVADO_DEFAULT,
                //ImplementationStrategy.FLOW_RUN_PHYS_OPT //default placement strategy
                ImplementationStrategy.FLOW_RUNTIME_OPTIMIZED, ImplementationStrategy.FLOW_QUICK,
                //ImplementationStrategy.AREA_EXPLORE, ImplementationStrategy.AREA_EXPLORE_SEQUENTIAL, ImplementationStrategy.AREA_EXPLORE_WITH_REMAP, //default placement strategy
                //ImplementationStrategy.POWER_DEFAULT_OPTS, ImplementationStrategy.POWER_EXPLORE_AREA, //default placement strategy
            };
            
            ImplementationStrategy[] postOptStrategies = new ImplementationStrategy[] {
                ImplementationStrategy.PERFORMANCE_REFINE_PLACEMENT_AGGEXPLORE_POST_ROUTE_PHYS_OPT,
                ImplementationStrategy.PERFORMANCE_EXPLORE_POST_ROUTE_PHYS_OPT,
                ImplementationStrategy.MAXELER1, ImplementationStrategy.MAXELER2,
                ImplementationStrategy.MAXELER3, ImplementationStrategy.MAXELER4
            };
            return isProbe ? probeStrategies : postOptStrategies;
        } else {
        
            HashMap<String, ImplementationStrategy> strategies = getImplementationStrategies(remap);
            
            ImplementationStrategy[] probeStrategies = new ImplementationStrategy[] { //nearly all different placement strategies...
                //for congestion strategies, explore same placement as high, aggressive physical strategy
                //strategies.get("AUTO_1"), strategies.get("AUTO_2"), strategies.get("AUTO_3"),
                strategies.get("CONGESTION_SPREAD_LOGIC_HIGH"), strategies.get("CONGESTION_SPREAD_LOGIC_MEDIUM"), strategies.get("CONGESTION_SPREAD_LOGIC_LOW"), strategies.get("CONGESTION_SPREAD_LOGIC_EXPLORE"),
                strategies.get("CONGESTION_SSI_SPREAD_LOGIC_HIGH"), strategies.get("CONGESTION_SSI_SPREAD_LOGIC_LOW"), strategies.get("CONGESTION_SSI_SPREAD_LOGIC_EXPLORE"),
                strategies.get("PERFORMANCE_BALANCE_SLRS"), strategies.get("PERFORMANCE_HIGH_UTIL_SLRS"),
                strategies.get("PERFORMANCE_SPREAD_SLLS"), strategies.get("PERFORMANCE_BALANCE_SLLS"),
                strategies.get("PERFORMANCE_EXPLORE"),
                strategies.get("PERFORMANCE_WL_BLOCK_PLACEMENT"), strategies.get("PERFORMANCE_WL_BLOCK_PLACEMENT_FANOUT_OPT"), //same placement strategy, different physical strategy
                strategies.get("PERFORMANCE_EARLY_BLOCK_PLACEMENT"),
                strategies.get("PERFORMANCE_NET_DELAY_HIGH"), strategies.get("PERFORMANCE_NET_DELAY_LOW"),
                strategies.get("PERFORMANCE_RETIMING"), strategies.get("PERFORMANCE_EXTRA_TIMING_OPT"),
                strategies.get("PERFORMANCE_REFINE_PLACEMENT"),
                strategies.get("VIVADO_DEFAULT"),
                //strategies.get("FLOW_RUN_PHYS_OPT") //default placement strategy
                strategies.get("FLOW_RUNTIME_OPTIMIZED"), strategies.get("FLOW_QUICK"),
                //strategies.get("AREA_EXPLORE"), strategies.get("AREA_EXPLORE_SEQUENTIAL"), strategies.get("AREA_EXPLORE_WITH_REMAP"), //default placement strategy
                //strategies.get("POWER_DEFAULT_OPTS"), strategies.get("POWER_EXPLORE_AREA"), //default placement strategy
            };
            
            ImplementationStrategy[] postOptStrategies = new ImplementationStrategy[] {
                strategies.get("PERFORMANCE_REFINE_PLACEMENT_AGGEXPLORE_POST_ROUTE_PHYS_OPT"),
                strategies.get("PERFORMANCE_EXPLORE_POST_ROUTE_PHYS_OPT"),
                strategies.get("MAXELER1"), strategies.get("MAXELER2"),
                strategies.get("MAXELER3"), strategies.get("MAXELER4")
            };
            return isProbe ? probeStrategies : postOptStrategies; 
        }                
        //PERFORMANCE_REFINE_PLACEMENT -> PERFORMANCE_REFINE_PLACEMENT_AGGEXPLORE_POST_ROUTE_PHYS_OPT
        //PERFORMANCE_EXPLORE -> PERFORMANCE_EXPLORE_POST_ROUTE_PHYS_OPT
        //FLOW_RUN_PHYS_OPT -> FLOW_RUN_POST_ROUTE_PHYS_OPT
        //PERFORMANCE_NET_DELAY_HIGH -> MAXELER1, MAXELER3
        //PERFORMANCE_RETIMING -> MAXELER2
        //CONGESTION_SSI_SPREAD_LOGIC_HIGH -> MAXELER4
    } 
    public static SynthesisStrategy getHighPerfSynthStrategy()
    {
        //SynthesisStrategy.VIVADO_DEFAULT, SynthesisStrategy.FLOW_ALTERNATE_ROUTABILITY, SynthesisStrategy.FLOW_AREA_MULT_THRESHOLD_DSP,
        //SynthesisStrategy.FLOW_AREA_OPTIMIZED_HIGH, SynthesisStrategy.FLOW_AREA_OPTIMIZED_MEDIUM, SynthesisStrategy.FLOW_PERF_OPTIMIZED_HIGH,
        //SynthesisStrategy.FLOW_PERF_THRESHOLD_CARRY, SynthesisStrategy.FLOW_RUNTIME_OPTIMIZED
        int bufg = 24; //base on number of streams, rather than default of 12, UltraScale maximum of 24
        int shreg_min_size = 4; //instead of 3, minimum chain before changing from registers to an SRL, breaks 3-stage fanout optimizations at 3 so higher better (shift register lookup table)
        int fanoutLimit = 32; //400 is the performance default
        //rather than FsmExtraction.ONE_HOT, AUTO may be better
        //Directive.ALTERNATE_ROUTABILITY does not seem to be better than Directive.DEFAULT
        //SynthesisStrategy.Directive directive, SynthesisStrategy.FlattenHierarchy flattenHierarchy, SynthesisStrategy.FsmExtraction fsmExtraction, SynthesisStrategy.CascadeDsp cascadeDsp, SynthesisStrategy.GatedClock gatedClock, SynthesisStrategy.ResourceSharing resourceSharing, boolean keepEquivalentRegisters, boolean lutCombining, boolean srlExtract, boolean retiming, int bufg, int fanoutLimit, int controlSetOptThreshold, int shregMinSize, int maxBram, int maxBramCascadeHeight, int maxUram, int maxUramCascadeHeight, int maxDsp, boolean vhdlAssert, boolean oocMode, java.lang.String options
        SynthesisStrategy customPerfOptHigh = SynthesisStrategy.createCustomStrategy(SynthesisStrategy.Directive.DEFAULT, SynthesisStrategy.FlattenHierarchy.REBUILT, SynthesisStrategy.FsmExtraction.AUTO, SynthesisStrategy.CascadeDsp.AUTO, SynthesisStrategy.GatedClock.OFF, SynthesisStrategy.ResourceSharing.OFF, true, true, true, false, bufg, fanoutLimit, -1, shreg_min_size, -1, -1, -1, -1, -1, false, false, "");
        return customPerfOptHigh;
    }
    public static void printKernelConfigurationOptimizationDefaults(ManagerKernelBase owner)
    {
        OptimizationOptions opt = owner.getCurrentKernelConfig().optimization;
        System.out.println("CEPipelining: " + opt.getCEPipelining()); //2
        System.out.println("CEReplicationNumPartitions: " + opt.getCEReplicationNumPartitions()); //0
        System.out.println("ClockPhaseBalanceThreshold: " + opt.getClockPhaseBalanceThreshold()); //0.1
        System.out.println("ClockPhaseRetries: " + opt.getClockPhaseRetries()); //50
        System.out.println("ConditionalArithmeticEnabled: " + opt.getConditionalArithmeticEnabled()); //true
        System.out.println("ConstantMultiplicationWithShiftAddThreshold: " + opt.getConstantMultiplicationWithShiftAddThreshold()); //3
        System.out.println("DeleteRedundantNodes: " + opt.getDeleteRedundantNodes()); //true
        System.out.println("DSPMulAddChainBehavior: " + opt.getDSPMulAddChainBehavior()); //OPTIMISE
        System.out.println("FIFOCoalescingEnabled: " + opt.getFIFOCoalescingEnabled()); //true
        System.out.println("FIFOImplementationBRAMThreshold: " + opt.getFIFOImplementationBRAMThreshold()); //2080
        System.out.println("Inlining: " + opt.getInlining()); //false
        System.out.println("MinimumStaticFIFOSplitDepth: " + opt.getMinimumStaticFIFOSplitDepth()); //1
        System.out.println("NumberOfPacketsInFlight: " + opt.getNumberOfPacketsInFlight()); //1
        System.out.println("OptimizationTechnique: " + opt.getOptimizationTechnique()); //DEFAULT
        System.out.println("OptimizePowerTwoFloatMultEnabled: " + opt.getOptimizePowerTwoFloatMultEnabled()); //true
        System.out.println("PreserveNodeRegisters: " + opt.getPreserveNodeRegisters()); //true
        System.out.println("ROMImplementationBRAMThreshold: " + opt.getROMImplementationBRAMThreshold()); //2080
        System.out.println("TriAddsEnabled: " + opt.getTriAddsEnabled()); //true
        System.out.println("UseAsapScheduler: " + opt.getUseAsapScheduler()); //false
        System.out.println("UseGlobalClockBuffer: " + opt.getUseGlobalClockBuffer()); //false
        System.out.println("UseGlobalClockLinesRst: " + opt.getUseGlobalClockLinesRst()); //false
    }
    public static void selectSLR(int slr_index, XilinxAlveoU250Manager owner) {
        XilinxAlveoU250Manager.NamedRegion[] availableRegions = XilinxAlveoU250Manager.NamedRegion.values();
        if (slr_index < 0 || slr_index >= availableRegions.length) {
            throw new IllegalArgumentException();
        }
        owner.pushNamedRegion(availableRegions[slr_index]);
    }
    public static void unselectSLR(XilinxAlveoU250Manager owner) { owner.popRegion(); }
    public static void printManagerConfigurationDefaults(BuildConfig conf)
    {
        System.out.println("Bram Mapping Scale Factor: " + conf.getBramMappingScaleFactor()); //1.0
        System.out.println("Build Effort: " + conf.getBuildEffort()); //MEDIUM
        System.out.println("Build Level: " + conf.getBuildLevel()); //FULL_BUILD
        System.out.println("Continue After Meeting Timing: " + conf.getContinueAfterMeetingTiming()); //false
        System.out.println("Enable Chipscope Inserter: " + conf.getEnableChipscopeInserter()); //false
        System.out.println("Enable Gen Max File: " + conf.getEnableGenMaxFile()); //true
        System.out.println("Enable Pcie Ibert: " + conf.getEnablePcieIbert()); //false
        System.out.println("Enable Pcie Jtag: " + conf.getEnablePcieJtag()); //false
        System.out.println("Enable Tandom Boot: " + conf.getEnableTandemBoot()); //false
        System.out.println("Enable Timing Analysis: " + conf.getEnableTimingAnalysis()); //true
        System.out.println("Enable Xpm Macro: " + conf.getEnableXpmMacro()); //true
        System.out.println("External Xdc Files: " + conf.getExternalXdcFiles()); //[]
        System.out.println("Generate Hex Bit File: " + conf.getGenerateHexBitfile()); //false
        System.out.println("Implementation Near Miss Retries: " + conf.getImplementationNearMissRetries()); //0
        System.out.println("Implementation Near Miss Threshold: " + conf.getImplementationNearMissThreshold()); //0
        System.out.println("Keep Wrapper Node Hierarchy: " + conf.getKeepWrapperNodeHierarchy()); //true
        System.out.println("Logic Mapping Scale Factor: " + conf.getLogicMappingScaleFactor()); //1.0
        System.out.println("Optimization Goal: " + conf.getOptimizationGoal()); //BALANCED
        System.out.println("Override System Jitter: " + conf.getOverrideSystemJitter()); //false
        System.out.println("Parallelism: " + conf.getParallelism()); //1
        System.out.println("Power Consumption Critical Warning Threshold: " + conf.getPowerConsumptionCriticalWarningThreshold()); //112.5
        System.out.println("Power Consumption Error Threshold: " + conf.getPowerConsumptionErrorThreshold()); //135.0
        System.out.println("Power Consumption Warning Threshold: " + conf.getPowerConsumptionWarningThreshold()); //112.5
        System.out.println("Suppress Uram Cascading Warning: " + conf.getSuppressUramCascadingWarning()); //false
        System.out.println("System Jitter: " + conf.getSystemJitter()); //0.2        
    }
}
