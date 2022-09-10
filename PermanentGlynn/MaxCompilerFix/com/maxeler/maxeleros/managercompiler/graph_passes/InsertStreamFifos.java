package com.maxeler.maxeleros.managercompiler.graph_passes;

import com.maxeler.maxeleros.managercompiler.libs.PlacementConstraint;
import com.maxeler.conf.base.BuildConfOption;
import com.maxeler.conf.base.MaxCompilerBuildConf;
import com.maxeler.maxdc.BuildManager;
import com.maxeler.maxeleros.managercompiler.nodes.WrapperNodeDualAspectMux;
import com.maxeler.maxeleros.managercompiler.nodes.WrapperNodeDualAspectReg;
import com.maxeler.maxeleros.managercompiler.core.WrapperClock;
import com.maxeler.maxeleros.managercompiler.core.Stream;
import com.maxeler.maxeleros.StreamingInterfaceTypePull;
import com.maxeler.maxeleros.StreamingInterfaceTypePush;
import com.maxeler.photon.software.SoftwareEq;
import com.maxeler.maxeleros.StreamingInterfaceType;
import java.util.Iterator;
import com.maxeler.maxeleros.managercompiler.nodes.WrapperNodeFifo;
import java.util.Collection;
import java.util.LinkedList;
import com.maxeler.maxeleros.managercompiler.core.WrapperNode;
import com.maxeler.maxeleros.managercompiler.core.WrapperGraphMod;
import com.maxeler.maxeleros.managercompiler.core.WrapperDesignData;
import com.maxeler.maxeleros.managercompiler.core.WrapperGraphPassPointwise;

public class InsertStreamFifos implements WrapperGraphPassPointwise
{
    @Override
    public Direction direction() {
        return Direction.FORWARDS;
    }
    
    @Override
    public WrapperGraphMod end(final WrapperDesignData wrapperDesignData) {
        return null;
    }
    
    @Override
    public WrapperGraphMod processNode(final WrapperNode wrapperNode, final WrapperDesignData wrapperDesignData) {
        final LinkedList<WrapperNode> list = new LinkedList<>();
        final Iterator<WrapperNode.IODesc> iterator = wrapperNode.iterateInputDescs().iterator();
        while (iterator.hasNext()) {
            final WrapperNodeFifo wrapperNodeFifo = this.processInput(iterator.next(), wrapperDesignData);
            if (wrapperNodeFifo != null) {
                list.add(wrapperNodeFifo);
            }
        }
        if (list.size() == 0) {
            return null;
        }
        return new WrapperGraphMod((WrapperNode[])list.toArray(new WrapperNode[0]), new WrapperNode[0]);
    }
    
    private boolean incompatibleTypes(final StreamingInterfaceType streamingInterfaceType, final StreamingInterfaceType streamingInterfaceType2) {
        if (streamingInterfaceType.getFlowControlType() != streamingInterfaceType2.getFlowControlType()) {
            return true;
        }
        if (!streamingInterfaceType.hasReadDataCount() && streamingInterfaceType2.hasReadDataCount()) {
            return true;
        }
        if (streamingInterfaceType.getFlowControlType() == StreamingInterfaceType.FlowControlType.PUSH) {
            if (((StreamingInterfaceTypePush)streamingInterfaceType2).getStallLatency().eval() < ((StreamingInterfaceTypePush)streamingInterfaceType).getStallLatency().eval()) {
                return true;
            }
        }
        else {
            final StreamingInterfaceTypePull streamingInterfaceTypePull = (StreamingInterfaceTypePull)streamingInterfaceType2;
            final StreamingInterfaceTypePull streamingInterfaceTypePull2 = (StreamingInterfaceTypePull)streamingInterfaceType;
            if (streamingInterfaceTypePull.hasAlmostEmpty()) {
                if (!streamingInterfaceTypePull2.hasAlmostEmpty()) {
                    return true;
                }
                if (streamingInterfaceTypePull2.getAlmostEmptyLatency().eval() < streamingInterfaceTypePull.getAlmostEmptyLatency().eval()) {
                    return true;
                }
            }
            if (streamingInterfaceTypePull2.getEmptyLatency().eval() < streamingInterfaceTypePull.getEmptyLatency().eval()) {
                return true;
            }
        }
        return this.calculateBufferSize(streamingInterfaceType, streamingInterfaceType2) > 0;
    }
    
    private int calculateBufferSize(final StreamingInterfaceType streamingInterfaceType, final StreamingInterfaceType streamingInterfaceType2) {
        return Math.max(streamingInterfaceType.getCustomDepth(), streamingInterfaceType2.getCustomDepth());
    }
    
    private WrapperNodeFifo processInput(final WrapperNode.IODesc ioDesc, final WrapperDesignData wrapperDesignData) {
        final WrapperNode.IODesc ioDesc2 = ioDesc.getStream().getSource();
        final StreamingInterfaceType streamingInterfaceType = ioDesc2.getStreamingInterfaceType();
        final StreamingInterfaceType streamingInterfaceType2 = ioDesc.getStreamingInterfaceType();
        final boolean b = this.incompatibleTypes(streamingInterfaceType, streamingInterfaceType2) | ioDesc2.getWidth() != ioDesc.getWidth();
        final WrapperClock inputClock = ioDesc2.getClock();
        final WrapperClock outputClock = ioDesc.getClock();
        if (b | inputClock != outputClock | (ioDesc2.getParentNode() instanceof WrapperNodeDualAspectReg && ioDesc.getParentNode() instanceof WrapperNodeDualAspectMux)) {
            final String s = ioDesc.getStream().getName();
            final int calculateBufferSize = this.calculateBufferSize(streamingInterfaceType, streamingInterfaceType2);
            streamingInterfaceType.setCustomFifoDepth(0);
            streamingInterfaceType2.setCustomFifoDepth(0);
            final WrapperNodeFifo wrapperNodeFifo = new WrapperNodeFifo(wrapperDesignData, s, "input", "output", streamingInterfaceType, streamingInterfaceType2, calculateBufferSize);
            wrapperNodeFifo.setInputClock(inputClock);
            wrapperNodeFifo.setOutputClock(outputClock);
            wrapperNodeFifo.setInputWidth(ioDesc2.getWidth());
            wrapperNodeFifo.setOutputWidth(ioDesc.getWidth());
            wrapperNodeFifo.getInput().reconnect(ioDesc2.getStream());
            ioDesc.getParentNode().reconnectInput(ioDesc, wrapperNodeFifo.getOutput());
            if (ioDesc2.getParentNode().getNodeGroup() == ioDesc.getParentNode().getNodeGroup()) {
                wrapperNodeFifo.setNodeGroup(ioDesc.getParentNode().getNodeGroup());
            }
            if (wrapperDesignData.getBuildManager().getParameter((BuildConfOption<Boolean>)MaxCompilerBuildConf.manager.auto_constraints)) {
                final PlacementConstraint placementConstraint = ioDesc2.getParentNode().getPlacementConstraint();
                if (placementConstraint != null && placementConstraint.equals(ioDesc.getParentNode().getPlacementConstraint())) {
                    wrapperNodeFifo.setPlacementConstraint(placementConstraint);
                // BEGIN CODE ADDITION FOR STREAM FIFO OUTPUT PLACEMENT CORRECTION                    
                } else if (ioDesc.getParentNode().getPlacementConstraint() != null) {
                    wrapperDesignData.getBuildManager().logInfo("Correcting StreamFIFO '" + s + "' placement to output"); 
                    wrapperNodeFifo.setPlacementConstraint(ioDesc.getParentNode().getPlacementConstraint());
                // END CODE ADDITION FOR STREAM FIFO OUTPUT PLACEMENT CORRECTION
                }
            }
            return wrapperNodeFifo;
        }
        return null;
    }
}
