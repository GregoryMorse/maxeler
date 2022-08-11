package com.maxeler.photon.graph_passes.scheduler;

import java.util.Iterator;
import com.maxeler.photon.core.VarTyped;
import com.maxeler.photon.nodes.NodeFIFO;
import com.maxeler.photon.nodes.NodeRegister;
import com.maxeler.photon.software.EqKernelPlaceholder;
import com.maxeler.photon.software.SoftwareEq;
import java.util.Set;
import com.maxeler.photon.nodes.NodeUnscheduled;
import com.maxeler.photon.core.Var;
import java.util.Collection;
import com.maxeler.photon.core.StreamOffsetEq;
import com.maxeler.photon.core.PhotonDesignData;
import com.maxeler.photon.core.Node;
import com.maxeler.photon.core.GraphPassPointwise;

public class ScheduleApplier implements GraphPassPointwise
{
    private final Schedule m_schedule;
    
    public ScheduleApplier(final Schedule schedule) {
        this.m_schedule = schedule;
    }
    
    @Override
    public Direction direction() {
        return Direction.FORWARDS;
    }
    
    @Override
    public void processNode(final Node node, final PhotonDesignData photonDesignData) {
        final StreamOffsetEq inputLatency = this.m_schedule.getOffset(node);
        if (inputLatency == null) {
            return;
        }
        node.setInputLatency(inputLatency);
        for (final Node.InputDesc inputDesc : node.getInputDescs()) {
            final Node node2 = inputDesc.getVar().getSrcNode();
            if (node2 instanceof NodeUnscheduled) {
                continue;
            }
            StreamOffsetEq streamOffsetEq = this.m_schedule.getOffset(node2);
            if (streamOffsetEq == null) {
                continue;
            }
            StreamOffsetEq streamOffsetEq2 = inputDesc.getVar().getSrcOutputDesc().getLatency();
            if (!this.m_schedule.isComplete()) {
                if (streamOffsetEq2.hasPlaceholderTerms()) {
                    for (final SoftwareEq softwareEq : streamOffsetEq2.getUsedVariables()) {
                        if (softwareEq instanceof EqKernelPlaceholder) {
                            final int coefficientForVariable = streamOffsetEq2.getCoefficientForVariable(softwareEq);
                            streamOffsetEq2 = streamOffsetEq2.add(-coefficientForVariable, softwareEq).add(((EqKernelPlaceholder)softwareEq).getPartialEq().mul(coefficientForVariable));
                        }
                    }
                }
                if (streamOffsetEq2.getC() != 0) {
                    streamOffsetEq2 = streamOffsetEq2.sub(streamOffsetEq2.getC());
                }
                if (streamOffsetEq.getC() != 0) {
                    streamOffsetEq = streamOffsetEq.sub(streamOffsetEq.getC());
                }
            }
            final StreamOffsetEq inputLatency2 = streamOffsetEq.add(streamOffsetEq2);
            final StreamOffsetEq streamOffsetEq3 = inputLatency.sub(inputLatency2);
            if (!streamOffsetEq3.hasSymbolicTerms() && streamOffsetEq3.getC() == 0) {
                continue;
            }
            if (streamOffsetEq3.max() == 0 && streamOffsetEq3.min() == 0) {
                continue;
            }
            if (streamOffsetEq3.max() == streamOffsetEq3.min()) {
                Node lastNode = null;
                StreamOffsetEq curDelta = streamOffsetEq3;
                StreamOffsetEq curOffset = inputLatency2;
                boolean found;
                do {
                    found = false;
                    for (final Node.OutputDesc outputDesc : (lastNode == null ? node2 : lastNode).getOutputDescs()) {
                        for (final Node.InputDesc inDesc : outputDesc.getVar().getDstInputDescs()) {
                            Node checkNode = inDesc.getNode();
                            if (checkNode instanceof NodeRegister && checkNode.getInputLatency().max() == curOffset.max()) {
                                lastNode = checkNode;
                                StreamOffsetEq lat = checkNode.getOutputDesc("output").getLatency();                                
                                curDelta = curDelta.sub(lat);
                                curOffset = curOffset.add(lat);
                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }
                } while (curDelta.max() != 0 && found);
                System.out.println("Replacing FIFO(" + streamOffsetEq3.max() + " -> " + curDelta.max() + ") with pipeline registers");
                if (curDelta.max() == 0) {
                    node.connectInput(inputDesc.getName(), lastNode.connectOutput("output"));
                }
                for (int i = 0; i < curDelta.max(); i++) { 
                    final NodeRegister nodeReg = new NodeRegister(photonDesignData, inputDesc.getVar().getSrcOutputDesc().getMaxFanout());
                    nodeReg.setConstant(false);
                    nodeReg.setInputLatency(curOffset.add(i));
                    nodeReg.setOperatorSupplier(photonDesignData.getOperatorSupplier());
                    this.m_schedule.add(nodeReg);
                    nodeReg.connectInput("input", lastNode == null ? inputDesc.getVar() : lastNode.connectOutput("output"));
                    lastNode = nodeReg;
                    if (i == curDelta.max()-1) 
                        node.connectInput(inputDesc.getName(), nodeReg.connectOutput("output"));
                }
                continue;
            }
            final NodeFIFO nodeFIFO = new NodeFIFO(photonDesignData, node.getGroupPath(), streamOffsetEq3);
            nodeFIFO.setConstant(false);
            nodeFIFO.setInputLatency(inputLatency2);
            this.m_schedule.add(nodeFIFO);
            final Node.InputDesc inputDesc2 = nodeFIFO.connectInput("input", inputDesc.getVar());
            final Node.InputDesc inputDesc3 = node.connectInput(inputDesc.getName(), nodeFIFO.connectOutput("output"));
        }
    }
    
    @Override
    public void end(final PhotonDesignData photonDesignData) {
    }
    
    @Override
    public ReprocessMode reProcessOnChange() {
        return ReprocessMode.NEVER;
    }
}
