package com.maxeler.photon.maxcompilersim.formatterBitAccurate;

import java.util.Iterator;
import java.util.List;
import com.maxeler.photon.maxcompilersim.StatementCall;
import com.maxeler.photon.maxcompilersim.CVarHWExcept;
import com.maxeler.photon.maxcompilersim.StatementDivMod;
import com.maxeler.photon.maxcompilersim.StatementWarning;
import com.maxeler.photon.maxcompilersim.StatementPrintf;
import com.maxeler.photon.configuration.PhotonKernelConfiguration;
import com.maxeler.photon.maxcompilersim.StatementTryCatch;
import com.maxeler.photon.maxcompilersim.StatementException;
import java.util.ArrayList;
import java.util.StringTokenizer;
import com.maxeler.photon.core.PhotonDesignData;
import com.maxeler.photon.types.HWType;
import com.maxeler.photon.maxcompilersim.StatementBitReverse;
import com.maxeler.photon.maxcompilersim.StatementLiteral;
import com.maxeler.photon.maxcompilersim.ExpConstantUndefined;
import com.maxeler.photon.maxcompilersim.CVarImplicit;
import com.maxeler.photon.maxcompilersim.SimCodeException;
import com.maxeler.photon.maxcompilersim.StatementSelect;
import com.maxeler.photon.maxcompilersim.StatementIf;
import com.maxeler.photon.maxcompilersim.StatementOutput;
import com.maxeler.photon.maxcompilersim.CArrayAccess;
import com.maxeler.photon.maxcompilersim.ExpLiteral;
import com.maxeler.photon.maxcompilersim.SimCodeType;
import com.maxeler.photon.maxcompilersim.CArray;
import com.maxeler.photon.maxcompilersim.StatementArrayAssign;
import com.maxeler.photon.maxcompilersim.COutput;
import com.maxeler.photon.maxcompilersim.CodeBlock;
import com.maxeler.photon.maxcompilersim.CodeBlockRoot;
import com.maxeler.photon.core.Var;
import com.maxeler.photon.maxcompilersim.CodeContext;
import com.maxeler.photon.core.Node;
import com.maxeler.photon.maxcompilersim.CVarState;
import com.maxeler.photon.maxcompilersim.Expression;
import com.maxeler.photon.maxcompilersim.ExpLValue;
import com.maxeler.photon.maxcompilersim.StatementAssign;
import com.maxeler.photon.maxcompilersim.formatterBase.SimCodeFormatter;
import com.maxeler.photon.maxcompilersim.formatterBase.CodeBuilder;
import com.maxeler.photon.maxcompilersim.formatterBase.FormatExpression;
import com.maxeler.photon.maxcompilersim.formatterBase.FormatStatement;

public class FormatStatementsBitAccurate extends FormatStatement
{
    private final FormatterBitAccurate m_formatter;
    
    public static CodeBuilder create(final FormatterBitAccurate formatterBitAccurate, final FormatExpression formatExpression, final int n) {
        return new FormatStatementsBitAccurate(formatterBitAccurate, formatExpression, n).getCodeBuilder();
    }
    
    public FormatStatementsBitAccurate(final FormatterBitAccurate formatter, final FormatExpression formatExpression, final int n) {
        super(formatter, formatExpression, n);
        this.m_formatter = formatter;
    }
    
    @Override
    public String format(final StatementAssign statementAssign) {
        this.m_cb.push();
        this.m_cb.addLine(String.valueOf(this.getStateAssignCond(statementAssign)) + statementAssign.getVar().format(this.m_expf) + " = " + statementAssign.getExpression().format(this.m_expf) + ";");
        return this.m_cb.pop();
    }
    
    private String getStateAssignCond(final StatementAssign statementAssign) {
        if (statementAssign.getVar() instanceof CVarState) {
            final Node node = ((CVarState)statementAssign.getVar()).getCodeContext().getNode();
            if (node.getStall() != null && statementAssign.getCodeBlock().getRoot().getType() == CodeBlockRoot.CodeBlockRootType.EXECUTE) {
                final Node.OutputDesc outputDesc = node.getStall().getSrcOutputDesc();
                return "if (!" + this.m_formatter.getCodeContextForNode(outputDesc.getNode()).getOutputVar(outputDesc.getName()).getName() + ".getValueAsBool()) ";
            }
        }
        return "";
    }
    
    @Override
    public String format(final StatementArrayAssign statementArrayAssign) {
        this.m_cb.push();
        final CArray cArray = statementArrayAssign.getArray();
        final CArrayAccess cArrayAccess = new CArrayAccess(cArray, new ExpLiteral("i", SimCodeType.SwLong));
        final Expression expression = statementArrayAssign.getExpression();
        final String s = this.getStateArrayAssignCond(statementArrayAssign);
        if (!s.isEmpty()) {
            this.m_cb.addLine(String.valueOf(s) + " {");
            final String s2 = this.m_cb.indent();
        }
        this.m_cb.addLine("for(int i=0;i<" + cArray.getSize() + ";i++)");
        final String s3 = this.m_cb.indent(1);
        this.m_cb.addLine(String.valueOf(cArrayAccess.format(this.m_expf)) + " = " + expression.format(this.m_expf) + ";");
        final String s4 = this.m_cb.indent(-1);
        if (!s.isEmpty()) {
            final String s5 = this.m_cb.indent(-1);
        }
        return this.m_cb.pop();
    }
    
    private String getStateArrayAssignCond(final StatementArrayAssign statementArrayAssign) {
        final Node node = statementArrayAssign.getArray().getCodeContext().getNode();
        if (node.getStall() != null && statementArrayAssign.getCodeBlock().getRoot().getType() == CodeBlockRoot.CodeBlockRootType.EXECUTE) {
            final Node.OutputDesc outputDesc = node.getStall().getSrcOutputDesc();
            return "if (!" + this.m_formatter.getCodeContextForNode(outputDesc.getNode()).getOutputVar(outputDesc.getName()).getName() + ".getValueAsBool()) ";
        }
        return "";
    }
    
    @Override
    public String format(final StatementOutput statementOutput) {
        this.m_cb.push();
        final COutput cOutput = statementOutput.getOutRef();
        if (cOutput.getOutputLatency() == 0) {
            this.m_cb.addLine(String.valueOf(cOutput.getName()) + " = " + statementOutput.getExpression().format(this.m_expf) + ";");
        }
        else if (statementOutput.getCodeBlock().getRoot().getType() == CodeBlockRoot.CodeBlockRootType.EXECUTE) {
            this.m_cb.addLine(String.valueOf(cOutput.getName()) + ("[(" + this.m_formatter.getEnabledCyclesCountForNode(cOutput.getNode()) + "+" + cOutput.getOutputLatency() + ")%" + (cOutput.getOutputLatency() + 1) + "]") + " = " + statementOutput.getExpression().format(this.m_expf) + ";");
        }
        else {
            this.m_cb.addLine("for(int i=0; i<" + (cOutput.getOutputLatency() + 1) + "; i++)");
            this.m_cb.addLine("{");
            final String s = this.m_cb.indent(1);
            this.m_cb.addLine(String.valueOf(cOutput.getName()) + "[i] = " + statementOutput.getExpression().format(this.m_expf) + ";");
            final String s2 = this.m_cb.indent(-1);
            this.m_cb.addLine("}");
        }
        return this.m_cb.pop();
    }
    
    @Override
    public String format(final StatementIf statementIf) {
        this.m_cb.push();
        this.m_cb.addLine("if(" + statementIf.getCondition().cast(SimCodeType.SwBool).format(this.m_expf) + ") {");
        final String s = this.m_cb.indent(1);
        this.m_cb.addCodeBlock(statementIf.getThen());
        final String s2 = this.m_cb.indent(-1);
        this.m_cb.addLine("}");
        if (statementIf.getElse() != null) {
            this.m_cb.addLine("else {");
            final String s3 = this.m_cb.indent(1);
            this.m_cb.addCodeBlock(statementIf.getElse());
            final String s4 = this.m_cb.indent(-1);
            this.m_cb.addLine("}");
        }
        return this.m_cb.pop();
    }
    
    @Override
    public String format(final StatementSelect statementSelect) {
        this.m_cb.push();
        if (statementSelect.isOneHot() && statementSelect.getOptions().length > 64) {
            throw new SimCodeException("Simulation currently doesn't support one-hot muxes with more then 64 alternatives", new Object[0]);
        }
        this.m_cb.addLine("switch(" + statementSelect.getWhich().cast(SimCodeType.SwLong).format(this.m_expf) + ") {");
        final String s = this.m_cb.indent(1);
        int n = 0;
        Expression[] array;
        for (int length = (array = statementSelect.getOptions()).length, i = 0; i < length; ++i) {
            final Expression expression = array[i];
            this.m_cb.addLine("case " + (statementSelect.isOneHot() ? (n == 63 ? ((1L << n) + 1) + "l - 1" : (1L << n)) : n) + "l:");
            final String s2 = this.m_cb.indent(1);
            this.m_cb.addLine(String.valueOf(statementSelect.getVar().getName()) + " = " + expression.format(this.m_expf) + ";");
            this.m_cb.addLine("break;");
            final String s3 = this.m_cb.indent(-1);
            ++n;
        }
        Expression expression2;
        if (statementSelect.hasDefault()) {
            expression2 = statementSelect.getDefault();
        }
        else {
            expression2 = statementSelect.getCodeBlock().undefined(statementSelect.getVar().getType());
        }
        this.m_cb.addLine("default:");
        final String s4 = this.m_cb.indent(1);
        this.m_cb.addLine(String.valueOf(statementSelect.getVar().getName()) + " = " + expression2.format(this.m_expf) + ";");
        this.m_cb.addLine("break;");
        final String s5 = this.m_cb.indent(-1);
        final String s6 = this.m_cb.indent(-1);
        this.m_cb.addLine("}");
        return this.m_cb.pop();
    }
    
    @Override
    public String format(final StatementLiteral statementLiteral) {
        this.m_cb.push();
        this.m_cb.addLine(statementLiteral.getStatement());
        return this.m_cb.pop();
    }
    
    @Override
    public String format(final StatementBitReverse statementBitReverse) {
        this.m_cb.push();
        final int totalBits = statementBitReverse.getInput().getType().getHWType().getTotalBits();
        final String s = statementBitReverse.getInput().format(this.m_expf);
        this.m_cb.addLine("{");
        final String s2 = this.m_cb.indent(1);
        this.m_cb.addLine("varint_u<" + totalBits + "> raw_bits = " + s + ".getBitString();");
        this.m_cb.addLine("for (int i=0; i<" + totalBits / 2 + "; i++) {");
        final String s3 = this.m_cb.indent(1);
        this.m_cb.addLine("int partner_bit = (" + totalBits + "-1) - i;");
        this.m_cb.addLine("bool low_val = raw_bits[i];");
        this.m_cb.addLine("bool high_val = raw_bits[partner_bit];");
        this.m_cb.addLine("raw_bits[i] = high_val;");
        this.m_cb.addLine("raw_bits[partner_bit] = low_val;");
        final String s4 = this.m_cb.indent(-1);
        this.m_cb.addLine("}");
        this.m_cb.addLine(String.valueOf(statementBitReverse.getVar().getName()) + " = " + this.typeToString(statementBitReverse.getVar().getType()) + "(raw_bits);");
        final String s5 = this.m_cb.indent(-1);
        this.m_cb.addLine("}");
        return this.m_cb.pop();
    }
    
    private String boost_format(final String s, final Expression... array) {
        if (this.m_formatter.useExplicitTemplateInstantiation()) {
            final String string = "format_string_" + this.m_code_formatter.getDesignData().getName() + "_" + this.m_formatter.getUniqueNumber();
            final StringBuilder sb = new StringBuilder();
            final StringBuilder sb2 = new StringBuilder();
            final StringBuilder sb3 = new StringBuilder();
            int n = 0;
            final StringBuilder sb4 = sb.append("const char* _format_arg_format_string");
            for (final Expression expression : array) {
                final String s2 = this.typeToString(expression.getType());
                final StringBuilder sb5 = sb.append(", const ");
                final StringBuilder sb6 = sb.append(s2);
                final StringBuilder sb7 = sb.append(" &_format_arg_" + n);
                this.m_cb.addLine(String.valueOf(new StringBuilder("const ").append(s2).append(" &_format_arg_").append(n).toString()) + (" = " + expression.format(this.m_expf) + ";"));
                final StringBuilder sb8 = sb3.append(", _format_arg_" + n);
                ++n;
            }
            final StringTokenizer stringTokenizer = new StringTokenizer(s, "%", true);
            final ArrayList<String> list = new ArrayList<String>();
            while (stringTokenizer.hasMoreTokens()) {
                if (stringTokenizer.nextToken().contains("%")) {
                    list.add(stringTokenizer.nextToken());
                }
            }
            final StringBuilder sb9 = new StringBuilder();
            final StringBuilder sb10 = sb9.append("return ( bfmt(_format_arg_format_string)");
            for (int j = 0; j < array.length; ++j) {
                final StringBuilder sb11 = sb9.append("% _format_arg_" + j + " ");
                if (list.get(j).startsWith("c") || list.get(j).startsWith("s")) {
                    final StringBuilder sb12 = sb9.append(".clearFormatting()");
                }
                if (list.get(j).startsWith("c")) {
                    final StringBuilder sb13 = sb9.append(".printAsChar()");
                }
                if (list.get(j).startsWith("s")) {
                    final StringBuilder sb14 = sb9.append(".printAsString()");
                }
            }
            final StringBuilder sb15 = sb9.append(").str();");
            this.m_formatter.addTemplatedCode(new FormatterBitAccurate.TemplatedMethod("std::string", string, sb.toString(), sb9.toString()));
            this.m_cb.addRaw(sb2.toString());
            return String.valueOf(string) + "(\"" + s + "\"" + sb3.toString() + ")";
        }
        final StringBuilder sb16 = new StringBuilder("(bfmt(\"" + s + "\")");
        final StringTokenizer stringTokenizer2 = new StringTokenizer(s, "%", true);
        final ArrayList<String> list2 = new ArrayList<>();
        while (stringTokenizer2.hasMoreTokens()) {
            if (stringTokenizer2.nextToken().contains("%")) {
                list2.add(stringTokenizer2.nextToken());
            }
        }
        for (int k = 0; k < array.length; ++k) {
            final StringBuilder sb17 = sb16.append(" % ");
            final StringBuilder sb18 = sb16.append(array[k].format(this.m_expf));
            if (((String)list2.get(k)).startsWith("c") || ((String)list2.get(k)).startsWith("s")) {
                final StringBuilder sb19 = sb16.append(".clearFormatting()");
            }
            if (((String)list2.get(k)).startsWith("c")) {
                final StringBuilder sb20 = sb16.append(".printAsChar()");
            }
            if (((String)list2.get(k)).startsWith("s")) {
                final StringBuilder sb21 = sb16.append(".printAsString()");
            }
        }
        final StringBuilder sb22 = sb16.append(").str()");
        return sb16.toString();
    }
    
    @Override
    public String format(final StatementException ex) {
        this.m_cb.push();
        this.m_cb.addLine("throw std::runtime_error((" + this.boost_format("Run-time exception during simulation: " + ex.getFormat(), (Expression[])ex.getArguments()) + "));");
        return this.m_cb.pop();
    }
    
    @Override
    public String format(final StatementTryCatch statementTryCatch) {
        this.m_cb.push();
        if (this.m_code_formatter.getDesignData().getKernelConfiguration().isExceptionHandlingActive()) {
            this.m_cb.addLine("try {");
            final String s = this.m_cb.indent(1);
            this.m_cb.addCodeBlock(statementTryCatch.getTry());
            final String s2 = this.m_cb.indent(-1);
            this.m_cb.addLine("}");
            this.m_cb.addLine("catch (" + statementTryCatch.getException() + ") {");
            final String s3 = this.m_cb.indent(1);
            this.m_cb.addCodeBlock(statementTryCatch.getCatch());
            final String s4 = this.m_cb.indent(-1);
            this.m_cb.addLine("}");
        }
        else {
            this.m_cb.addCodeBlock(statementTryCatch.getTry());
        }
        return this.m_cb.pop();
    }
    
    @Override
    public String format(final StatementPrintf statementPrintf) {
        this.m_cb.push();
        this.m_cb.addLine("simPrintf(\"\", " + statementPrintf.getSequenceNumber() + ", " + this.boost_format(statementPrintf.getFormat(), (Expression[])statementPrintf.getArguments()) + ");");
        return this.m_cb.pop();
    }
    
    @Override
    public String format(final StatementWarning statementWarning) {
        this.m_cb.push();
        this.m_cb.addLine("simPrintf(\"\", " + statementWarning.getNodeId() + ", " + this.boost_format(statementWarning.getFormat(), (Expression[])statementWarning.getArguments()) + ");");
        return this.m_cb.pop();
    }
    
    @Override
    public String format(final StatementDivMod statementDivMod) {
        this.m_cb.push();
        final CVarHWExcept cVarHWExcept = statementDivMod.getExceptionVar();
        final String s = (cVarHWExcept == null) ? "" : (", (&" + cVarHWExcept.format(this.m_expf) + ")");
        final String s2 = (cVarHWExcept == null) ? "" : (", " + this.typeToString(cVarHWExcept.getType()) + "* exception ");
        this.m_cb.addLine(String.valueOf(statementDivMod.getQuotient().getName()) + " = divmod_fixed<" + this.m_formatter.formatter_utils.hwTypeToParms(statementDivMod.getQuotient().getType()) + ">(" + statementDivMod.getNumerator().format(this.m_expf) + "," + statementDivMod.getDenominator().format(this.m_expf) + ", &" + statementDivMod.getRemainder().getName() + s + ");");
        this.m_formatter.registerTemplateInstantiation("template " + this.typeToString(statementDivMod.getQuotient().getType()) + " divmod_fixed<>( const " + this.typeToString(statementDivMod.getNumerator().getType()) + "&, const " + this.typeToString(statementDivMod.getDenominator().getType()) + "&, " + this.typeToString(statementDivMod.getRemainder().getType()) + "*" + s2 + " );");
        return this.m_cb.pop();
    }
    
    @Override
    public String format(final StatementCall statementCall) {
        this.m_cb.push();
        final StringBuilder sb = new StringBuilder(statementCall.getNameToCall());
        if (statementCall.getTemplateArguments() != null) {
            final StringBuilder sb2 = sb.append("< ");
            int n = 0;
            for (final SimCodeType simCodeType : statementCall.getTemplateArguments()) {
                if (n > 0) {
                    final StringBuilder sb3 = sb.append(", ");
                }
                else {
                    ++n;
                }
                final StringBuilder sb4 = sb.append(this.typeToString(simCodeType));
            }
            final StringBuilder sb5 = sb.append(" > ");
        }
        final StringBuilder sb6 = sb.append("(");
        int n2 = 0;
        if (statementCall.getArguments() != null) {
            for (final String s : statementCall.getArguments()) {
                if (n2 > 0) {
                    final StringBuilder sb7 = sb.append(", ");
                }
                else {
                    ++n2;
                }
                final StringBuilder sb8 = sb.append(s);
            }
        }
        final StringBuilder sb9 = sb.append("); ");
        this.m_cb.addLine(sb.toString());
        return this.m_cb.pop();
    }
}
