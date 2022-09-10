package com.maxeler.platform.max5.toolchain.build;

import com.maxeler.platform.max5.toolchain.file.BuildFileVivadoLog;
import com.maxeler.conf.base.BuildConfOption;
import com.maxeler.platform.max5.conf.Max5MaxCompilerBuildConf;
import com.maxeler.maxdc.proc_management.ProcResult;
import com.maxeler.maxdc.proc_management.ProcSpec;
import java.io.IOException;
import com.maxeler.maxdc.MaxDCException;
import java.util.regex.Matcher;
import java.io.BufferedReader;
import com.maxeler.maxdc.BuildFile;
import com.maxeler.platform.max5.toolchain.VivadoPlatform;
import com.maxeler.platform.max5.toolchain.parts.VivadoFpgaPart;
import com.maxeler.maxdc.Entity;
import com.maxeler.maxdc.BuildPassNonFatalException;
import com.maxeler.platform.max5.conf.Max5TcConfOptions;
import com.maxeler.maxdc.BuildManager;
import java.util.regex.Pattern;
import com.maxeler.utils.BuildManagerSupplier;
import com.maxeler.maxdc.BuildPass;

public abstract class VivadoBuildPass implements BuildPass, BuildManagerSupplier
{
    public static final String WORKING_SUBDIR = "xilinx_vivado";
    public static final int DEFAULT_RAM_USAGE = 8192;
    public static final String DEFAULT_CLUSTER_TAGS = "linux";
    private static final Pattern A;
    protected final BuildManager m_buildManager;
    private final Max5TcConfOptions B;
    
    static {
        A = Pattern.compile("peak = (\\d+)");
    }
    
    protected VivadoBuildPass(final BuildManager buildManager, final Max5TcConfOptions b) {
        this.m_buildManager = buildManager;
        this.B = b;
    }
    
    public void pass(final BuildManager buildManager) throws BuildPassNonFatalException {
        try {
            this.m_buildManager.pushWorkingDir("xilinx_vivado");
            this.vivadoPass();
        }
        finally {
            this.m_buildManager.popWorkingDir();
        }
    }
    
    protected String getSig() {
        return this.m_buildManager.getRootEntity().getSignature();
    }
    
    protected VivadoFpgaPart getPart() {
        return (VivadoFpgaPart)VivadoPlatform.get(this.m_buildManager).getFPGAPart();
    }
    
    public BuildManager getBuildManager() {
        return this.m_buildManager;
    }
    
    protected int getPeakVirtualMemoryUsage(final BuildFile buildFile) {
        try {
            final BufferedReader bufferedReader = buildFile.getBufferedReader();
            try {
                boolean b = false;
                int n = 0;
                String s;
                while ((s = bufferedReader.readLine()) != null) {
                    final Matcher matcher = VivadoBuildPass.A.matcher(s);
                    if (matcher.find()) {
                        final int int1 = Integer.parseInt(matcher.group(1));
                        if (int1 > n) {
                            n = int1;
                        }
                        b = true;
                    }
                }
                if (b) {
                    return n;
                }
                throw new MaxDCException("Failed to find memory usage information in report \"" + buildFile + "\"");
            }
            finally {
                bufferedReader.close();
            }
        }
        catch (final IOException ex) {
            throw new MaxDCException("Error parsing report \"" + buildFile + "\"");
        }
    }
    
    protected void logPeakMemoryUsageNoException(final String s, final BuildFile buildFile, final BuildManager buildManager) {
        if (buildFile.exists()) {
            try {
                buildManager.logInfo(String.valueOf(s) + " peak virtual memory: " + this.getPeakVirtualMemoryUsage(buildFile));
            }
            catch (final Exception ex) {
                buildManager.logInfo("Error getting peak memory usage: " + ex.getMessage());
            }
        }
        else {
            buildManager.logInfo(String.valueOf(s) + " log file was not created.");
        }
    }
    
    protected ProcSpec makeVivadoProcSpec(final String s) {
        return makeVivadoProcSpec(this.m_buildManager, s, (int)(this.getRamUsage() * 1024 * 1.1));
    }
    
    protected ProcResult runVivadoProcOnCluster(final ProcSpec procSpec, final String s) {
        procSpec.setRunnableOnCluster(s, (double)this.getCpuUsage(), this.getRamUsage(), (String)this.getClusterTags());
        return procSpec.runSync();
    }
    
    public static String getVivadoCommandGui(final BuildManager buildManager) {
        final StringBuilder sb = new StringBuilder();
        final StringBuilder sb2 = sb.append((String)buildManager.getParameter(Max5MaxCompilerBuildConf.max5.vivado.command_prefix));
        final StringBuilder sb3 = sb.append(" ");
        final StringBuilder sb4 = sb.append((String)buildManager.getParameter(Max5MaxCompilerBuildConf.max5.vivado.command));
        final StringBuilder sb5 = sb.append(" ");
        final StringBuilder sb6 = sb.append((String)buildManager.getParameter(Max5MaxCompilerBuildConf.max5.vivado.options));
        final StringBuilder sb7 = sb.append(" ");
        return sb.toString();
    }
    
    public static String getVivadoCommandBatch(final BuildManager buildManager) {
        final StringBuilder sb = new StringBuilder();
        final StringBuilder sb2 = sb.append(getVivadoCommandGui(buildManager));
        final StringBuilder sb3 = sb.append("-mode batch -source ");
        return sb.toString();
    }
    
    public static ProcSpec makeVivadoProcSpec(final BuildManager buildManager, final String s, final int n) {
        return new ProcSpec(buildManager, String.valueOf((n > 0) ? new StringBuilder("ulimit -v ").append(n).append(";").toString() : "") + s);
    }
    
    protected abstract void vivadoPass() throws BuildPassNonFatalException;
    
    private static BuildConfOption.BuildConfStringOption C(final Max5TcConfOptions max5TcConfOptions) {
        return max5TcConfOptions.cluster_tags;
    }
    
    private static BuildConfOption.BuildConfIntOption A(final Max5TcConfOptions max5TcConfOptions) {
        return max5TcConfOptions.ram_usage;
    }
    
    private static BuildConfOption.BuildConfIntOption B(final Max5TcConfOptions max5TcConfOptions) {
        return max5TcConfOptions.cpu_usage;
    }
    
    protected final String getClusterTags() {
        return getClusterTags(this.m_buildManager, this.B);
    }
    
    public static final String getClusterTags(final BuildManager buildManager, final Max5TcConfOptions max5TcConfOptions) {
        return (String)buildManager.getParameter(C(max5TcConfOptions));
    }
    
    protected int getRamUsage() {
        return getRamUsage(this.m_buildManager, this.B);
    }
    
    public static int getRamUsage(final BuildManager buildManager, final Max5TcConfOptions max5TcConfOptions) {
        return (int)buildManager.getParameter(A(max5TcConfOptions));
    }
    
    public static void setRamUsage(final BuildManager buildManager, final Max5TcConfOptions max5TcConfOptions, final int n) {
        buildManager.setParameter(A(max5TcConfOptions), n);
    }
    
    protected int getCpuUsage() {
        final int maxThreads = VivadoVersion.get(this.m_buildManager).getMaxThreads();
        final BuildConfOption.BuildConfIntOption buildConfIntOption = B(this.B);
        int intValue = (int)this.m_buildManager.getParameter(buildConfIntOption);
        if (intValue > maxThreads) {
            this.m_buildManager.logWarning(buildConfIntOption + " (" + intValue + ") cannot be greater than " + maxThreads + ". Limiting to " + maxThreads + ".");
            intValue = maxThreads;
        // BEGIN CODE ADDITION
        } else if (intValue < maxThreads) {
            this.m_buildManager.logInfo(buildConfIntOption + " (" + intValue + ") being adjusted to maximum of " + maxThreads + ".");
            intValue = maxThreads;
        // END CODE ADDITION
        }
        return intValue;
    }
    
    protected String[] getVivadoWhiteList() {
        return new String[0];
    }
    
    protected BuildFileVivadoLog createLogFile(final String s) {
        final BuildFileVivadoLog buildFileVivadoLog = new BuildFileVivadoLog(this.getVivadoWhiteList(), this.m_buildManager, s, false);
        buildFileVivadoLog.delete();
        return buildFileVivadoLog;
    }
}
