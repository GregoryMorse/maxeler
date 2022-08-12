package com.maxeler.maxdc;

import java.util.Optional;
import java.util.Collections;
import com.maxeler.timinganalyser.MaxTimingAnalyserUtils;
import com.maxeler.maxdc.resource_usage.summaries.GenericResourceSummary;
import com.maxeler.maxdc.resource_usage.BuildResourceUsage;
import com.maxeler.maxdc.maxinfo.MaxInfoCreator;
import com.maxeler.utils.MailFactory;
import com.maxeler.utils.AmountOfTime;
import com.maxeler.maxdc.proc_management.ProcResult;
import java.util.Arrays;
import com.maxeler.utils.io.MaxFile;
import java.util.Collection;
import java.util.TreeSet;
import java.util.regex.Pattern;
import com.maxeler.maxdc.proc_management.ArbitratedCoreCache;
import com.maxeler.corecache.common.CoreCacheException;
import com.maxeler.maxdc.proc_management.MaxCoreCache;
import com.maxeler.utils.UncaughtMaxelerExceptionLogger;
import java.io.BufferedWriter;
import java.nio.channels.WritableByteChannel;
import java.io.Writer;
import java.util.LinkedList;
import java.io.BufferedReader;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.Channels;
import java.io.Reader;
import java.nio.channels.FileLock;
import java.nio.file.StandardOpenOption;
import java.nio.file.OpenOption;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.Path;
import com.maxeler.utils.maxide.BuildReport;
import java.net.UnknownHostException;
import java.net.InetAddress;
import com.maxeler.utils.MakeClone;
import com.maxeler.maxcompiler.v2._MaxCompiler;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import com.maxeler.maxcompiler.Version;
import com.maxeler.licensing.LicenseManager;
import com.maxeler.utils.StopWatch;
import com.maxeler.utils.Env;
import java.io.FileFilter;
import com.maxeler.conf.base.BuildConfOption;
import com.maxeler.conf.base.MaxCompilerBuildConf;
import java.io.InputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Iterator;
import java.util.HashMap;
import java.util.TreeMap;
import java.text.DateFormat;
import com.maxeler.maxdc.resource_usage.BuildFileMXRU;
import com.maxeler.maxdc.maxinfo.MaxInfo;
import com.maxeler.utils.CreationLogger;
import java.lang.ref.WeakReference;
import java.util.Date;
import com.maxeler.maxdc.proc_management.ProcAsync;
import com.maxeler.maxdc.proc_management.CoreCache;
import com.maxeler.maxdc.proc_management.BuildProcessDB;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.Stack;
import java.io.File;
import com.maxeler.utils.Hasher;

public final class BuildManager
{
    public static final String s_scratchDir;
    private static final String s_resultsDir;
    private PrevBuildData m_prev_build_data;
    private final BuildLogger m_log;
    private Entity m_root_entity;
    private final EntityBuilder m_entity_builder;
    private File m_root_dir;
    private final Stack<File> m_dir_stack;
    private Map<File, BuildFile> m_source_files;
    private final Set<BuildFile> m_build_files;
    private final Set<BuildFile> m_result_files;
    private final List<BuildPass> m_build_passes;
    private final List<BuildManagerCallBack> m_final_callbacks;
    private boolean m_build_finished;
    private BuildProcessDB m_processdb_local;
    private CoreCache m_core_cache;
    private Map<String, Object> m_global_variables;
    private Platform<?, ?> m_platform;
    private boolean m_target_simulation;
    private final BuildParameters m_props;
    private String m_build_name;
    private String m_unique_build_name;
    private final String m_slic_stem_name;
    private final List<ProcAsync> m_build_jobs;
    private final List<ProcAsync> m_copied_submanager_jobs;
    private final BuildManager m_parent_bm;
    private final List<BuildManager> m_sub_bms;
    private File[] m_sub_bm_root_path;
    private final Date m_start_time;
    private File m_lock_file;
    private static final Map<String, BuildManager> m_build_directory_owners_map;
    private final List<WeakReference<CreationLogger>> m_creation_loggers;
    private final List<BuildPassNonFatalException> m_non_fatal_errors;
    private boolean backupAlreadyDone;
    private Integer m_bestTimingScore;
    private MaxInfo m_maxInfoData;
    private final IpCoreInfo m_ipCoreInfo;
    private CacheManager m_buildCacheManager;
    private static final String s_maxdcBuildPrefix = ".maxdc_builds_";
    private BuildFileMXRU m_most_recent_mxru;
    private boolean m_firstCall_createPrelimCompilerDesignFile;
    private final DateFormat m_logDateFormat;
    
    static {
        s_scratchDir = String.valueOf(File.separatorChar) + "scratch";
        s_resultsDir = String.valueOf(File.separatorChar) + "results";
        m_build_directory_owners_map = new TreeMap<String, BuildManager>();
    }
    
    public BuildManager getSubBuildManager(final String s) {
        this.assertHasBuildDirLock();
        final BuildManager buildManager = new BuildManager(this);
        this.m_sub_bms.add(buildManager);
        final File[] array = this.getWorkingDirStack();
        System.arraycopy(array, 0, buildManager.m_sub_bm_root_path = new File[array.length + 1], 0, array.length);
        buildManager.m_sub_bm_root_path[array.length] = new File(s);
        buildManager.m_root_entity = this.m_root_entity;
        this.mkdir(buildManager.m_root_dir = new File(new File(this.m_root_dir.getAbsoluteFile(), this.getWorkingDirStr()), s));
        buildManager.m_log.createLogFile(buildManager, false);
        final File[] array2 = new File[this.getWorkingDirStack().length + 1];
        for (int i = 0; i < this.getWorkingDirStack().length; ++i) {
            array2[i] = this.getWorkingDirStack()[i];
        }
        array2[this.getWorkingDirStack().length] = new File(s);
        final Iterator<BuildFile> iterator = this.m_build_files.iterator();
        while (iterator.hasNext()) {
            buildManager.addBuildFile(iterator.next().dupForSubBuildManager(buildManager, array2));
        }
        buildManager.m_processdb_local = new BuildProcessDB(buildManager);
        if (this.m_core_cache != null) {
            buildManager.m_core_cache = this.newCoreCacheClient();
        }
        buildManager.m_global_variables = new HashMap<String, Object>(this.m_global_variables);
        buildManager.m_platform = this.m_platform;
        buildManager.m_target_simulation = this.m_target_simulation;
        buildManager.m_build_name = String.valueOf(s) + "_" + this.m_build_name;
        buildManager.m_unique_build_name = this.m_unique_build_name;
        buildManager.m_build_finished = this.m_build_finished;
        buildManager.m_maxInfoData = this.m_maxInfoData;
        buildManager.m_buildCacheManager = this.m_buildCacheManager;
        this.logInfo("Created new sub build manager running in: " + buildManager.m_root_dir);
        this.logInfo("For detailed output from this sub build manager see: " + buildManager.m_root_dir + "/_build.log");
        buildManager.logInfo("Sub build manager running in: " + buildManager.m_root_dir);
        return buildManager;
    }
    
    public void transferBuildFilesToParent() {
        if (this.m_parent_bm == null) {
            throw new MaxDCException(this, "This is not a child BuildManager.");
        }
        for (final BuildFile buildFile : this.m_build_files) {
            if (buildFile.getCopyToParent()) {
                this.m_parent_bm.addBuildFile(buildFile.dupForParentBuildManager(this.m_parent_bm, this.m_sub_bm_root_path));
            }
        }
        for (final BuildFile buildFile2 : this.m_result_files) {
            if (buildFile2.getCopyToParent()) {
                this.m_parent_bm.addBuildFileResult(buildFile2.dupForParentBuildManager(this.m_parent_bm, this.m_sub_bm_root_path));
            }
        }
    }
    
    public void transferBuildFilesFromParent() {
        if (this.m_parent_bm == null) {
            throw new MaxDCException(this, "This is not a child BuildManager.");
        }
        final Iterator<BuildFile> iterator = this.m_parent_bm.m_build_files.iterator();
        while (iterator.hasNext()) {
            this.addBuildFile(iterator.next().dupForSubBuildManager(this, this.m_sub_bm_root_path));
        }
    }
    
    public static ChooseBuildRootRes chooseBuildRoot(String build_name, final BuildParameters buildParameters, final Date date) {
        String s = buildParameters.getProperty("build.root_dir");
        if (s == null || s.isEmpty()) {
            s = new File(".").getAbsolutePath();
        }
        final String s2 = System.getProperty("file.separator");
        if (!s.endsWith(s2)) {
            s = String.valueOf(s) + s2;
        }
        String string;
        if (build_name.startsWith("/")) {
            string = build_name;
            build_name = new File(build_name).getName();
        }
        else {
            if (build_name.contains("/")) {
                s = String.valueOf(s) + build_name.substring(0, build_name.lastIndexOf("/")) + s2;
                build_name = new File(build_name).getName();
            }
            if (buildParameters.getPropertyBool("build.datestamp_builds", false)) {
                s = String.valueOf(new StringBuilder(String.valueOf(s)).append(new SimpleDateFormat(buildParameters.getProperty("build.datestamp_builds_format", "dd-MM-yy")).format(date)).toString()) + s2;
            }
            string = String.valueOf(new StringBuilder(String.valueOf(s)).append(build_name).toString()) + s2;
        }
        final ChooseBuildRootRes chooseBuildRootRes = new ChooseBuildRootRes();
        chooseBuildRootRes.build_dir = string;
        chooseBuildRootRes.build_name = build_name;
        return chooseBuildRootRes;
    }
    
    private void dumpParameters(final String s) {
        final BuildFileOutput buildFileOutput = new BuildFileOutput(this, s, false);
        final PrintStream printStream = buildFileOutput.getPrintStream();
        try {
            this.m_props.store(printStream, "");
        }
        catch (final IOException ex) {
            throw new MaxDCException(this, "Error saving parameters to: " + buildFileOutput.getAbsoluteName() + "(" + ex + ")");
        }
    }
    
    private static void copyAndCloseInputStream(final InputStream inputStream, final OutputStream outputStream) throws IOException {
        final byte[] array = new byte[4096];
        int read;
        while (-1 != (read = inputStream.read(array))) {
            outputStream.write(array, 0, read);
        }
        inputStream.close();
    }
    
    private static void runAndLogCommand(final PrintStream printStream, final String s, final File file) {
        try {
            printStream.println("* Executing '" + s + "' in " + file.getCanonicalPath());
            final Process process = Runtime.getRuntime().exec(s, null, file);
            copyAndCloseInputStream(process.getInputStream(), printStream);
            copyAndCloseInputStream(process.getErrorStream(), printStream);
            printStream.println("Exit code: " + process.waitFor() + "\n");
        }
        catch (final Exception ex) {
            ex.printStackTrace(printStream);
        }
    }
    
    private void importSourceDir(final File file, final File file2) {
        if (file2 == null && this.getParameter((BuildConfOption<Boolean>)MaxCompilerBuildConf.build.enable_source_svn_info) && new File(file, ".svn").isDirectory()) {
            final PrintStream printStream = new BuildLogFile(this, "svn-revision-info.log", false).getPrintStreamAppend();
            runAndLogCommand(printStream, "svn st", file);
            runAndLogCommand(printStream, "svn info -R", file);
            printStream.close();
        }
        File[] array;
        for (int length = (array = file.listFiles(new FileFilter() {
            @Override
            public boolean accept(final File file) {
                return (file.isDirectory() && !file.getName().equals(".svn") && !file.getName().equals(".git")) || file.getName().endsWith(".java") || file.getName().endsWith(".maxj");
            }
        })).length, i = 0; i < length; ++i) {
            final File file3 = array[i];
            final File file4 = new File(file2, file3.getName());
            if (file3.isFile()) {
                if (new File(this.getWorkingDirStrAbsolute(), file3.getName()).exists()) {
                    throw new MaxDCException(this, "Found duplicate source file '" + file3 + "'.\n" + "This probably means overlapping source directories " + "are included in MAXSOURCEDIRS.\n");
                }
                final BuildFileBuildSource buildFileBuildSource = new BuildFileBuildSource(this, file3.getAbsolutePath(), true);
                final BuildFile buildFile = this.m_source_files.put(file4, buildFileBuildSource);
                final String s = this.m_prev_build_data.m_relative_source_file_paths.put(file4, buildFileBuildSource.getBuildRootRelativeName());
            }
            else if (file3.isDirectory()) {
                this.pushWorkingDir(file3.getName());
                this.importSourceDir(file3, file4);
                this.popWorkingDir();
            }
        }
    }
    
    private void importSourceFilesFromPreviousBuild() {
        this.m_source_files = new HashMap<File, BuildFile>();
        final String workingDir = this.getWorkingDir();
        this.setWorkingDir("/");
        for (final Map.Entry entry : this.m_prev_build_data.m_relative_source_file_paths.entrySet()) {
            final BuildFile buildFile = this.m_source_files.put((File)entry.getKey(), new BuildFileBuildSource(this, (String)entry.getValue(), false));
        }
        this.setWorkingDir(workingDir);
    }
    
    private void importSourceFiles() {
        if (this.getParameter((BuildConfOption<Boolean>)MaxCompilerBuildConf.build.enable_source_backup) && Env.get("MAXSOURCEDIRS") != null) {
            this.m_source_files = new HashMap<File, BuildFile>();
            this.logInfo("Backing-up source-files (old source files in build directory will be removed first)...");
            this.logInfo("Source directories: " + Env.get("MAXSOURCEDIRS"));
            final StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            this.setWorkingDir("/");
            this.removeDirectory("src");
            this.setWorkingDir("/src");
            String[] array;
            for (int length = (array = Env.get("MAXSOURCEDIRS").trim().split(":")).length, i = 0; i < length; ++i) {
                final String s = array[i];
                final File file = new File(s);
                if (!file.exists()) {
                    throw new MaxDCException(this, "Path in MAXSOURCEDIRS does not exist: " + s);
                }
                if (file.isFile()) {
                    final BuildFileBuildSource buildFileBuildSource = new BuildFileBuildSource(this, s, true);
                    final File file2 = new File(file.getName());
                    final BuildFile buildFile = this.m_source_files.put(file2, buildFileBuildSource);
                    final String s2 = this.m_prev_build_data.m_relative_source_file_paths.put(file2, buildFileBuildSource.getBuildRootRelativeName());
                }
                else if (file.isDirectory()) {
                    try {
                        if (containsDirectory(file, this.m_root_dir)) {
                            throw new MaxDCException(this, "Source directories cannot include build directory");
                        }
                    }
                    catch (final IOException ex) {
                        throw new MaxDCException(this, "Error getting absolute path name", ex);
                    }
                    this.importSourceDir(file, null);
                }
            }
            this.setWorkingDir(BuildManager.s_scratchDir);
            stopWatch.stop();
            this.logInfo("Copying source-files took %s", stopWatch);
        }
        else {
            this.logWarning("Source for this build will not be backed-up and source-code annotations will not be");
            this.logWarning("made. This can be caused either by the environment variable MAXSOURCEDIRS not being");
            this.logWarning("set or the MaxCompiler configuration option 'build.enable_source_backup' being");
            this.logWarning("unset/set to false.");
        }
    }
    
    private static boolean containsDirectory(final File file, final File file2) throws IOException {
        final File file3 = file.getCanonicalFile();
        for (File file4 = file2.getCanonicalFile(); file4 != null; file4 = file4.getParentFile()) {
            if (file3.equals(file4)) {
                return true;
            }
        }
        return false;
    }
    
    public BuildFile findSourceFile(final File file) {
        if (this.m_source_files == null) {
            return null;
        }
        return this.m_source_files.get(file);
    }
    
    boolean hasBuildDirLock() {
        return this.m_lock_file != null || this.m_parent_bm != null;
    }
    
    public void assertHasBuildDirLock() {
        if (!this.hasBuildDirLock()) {
            throw new LostBuildLock();
        }
    }
    
    private void checkLicenseValid() {
        LicenseManager.get(this).enforceLicenseFor(Version.isSimOnlyRun() ? "MaxCompiler_core_sim" : "MaxCompiler_core");
    }
    
    private BuildManager(final BuildManager parent_bm) {
        this.m_prev_build_data = null;
        this.m_log = new BuildLogger();
        this.m_root_entity = null;
        this.m_dir_stack = new Stack<File>();
        final File file = this.m_dir_stack.push(new File("."));
        this.m_build_files = new LinkedHashSet<BuildFile>();
        this.m_result_files = new LinkedHashSet<BuildFile>();
        this.m_build_passes = new ArrayList<BuildPass>();
        this.m_final_callbacks = new ArrayList<BuildManagerCallBack>();
        this.m_build_finished = false;
        this.m_core_cache = null;
        this.m_global_variables = new HashMap<String, Object>();
        this.m_platform = null;
        this.m_build_jobs = new ArrayList<ProcAsync>();
        this.m_copied_submanager_jobs = new ArrayList<ProcAsync>();
        this.m_sub_bms = new ArrayList<BuildManager>();
        this.m_sub_bm_root_path = null;
        this.m_lock_file = null;
        this.m_creation_loggers = new ArrayList<WeakReference<CreationLogger>>();
        this.m_non_fatal_errors = new ArrayList<BuildPassNonFatalException>();
        this.backupAlreadyDone = false;
        this.m_bestTimingScore = null;
        this.m_maxInfoData = null;
        this.m_buildCacheManager = null;
        this.m_firstCall_createPrelimCompilerDesignFile = true;
        this.m_logDateFormat = new SimpleDateFormat("HH:mm:ss dd/MM/yy");
        this.m_start_time = new Date();
        this.m_parent_bm = parent_bm;
        this.m_lock_file = null;
        this.m_props = parent_bm.m_props;
        this.m_slic_stem_name = parent_bm.m_slic_stem_name;
        this.m_ipCoreInfo = parent_bm.getIpCoreInfo();
        this.m_entity_builder = new EntityBuilder(this);
    }
    
    public BuildManager(final String s, final boolean b) {
        this(s, s, b);
    }
    
    public BuildManager(final String slic_stem_name, final String s, final boolean target_simulation) {
        this.m_prev_build_data = null;
        this.m_log = new BuildLogger();
        this.m_root_entity = null;
        this.m_dir_stack = new Stack<File>();
        final File file = this.m_dir_stack.push(new File("."));
        this.m_build_files = new LinkedHashSet<BuildFile>();
        this.m_result_files = new LinkedHashSet<BuildFile>();
        this.m_build_passes = new ArrayList<BuildPass>();
        this.m_final_callbacks = new ArrayList<BuildManagerCallBack>();
        this.m_build_finished = false;
        this.m_core_cache = null;
        this.m_global_variables = new HashMap<String, Object>();
        this.m_platform = null;
        this.m_build_jobs = new ArrayList<ProcAsync>();
        this.m_copied_submanager_jobs = new ArrayList<ProcAsync>();
        this.m_sub_bms = new ArrayList<BuildManager>();
        this.m_sub_bm_root_path = null;
        this.m_lock_file = null;
        this.m_creation_loggers = new ArrayList<WeakReference<CreationLogger>>();
        this.m_non_fatal_errors = new ArrayList<BuildPassNonFatalException>();
        this.backupAlreadyDone = false;
        this.m_bestTimingScore = null;
        this.m_maxInfoData = null;
        this.m_buildCacheManager = null;
        this.m_firstCall_createPrelimCompilerDesignFile = true;
        this.m_logDateFormat = new SimpleDateFormat("HH:mm:ss dd/MM/yy");
        if (_MaxCompiler.isUsingMaxCompilerRunner()) {
            this.m_prev_build_data = _MaxCompiler.getAndClearPreviousRunData();
        }
        if (this.m_prev_build_data == null) {
            this.m_start_time = new Date();
        }
        else {
            this.m_start_time = this.m_prev_build_data.m_start_date;
        }
        this.m_parent_bm = null;
        final String s2 = Version.getVersionString();
        if (this.m_prev_build_data == null) {
            this.logProgress("MaxCompiler version: " + s2);
            this.logProgress("Build \"" + slic_stem_name + "\" start time: " + this.m_start_time);
            this.logProgress("Main build process running as user " + this.getUserName() + " on host " + this.getHostname());
        }
        if (this.m_prev_build_data == null) {
            this.m_props = BuildParameters.loadDefaultParameters(this);
        }
        else {
            this.m_props = MakeClone.now(this.m_prev_build_data.m_initial_properties);
        }
        this.m_ipCoreInfo = new IpCoreInfo();
        CreationLogger.setupCreationLoggerParams(this);
        if (Env.get("MAXCOMPILERDIR") == null) {
            throw new MaxDCException("Environment variable MAXCOMPILERDIR is not set.");
        }
        try {
            this.m_unique_build_name = String.valueOf(this.m_unique_build_name) + " " + InetAddress.getLocalHost().getHostName();
        }
        catch (final UnknownHostException ex) {}
        final ChooseBuildRootRes chooseBuildRootRes = chooseBuildRoot(s, this.m_props, this.m_start_time);
        this.m_slic_stem_name = slic_stem_name;
        if (!VHDLNameManager.isLegalVHDLName(this.m_slic_stem_name)) {
            throw new MaxDCException("Can't use name '" + slic_stem_name + "' as it is not a legal VHDL identifier.");
        }
        if (slic_stem_name.equals("max")) {
            throw new MaxDCException("The stem name 'max' cannot be used");
        }
        this.m_build_name = this.m_slic_stem_name;
        BuildReport.SlicStem(this.m_slic_stem_name);
        if (!VHDLNameManager.isLegalVHDLName(this.m_build_name)) {
            throw new MaxDCException("Can't use name '" + this.m_build_name + "' as it is not a legal VHDL identifier.");
        }
        this.m_unique_build_name = String.valueOf(chooseBuildRootRes.build_name) + " " + this.m_start_time;
        this.mkdir(this.m_root_dir = new File(chooseBuildRootRes.build_dir));
        try {
            final Path path = Paths.get(".maxdc_builds_" + slic_stem_name, new String[0]).toAbsolutePath();
            final FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.SYNC, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.READ);
            this.logInfo("Appending build directory location to '" + path.toString() + "'");
            final FileLock fileLock = fileChannel.lock();
            final BufferedReader bufferedReader = new BufferedReader(Channels.newReader(fileChannel, "UTF-8"));
            final LinkedList<String> list = new LinkedList<>();
            String s3;
            while ((s3 = bufferedReader.readLine()) != null) {
                list.add(s3);
            }
            final Iterator iterator = list.iterator();
            while (iterator.hasNext()) {
                if (((String)iterator.next()).equals(this.m_root_dir.getAbsolutePath())) {
                    iterator.remove();
                    break;
                }
            }
            list.add(this.m_root_dir.getAbsolutePath());
            final FileChannel fileChannel2 = fileChannel.truncate(0L);
            final BufferedWriter bufferedWriter = new BufferedWriter(Channels.newWriter(fileChannel, "UTF-8"));
            final Iterator iterator2 = list.iterator();
            while (iterator2.hasNext()) {
                bufferedWriter.write(String.valueOf(iterator2.next()) + "\n");
            }
            bufferedWriter.flush();
            fileLock.release();
            fileChannel.close();
        }
        catch (final IOException ex2) {
            ex2.printStackTrace();
            throw new MaxDCException("Error writing maxdc_builds file");
        }
        synchronized (BuildManager.m_build_directory_owners_map) {
            final BuildManager buildManager = BuildManager.m_build_directory_owners_map.get(this.m_root_dir.getAbsolutePath());
            if (buildManager != null) {
                this.m_lock_file = buildManager.m_lock_file;
                buildManager.m_lock_file = null;
            }
            else {
                this.m_lock_file = new File(this.m_root_dir, "_lock");
                boolean newFile = false;
                try {
                    newFile = this.m_lock_file.createNewFile();
                }
                catch (final IOException ex3) {}
                if (!newFile) {
                    throw new MaxDCException("\nCannot create lock file in the build directory.\nThis means a build is most likely already running in that directory.\nIf you are sure this is not the case run:\nrm " + this.m_lock_file.getAbsolutePath());
                }
                this.m_lock_file.deleteOnExit();
                Runtime.getRuntime().addShutdownHook(new Thread() {
                    @Override
                    public void run() {
                        if (BuildManager.this.m_lock_file != null) {
                            try {
                                BuildManager.this.m_lock_file.delete();
                            }
                            catch (final SecurityException ex) {}
                        }
                    }
                });
            }
            final BuildManager buildManager2 = BuildManager.m_build_directory_owners_map.put(this.m_root_dir.getAbsolutePath(), this);
        }
        this.m_entity_builder = new EntityBuilder(this);
        this.m_target_simulation = target_simulation;
        this.m_log.createLogFile(this, this.m_prev_build_data != null);
        UncaughtMaxelerExceptionLogger.register();
        BuildReport.BuildLogStart(this.m_log.getLogFile());
        if (this.m_prev_build_data == null) {
            this.logProgress("Build location: " + this.m_root_dir.getAbsolutePath());
            this.logProgress("Detailed build log available in \"_build.log\"");
            final Date date = this.logInfoAtTime("Created build manager " + slic_stem_name + " (" + this.m_unique_build_name + ").");
            this.logInfo("Working in dir: " + this.m_root_dir);
            this.logClassPath();
            this.logMainMethod();
            this.dumpParameters("_init.conf");
            if (!System.getProperty("line.separator").equals("\n")) {
                throw new MaxDCException(this, "Line separator should be set to \\n in the JVM properties.");
            }
        }
        this.m_processdb_local = new BuildProcessDB(this);
        this.m_core_cache = this.newCoreCacheClient();
        this.setWorkingDir(BuildManager.s_scratchDir);
        if (this.m_prev_build_data == null) {
            this.checkLicenseValid();
        }
        if (_MaxCompiler.isUsingMaxCompilerRunner()) {
            _MaxCompiler.setAssociatedBuildManager(this);
        }
        if (this.m_prev_build_data == null) {
            PrevBuildData.access$3(this.m_prev_build_data = new PrevBuildData(), false);
            PrevBuildData.access$4(this.m_prev_build_data, this.m_start_time);
            PrevBuildData.access$5(this.m_prev_build_data, MakeClone.now(this.m_props));
        }
    }
    
    private CoreCache newCoreCacheClient() {
        if (this.hasParameter(MaxCompilerBuildConf.build.core_cache_server)) {
            final String s = this.getParameter((BuildConfOption<String>)MaxCompilerBuildConf.build.core_cache_server);
            final int intValue = this.getParameter((BuildConfOption<Integer>)MaxCompilerBuildConf.build.core_cache_server_tcp_timeout_ms);
            if (s.compareTo("") == 0) {
                this.logInfo("build.core_cache_server is blank, no core cache server will be used.");
                return null;
            }
            String s2;
            int int1;
            try {
                final String[] array = s.split(":");
                s2 = array[0].trim();
                int1 = Integer.parseInt(array[1].trim());
            }
            catch (final NumberFormatException | ArrayIndexOutOfBoundsException ex) {
                this.logWarning("'" + s + "' is not a valid address (" + ex + "), no core cache will be used.");
                return null;
            }
            try {
                return new MaxCoreCache(this, s2, int1, intValue);
            }
            catch (final CoreCacheException ex2) {
                this.logWarning("Failed to instantiate core cache client: " + ex2);
                return null;
            }
        }
        if (!this.hasParameter(MaxCompilerBuildConf.build.arbitrated_core_cache)) {
            return null;
        }
        this.logWarning("build.arbitrated_core_cache is set, the arbitrated core cache is deprecated and will be removed in a future release.");
        final String s3 = this.getParameter((BuildConfOption<String>)MaxCompilerBuildConf.build.arbitrated_core_cache);
        if (s3.compareTo("") == 0) {
            this.logInfo("build.arbitrated_core_cache parameter is blank, arbitrated core cache will not be used.");
            return null;
        }
        this.logInfo("Using " + s3 + " for core-cache.");
        this.mkdir(new File(s3));
        return new ArbitratedCoreCache(this, s3);
    }
    
    public void backupSourceFiles() {
        if (this.backupAlreadyDone) {
            return;
        }
        this.backupAlreadyDone = true;
        if (_MaxCompiler.getAndClearPreviousRunData() == null) {
            this.importSourceFiles();
        }
        else {
            this.importSourceFilesFromPreviousBuild();
        }
    }
    
    public boolean isRestartedBuild() {
        return this.m_prev_build_data.m_is_restarted_build;
    }
    
    public void setMostRecentMXRUFile(final BuildFileMXRU most_recent_mxru) {
        PrevBuildData.access$7(this.m_prev_build_data, most_recent_mxru.getBuildRootRelativeName());
        this.m_most_recent_mxru = most_recent_mxru;
    }
    
    public BuildFileMXRU getMostRecentMXRUFile() {
        if (this.isSubBuildManager()) {
            return null;
        }
        if (this.m_most_recent_mxru == null && this.m_prev_build_data.m_path_to_prev_mxru != null) {
            final String workingDir = this.getWorkingDir();
            this.setWorkingDir("/");
            final BuildFileMXRU most_recent_mxru = new BuildFileMXRU(this, this.m_prev_build_data.m_path_to_prev_mxru, false);
            this.setWorkingDir(workingDir);
            if (most_recent_mxru.exists()) {
                this.m_most_recent_mxru = most_recent_mxru;
            }
        }
        return this.m_most_recent_mxru;
    }
    
    public void rerunCompileNow() {
        if (!_MaxCompiler.isUsingMaxCompilerRunner()) {
            throw new MaxDCException("Cannot restart build as it has not been started using MaxCompiler.run()");
        }
        if (this.m_prev_build_data == null) {
            throw new MaxDCException("Cannot re-run a compile from a sub-BuildManager.");
        }
        this.releaseBuildLock();
        this.logProgress("----------------------------------------------------------------------");
        this.logProgress("Re-running compile process");
        this.logProgress("----------------------------------------------------------------------");
        this.m_log.closeLogFile();
        PrevBuildData.access$3(this.m_prev_build_data, true);
        _MaxCompiler.rerunCompileNow(this, this.m_prev_build_data);
    }
    
    private void releaseBuildLock() {
        if (this.m_lock_file != null) {
            try {
                this.m_lock_file.delete();
            }
            catch (final SecurityException ex) {}
        }
    }
    
    @Override
    protected void finalize() {
        this.releaseBuildLock();
    }
    
    private void logClassPath() {
        final String[] array = System.getProperty("java.class.path").split(Pattern.quote(System.getProperty("path.separator")));
        this.logInfo("Java class path (%d paths):", array.length);
        String[] array2;
        for (int length = (array2 = array).length, i = 0; i < length; ++i) {
            this.logInfo("\t" + array2[i]);
        }
    }
    
    private void logMainMethod() {
        final Set<String> set = getMainClass();
        switch (set.size()) {
            case 0: {
                this.logInfo("Couldn't determine which 'main' method started the Java process.");
                break;
            }
            case 1: {
                this.logInfo("Java process started by 'main' method in class '%s'.", set.iterator().next());
                break;
            }
            default: {
                this.logInfo("Java process started by 'main' method in one of the following classes:");
                final Iterator<String> iterator = set.iterator();
                while (iterator.hasNext()) {
                    this.logInfo("\t%s", iterator.next());
                }
                break;
            }
        }
    }
    
    private static Set<String> getMainClass() {
        final TreeSet<String> set = new TreeSet<>();
        for (final StackTraceElement[] array : Thread.getAllStackTraces().values()) {
            if (array.length == 0) {
                continue;
            }
            final StackTraceElement stackTraceElement = array[array.length - 1];
            if (!stackTraceElement.getMethodName().equals("main")) {
                continue;
            }
            set.add(stackTraceElement.getClassName());
        }
        return set;
    }
    
    public String getBuildLogLocation() {
        return String.valueOf(this.m_root_dir.getAbsolutePath()) + "/" + "_build.log";
    }
    
    private static boolean isRelease(final BuildParameters buildParameters) {
        return buildParameters.getPropertyBool("release_mode", true);
    }
    
    public boolean isRelease() {
        return isRelease(this.m_props);
    }
    
    public boolean usingCluster() {
        return this.getParameter((BuildConfOption<Boolean>)MaxCompilerBuildConf.use_cluster);
    }
    
    public String getBuildName() {
        return this.m_build_name;
    }
    
    public String getSlicStemName() {
        return this.m_slic_stem_name;
    }
    
    public String getBuildRootName() {
        if (this.m_parent_bm == null) {
            return this.m_root_dir.getName();
        }
        return this.m_parent_bm.getBuildRootName();
    }
    
    public String getUniqueBuildName() {
        return this.m_unique_build_name;
    }
    
    public BuildProcessDB getBuildProcessDB() {
        return this.m_processdb_local;
    }
    
    public CoreCache getCoreCache() {
        return this.m_core_cache;
    }
    
    public boolean isTargetSimulation() {
        return this.m_target_simulation;
    }
    
    public void setPlatform(final Platform<?, ?> platform) {
        this.m_platform = platform;
        if (this.m_platform != null) {
            this.m_platform.setBuildManager(this);
        }
    }
    
    public boolean hasPlatformSet() {
        return this.m_platform != null;
    }
    
    public Platform<?, ?> getPlatform() {
        if (this.m_platform == null) {
            throw new MaxDCException(this, "No platform has been registered.");
        }
        return this.m_platform;
    }
    
    private void mkdir(final File file) {
        if (!file.exists() && !file.mkdirs()) {
            throw new MaxDCException(this, "Could not make dir: " + file.getAbsolutePath());
        }
    }
    
    public String getHostname() {
        String s;
        try {
            s = InetAddress.getLocalHost().getHostName();
        }
        catch (final UnknownHostException ex) {
            s = "<unknown host>";
        }
        return s;
    }
    
    public String getUserName() {
        return System.getProperty("user.name");
    }
    
    public File getRootDir() {
        return this.m_root_dir;
    }
    
    public File getResultsDir() {
        return new File(String.valueOf(this.m_root_dir.getAbsolutePath()) + BuildManager.s_resultsDir);
    }
    
    private File getAbsoluteFile(final String s) {
        File file;
        try {
            file = new File(this.m_root_dir.getCanonicalPath(), new File(s).getPath());
        }
        catch (final Exception ex) {
            throw new MaxDCException(this, "Error getting absolute path (" + ex + "): " + s);
        }
        return file;
    }
    
    public void setWorkingDir(final String s) {
        this.assertHasBuildDirLock();
        final String[] array = s.replace('/', '\n').replace('\\', '\n').trim().split("\n+");
        final File[] array2 = new File[array.length];
        for (int i = 0; i < array.length; ++i) {
            array2[i] = new File(array[i]);
        }
        this.m_dir_stack.clear();
        final Iterator<File> iterator = MaxFile.tidyPath(array2).iterator();
        while (iterator.hasNext()) {
            final File file = this.m_dir_stack.push(iterator.next());
        }
        this.mkdir(this.getAbsoluteFile(this.getWorkingDir()));
    }
    
    public void pushWorkingDir(final String s) {
        this.assertHasBuildDirLock();
        this.mkdir(this.getAbsoluteFile(new File(this.getWorkingDirStr(), s).toString()));
        final File file = this.m_dir_stack.push(new File(s));
    }
    
    public void popWorkingDir() {
        this.assertHasBuildDirLock();
        final File file = this.m_dir_stack.pop();
    }
    
    public String getWorkingDirStr() {
        return this.getWorkingDir().toString().replace("\\", "/");
    }
    
    public String getWorkingDirStrAbsolute() {
        return String.valueOf(this.m_root_dir.getAbsolutePath()) + "/" + this.getWorkingDir();
    }
    
    public File[] getWorkingDirStack() {
        return MaxFile.tidyPath(this.m_dir_stack.toArray(new File[0])).toArray(new File[0]);
    }
    
    public void setWorkingDirStack(final File[] array) {
        this.m_dir_stack.clear();
        this.m_dir_stack.addAll(Arrays.asList((File[])array));
    }
    
    public String getWorkingDir() {
        final File[] array = new File[this.m_dir_stack.size()];
        this.m_dir_stack.copyInto(array);
        for (int i = 1; i < array.length; ++i) {
            array[0] = new File(array[0], array[i].toString());
        }
        return array[0].toString();
    }
    
    public File getAbsoluteWorkingDir() {
        return this.getAbsoluteFile(this.getWorkingDir());
    }
    
    public boolean removeDirectory(final String s) {
        this.assertHasBuildDirLock();
        final File file = new File(this.getRootDir(), new File(this.getWorkingDir(), s).toString());
        if (file.exists()) {
            this.logInfo("Deleting directory: " + s);
            MaxFile.deleteDirTree(file);
            return true;
        }
        this.logInfo("Not deleting directory (does not exist): " + s);
        return false;
    }
    
    public synchronized void addBuildFile(final BuildFile buildFile) {
        this.assertHasBuildDirLock();
        this.m_build_files.add(buildFile);
    }
    
    public void addBuildFiles(final Collection<BuildFile> collection) {
        this.assertHasBuildDirLock();
        final Iterator<BuildFile> iterator = collection.iterator();
        while (iterator.hasNext()) {
            this.m_build_files.add(iterator.next());
        }
    }
    
    public void addBuildPass(final BuildPass buildPass) {
        this.assertHasBuildDirLock();
        this.m_build_passes.add(buildPass);
    }
    
    public void addCreationLogger(final CreationLogger creationLogger) {
        this.assertHasBuildDirLock();
        this.m_creation_loggers.add(new WeakReference<CreationLogger>(creationLogger));
    }
    
    public void addProcAsync(final ProcAsync procAsync) {
        this.assertHasBuildDirLock();
        if (procAsync.getBuildManager() != this) {
            throw new MaxDCException(this, "Tried to add async proc from another build manager.");
        }
        synchronized (this.m_build_jobs) {
            this.m_build_jobs.add(procAsync);
        }
    }
    
    public void removeProcAsync(final ProcAsync procAsync) {
        this.assertHasBuildDirLock();
        if (procAsync.getBuildManager() != this) {
            throw new MaxDCException("Tried to remove async proc from another build manager.");
        }
        synchronized (this.m_build_jobs) {
            this.m_build_jobs.remove(procAsync);
        }
    }
    
    public static boolean isWindows() {
        return System.getProperty("os.name").matches(".*Windows.*");
    }
    
    public static String getUser() {
        if (isWindows()) {
            return Env.get("USERNAME");
        }
        return Env.get("USER");
    }
    
    public Entity getRootEntity() {
        this.assertHasBuildDirLock();
        if (this.m_root_entity == null) {
            throw new MaxDCException(this, "Cannot get root entity signature before build() is called.");
        }
        return this.m_root_entity;
    }
    
    public List<BuildFile> findBuildFiles(final String s, final Class<?>... array) {
        final ArrayList<BuildFile> list = new ArrayList<>();
        for (final BuildFile buildFile : this.m_build_files) {
            if (buildFile.getRelativeName().matches(s)) {
                for (int length = array.length, i = 0; i < length; ++i) {
                    if (array[i].isInstance(buildFile)) {
                        list.add(buildFile);
                        break;
                    }
                }
            }
        }
        return list;
    }
    
    public List<BuildFile> findBuildFiles(final String s) {
        return this.findBuildFiles(s, BuildFile.class);
    }
    
    public List<BuildFile> findBuildFiles(final Class<?>... array) {
        return this.findBuildFiles(".*", array);
    }
    
    public <T extends BuildFile> List<T> findBuildFiles(final Class<T> clazz) {
        final ArrayList<T> list = new ArrayList<>();
        for (final BuildFile buildFile : this.m_build_files) {
            if (clazz.isInstance(buildFile)) {
                list.add(clazz.cast(buildFile));
            }
        }
        return list;
    }
    
    public <T extends BuildFile> T findBuildFile(final Class<T> clazz) {
        final List<T> list = this.findBuildFiles(clazz);
        if (list.size() != 1) {
            throw new MaxDCException(this, "expected 1 '" + clazz.getSimpleName() + "' build file, found " + list.size());
        }
        return (T)list.get(0);
    }
    
    public <T extends BuildFile> T tryFindBuildFile(final Class<T> clazz) {
        final List<T> list = this.findBuildFiles(clazz);
        return (T)((list.size() == 1) ? ((T)list.get(0)) : null);
    }
    
    public void removeBuildFiles(final List<? extends BuildFile> list) {
        this.assertHasBuildDirLock();
        this.m_build_files.removeAll(list);
    }
    
    public void removeBuildFile(final BuildFile buildFile) {
        this.assertHasBuildDirLock();
        this.m_build_files.remove(buildFile);
    }
    
    void buildEntity(final Entity entity) {
        this.assertHasBuildDirLock();
        this.m_entity_builder.build(entity);
        this.logInfo("Waiting for any external asynchronous jobs to complete (e.g. MegaWizard/CoreGen)...");
        try {
            while (this.m_build_jobs.size() > 0) {
                final ProcAsync procAsync = this.m_build_jobs.get(0);
                final ProcResult procResult = procAsync.waitForCompletion();
                if (procResult.shouldRetry()) {
                    this.logInfo("Retrying " + procAsync.toString());
                    final EntityBlackBox entityBlackBox = procResult.getEntityBlackBox();
                    entityBlackBox.getBuildManager().pushWorkingDir(entityBlackBox.getSignature());
                    procResult.getEntityBlackBox().implement();
                    entityBlackBox.getBuildManager().popWorkingDir();
                }
            }
        }
        catch (final RuntimeException ex) {
            this.killAllOutstandingJobs();
            throw ex;
        }
        this.logInfo("All asynchronous jobs are now completed.");
    }
    
    private void markEndOfBuild() {
        BuildReport.BuildLogDone(this.m_log.getLogFile());
        BuildReport.EndOfBuild();
        final Date date = new Date();
        final AmountOfTime amountOfTime = AmountOfTime.durationBetween(this.m_start_time, date);
        this.logProgress("Build completed: " + date + " (took " + amountOfTime + ")");
        if (this.m_parent_bm != null) {
            this.logProgress("Sub-build location: " + this.getRootDir().getAbsolutePath());
        }
        else if (!this.m_target_simulation) {
            this.sendMail(amountOfTime);
        }
    }
    
    private void sendMail(final AmountOfTime amountOfTime) {
        final String s = System.getenv("MAXBUILDUPDATEMAILTO");
        if (s == null || !s.contains("@")) {
            return;
        }
        final MailFactory mailFactory = new MailFactory("builds@maxeler.com", "Build Update");
        final String s2 = this.getBuildRootName();
        String[] array;
        for (int length = (array = s.split(",")).length, i = 0; i < length; ++i) {
            final MailFactory.MailMessage mailMessage = mailFactory.newMessage("Build Update " + s2, array[i]);
            if (!this.m_non_fatal_errors.isEmpty()) {
                mailMessage.appendLn("Build (" + s2 + ") terminated as " + this.m_non_fatal_errors.size() + " build passes failed.");
                final Iterator<BuildPassNonFatalException> iterator = this.m_non_fatal_errors.iterator();
                while (iterator.hasNext()) {
                    mailMessage.appendLn(iterator.next().getMessage());
                }
                mailMessage.appendLn("Build took " + amountOfTime);
            }
            else if (this.m_bestTimingScore != null && this.m_bestTimingScore != 0) {
                mailMessage.appendLn("Build (" + s2 + ") failed to meet timing. Best timing score = " + this.m_bestTimingScore);
                mailMessage.appendLn("Build took " + amountOfTime);
            }
            else {
                mailMessage.appendLn("Build (" + s2 + ") completed in " + amountOfTime);
            }
            mailMessage.send();
        }
        this.logInfo("Sending build update mail to: " + s);
    }
    
    public void runBuildPasses() {
        this.assertHasBuildDirLock();
        final int size = this.m_build_passes.size();
        if (this.m_parent_bm == null) {
            this.logProgress("Running back-end " + (this.m_target_simulation ? "simulation" : "") + " build (" + size + " phases)");
            BuildReport.Start(this.m_build_name, size);
        }
        try {
            final StopWatch stopWatch = new StopWatch();
            for (int i = 0; i < size; ++i) {
                final BuildPass buildPass = this.m_build_passes.get(i);
                final String s = buildPass.getClass().getSimpleName();
                final String s2 = buildPass.getTitle().toString();
                BuildReport.StartTask(s2);
                if (this.m_parent_bm == null) {
                    this.logProgress("(%d/%d) - %s (%s)", i + 1, size, s2, s);
                }
                stopWatch.start();
                try {
                    buildPass.pass(this);
                }
                catch (final BuildPassNonFatalException ex) {
                    if (ex.getMessage() != null) {
                        this.logError(ex.getMessage());
                    }
                    this.logError("Build pass '%s' failed.", s2);
                    this.m_non_fatal_errors.add(ex);
                }
                stopWatch.stop();
                if (!this.isSubBuildManager()) {
                    MaxInfoCreator.writeMaxInfoFile(this);
                }
                this.logInfo("Build pass '%s' took %s.", s, stopWatch);
                BuildReport.Worked(1);
            }
        }
        finally {
            BuildReport.Done();
        }
        BuildReport.Done();
    }
    
    public synchronized void killAllOutstandingJobs() {
        this.assertHasBuildDirLock();
        while (this.m_build_jobs.size() > 0) {
            this.m_build_jobs.get(0).kill();
        }
        this.killAllCopiedSubBuildManagerJobs();
    }
    
    public synchronized void killAllCopiedSubBuildManagerJobs() {
        this.assertHasBuildDirLock();
        for (int i = 0; i < this.m_copied_submanager_jobs.size(); ++i) {
            final ProcAsync procAsync = this.m_copied_submanager_jobs.get(i);
            if (procAsync != null) {
                final List<ProcAsync> build_jobs = procAsync.getBuildManager().m_build_jobs;
                synchronized (build_jobs) {
                    if (build_jobs.contains(procAsync)) {
                        procAsync.kill();
                    }
                }
            }
        }
        this.m_copied_submanager_jobs.clear();
    }
    
    private synchronized void waitForAllOutstandingJobs() {
        this.assertHasBuildDirLock();
        while (this.m_build_jobs.size() > 0) {
            this.logInfo("Waiting for " + this.m_build_jobs.get(0).toString());
            try {
                final ProcResult procResult = this.m_build_jobs.get(0).waitForCompletion();
            }
            catch (final RuntimeException ex) {
                this.logInfo("Received exception while waiting for job: " + ex.toString());
                try {
                    this.m_build_jobs.get(0).kill();
                }
                catch (final RuntimeException ex2) {}
            }
        }
        for (int i = 0; i < this.m_copied_submanager_jobs.size(); ++i) {
            final ProcAsync procAsync = this.m_copied_submanager_jobs.get(i);
            if (procAsync != null) {
                final List<ProcAsync> build_jobs = procAsync.getBuildManager().m_build_jobs;
                synchronized (build_jobs) {
                    if (build_jobs.contains(procAsync)) {
                        this.logInfo("Waiting for " + procAsync.toString());
                        try {
                            final ProcResult procResult2 = procAsync.waitForCompletion();
                        }
                        catch (final RuntimeException ex3) {
                            this.logInfo("Received exception while waiting for job: " + ex3.toString());
                            try {
                                procAsync.kill();
                            }
                            catch (final RuntimeException ex4) {}
                        }
                    }
                }
            }
        }
        this.m_copied_submanager_jobs.clear();
    }
    
    public void copyOutstandingJobsToParent() {
        this.assertHasBuildDirLock();
        if (this.m_parent_bm == null) {
            throw new MaxDCException(this, "This is not a child build manager.");
        }
        synchronized (this.m_build_jobs) {
            for (int i = 0; i < this.m_build_jobs.size(); ++i) {
                this.m_parent_bm.m_copied_submanager_jobs.add(this.m_build_jobs.get(i));
            }
        }
    }
    
    public void destroy() {
        this.killAllOutstandingJobs();
    }
    
    public List<BuildFile> build(final Entity root_entity) {
        this.assertHasBuildDirLock();
        if (this.m_root_entity != null) {
            throw new MaxDCException(this, "build() has already been called for this BuildManager.");
        }
        if (this.m_platform != null) {
            this.addBuildFiles(this.m_platform.getPlatformSpecificBuildFiles(this));
        }
        this.m_entity_builder.collectTopLevelBuildFiles();
        this.createPreliminaryCompilerDesignFile("Main build");
        this.logProgress("Generating input files (VHDL, netlists, vendor specific IP cores)");
        this.buildEntity(this.m_root_entity = root_entity);
        this.runBuildPasses();
        this.finalizeBuild();
        this.m_log.closeLogFile();
        return new ArrayList<BuildFile>(this.m_build_files);
    }
    
    public void createPreliminaryCompilerDesignFile(final String s) {
        if (this.m_firstCall_createPrelimCompilerDesignFile) {
            this.logInfo("Creating a preliminary MaxCompilerDesignData.dat. (@ " + s + ")");
            this.m_firstCall_createPrelimCompilerDesignFile = false;
        }
        else {
            this.logInfo("Updating the preliminary MaxCompilerDesignData.dat. (@ " + s + ")");
        }
        final List<MaxFileDataSegment> list = MaxFileManager.getMaxFileManager(this).getMaxFileDataSegmentsUnchecked();
        final MaxFileDataFile maxFileDataFile = new MaxFileDataFile(this, "MaxCompilerDesignData.dat", false);
        maxFileDataFile.delete();
        final PrintStream printStream = maxFileDataFile.getPrintStream();
        printStream.println("/*             !!!!!!!!!!!!!!!!!!!!!!!!!!!!! */");
        printStream.println("/*             !!!!!!!!!!!!!!!!!!!!!!!!!!!!! */");
        printStream.println("/*             !!                         !! */");
        printStream.println("/*             !!  Preliminary   Version  !! */");
        printStream.println("/*             !!                         !! */");
        printStream.println("/*             !!!!!!!!!!!!!!!!!!!!!!!!!!!!! */");
        printStream.println("/*             !!!!!!!!!!!!!!!!!!!!!!!!!!!!! */");
        printStream.println("/* At: " + s + " */");
        printStream.println("#error");
        final Iterator<MaxFileDataSegment> iterator = list.iterator();
        while (iterator.hasNext()) {
            printStream.println(iterator.next().getMaxFileString());
        }
        printStream.close();
    }
    
    public synchronized void addBuildFileResult(final BuildFile buildFile) {
        if (this.m_build_finished) {
            this.logInfo("Adding post-build result file: " + buildFile.getFileName() + " (" + buildFile.getClass().getSimpleName() + ")");
            final String workingDir = this.getWorkingDir();
            this.setWorkingDir(BuildManager.s_resultsDir);
            buildFile.linkToHere();
            this.setWorkingDir(workingDir);
        }
        else {
            this.m_result_files.add(buildFile);
        }
    }
    
    public void finalizeBuild() throws BuildFailedTimingException {
        this.assertHasBuildDirLock();
        this.logInfo("Final result files after build: ");
        this.setWorkingDir(BuildManager.s_resultsDir);
        final File file = this.getAbsoluteWorkingDir();
        File[] array;
        for (int length = (array = file.listFiles()).length, i = 0; i < length; ++i) {
            final File file2 = array[i];
            if (!file2.delete()) {
                this.logInfo("failed to delete: " + file2);
            }
        }
        if (this.m_result_files.size() == 0) {
            this.logInfo("(none)");
        }
        for (final BuildFile buildFile : this.m_result_files) {
            buildFile.linkToHere();
            this.logInfo(String.valueOf(buildFile.getFileName()) + " (" + buildFile.getClass().getSimpleName() + ")");
        }
        if (this.hasParameter(MaxCompilerBuildConf.build.copy_results_to)) {
            final File file3 = new File(this.getParameter((BuildConfOption<String>)MaxCompilerBuildConf.build.copy_results_to));
            final String[] array2 = { ".*\\.max", ".*\\.h", ".*\\.pxg" };
            this.logInfo("Result files will be copied to '%s'", file3.getAbsolutePath());
            if (!file3.exists()) {
                this.logInfo("Creating directory '%s'", file3.getAbsolutePath());
                file3.mkdirs();
            }
            for (final BuildFile buildFile2 : this.m_result_files) {
                final String[] array3;
                final int length2 = (array3 = array2).length;
                int j = 0;
                while (j < length2) {
                    if (buildFile2.getFileName().matches(array3[j])) {
                        final String string = String.valueOf(file3.getAbsolutePath()) + "/" + buildFile2.getFileName();
                        final File file4 = new File(string);
                        int n = 1;
                        if (file4.exists() && buildFile2.getAbsoluteFile().length() == file4.length()) {
                            try {
                                Hasher h = new Hasher();
                                h.append(new File(string));
                                n = (h.getHash().equals(buildFile2.getHash()) ? 0 : 1);
                            }
                            catch (final MaxDCException ex) {}
                        }
                        if (n != 0) {
                            this.logInfo("Copying result '%s' to '%s'", buildFile2.getFileName(), string);
                            buildFile2.copy(string);
                            BuildReport.ResultFileCopied(buildFile2);
                            break;
                        }
                        this.logInfo("Result '%s' is identical to '%s': not overwriting.", buildFile2.getFileName(), string);
                        break;
                    }
                    else {
                        ++j;
                    }
                }
            }
        }
        this.setWorkingDir(BuildManager.s_scratchDir);
        this.m_build_finished = true;
        if (this.m_parent_bm == null) {
            this.showBuildSummary(file, this.m_bestTimingScore);
        }
        this.logInfo("Waiting for any outstanding jobs to finish... ");
        try {
            this.waitForAllOutstandingJobs();
        }
        catch (final RuntimeException ex2) {
            this.killAllOutstandingJobs();
            throw ex2;
        }
        finally {
            for (final BuildManagerCallBack buildManagerCallBack : this.m_final_callbacks) {
                try {
                    buildManagerCallBack.buildManagerFinalCallback(this);
                }
                catch (final RuntimeException ex3) {}
            }
        }
        for (final BuildManagerCallBack buildManagerCallBack2 : this.m_final_callbacks) {
            try {
                buildManagerCallBack2.buildManagerFinalCallback(this);
            }
            catch (final RuntimeException ex4) {}
        }
        if (!this.m_non_fatal_errors.isEmpty()) {
            this.sendMail(AmountOfTime.durationBetween(this.m_start_time, new Date()));
            this.logError("Terminating build as some build passes did not run successfully.");
            throw new MaxDCException(this, "%d build passes failed.", new Object[] { this.m_non_fatal_errors.size() });
        }
        if (this.m_bestTimingScore != null && this.m_bestTimingScore != 0) {
            this.sendMail(AmountOfTime.durationBetween(this.m_start_time, new Date()));
            throw new BuildFailedTimingException(this, this.m_bestTimingScore);
        }
        this.markEndOfBuild();
    }
    
    private void showBuildSummary(final File file, final Integer n) {
        final BuildResourceUsage buildResourceUsage = BuildResourceUsage.getBuildResourceUsage(this);
        if (buildResourceUsage != null) {
            final GenericResourceSummary genericResourceSummary = buildResourceUsage.getResourceSummary();
            if (genericResourceSummary == null) {
                this.logError("Unable to log resource report: No final resource summary available.");
            }
            else {
                this.logProgress("");
                genericResourceSummary.log(this, GenericResourceSummary.Stage.Final);
                this.logProgress("");
                MaxInfoCreator.writeMaxInfoFile(this);
            }
        }
        final BuildFileMaxFile buildFileMaxFile = this.tryFindBuildFile(BuildFileMaxFile.class);
        if (buildFileMaxFile != null) {
            this.logProgress("MaxFile: " + new File(file, buildFileMaxFile.getFileName()).getAbsolutePath() + " (MD5Sum: " + buildFileMaxFile.getHash() + ")");
        }
        if (n != null && n != 0) {
            this.logError("FAILED TO MEET TIMING! Best timing score: %d. We strongly recommend you do not use this .max file.", n);
            this.logError("Please see the timing report for more information and consult the MaxCompiler optimization cheat sheet for ideas on how to improve your design.");
            this.logError("Timing report: %s", MaxTimingAnalyserUtils.getReportIndex(this.getRootDir().getAbsolutePath()));
            this.logError("MaxCompiler documentation: %s", String.valueOf(Env.get("MAXCOMPILERDIR")) + File.separator + "docs" + File.separator + "maxcompiler-optimization-cheat-sheet.pdf");
        }
    }
    
    public void setBestTimingScore(final int n) {
        this.m_bestTimingScore = n;
    }
    
    public boolean hasNoErrors() {
        return this.m_non_fatal_errors.isEmpty();
    }
    
    Set<BuildFile> getBuildFiles() {
        return Collections.unmodifiableSet((Set<? extends BuildFile>)this.m_build_files);
    }
    
    public String getLogTail(final int n) {
        return this.m_log.getTail(n);
    }
    
    public void logSetConsoleTag(final String consoleTag) {
        this.m_log.setConsoleTag(consoleTag);
    }
    
    public void logSetLogTag(final String logTag) {
        this.m_log.setLogTag(logTag);
    }
    
    public void logError(final String s, final Object... array) {
        this.logError(String.format(s, array));
    }
    
    public void logError(final String s) {
        this.m_log.log(BuildLogger.LogLevel.ERROR, s);
    }
    
    public void logProgress(final String s, final Object... array) {
        this.logProgress(String.format(s, array));
    }
    
    public void logProgress(final String s) {
        this.m_log.log(BuildLogger.LogLevel.PROGRESS, s);
    }
    
    public void logUser(final String s, final Object... array) {
        this.logUser(String.format(s, array));
    }
    
    public void logUser(final String s) {
        this.m_log.log(BuildLogger.LogLevel.USER, s);
    }
    
    public void logWarning(final String s, final Object... array) {
        this.logWarning(String.format(s, array));
    }
    
    public void logWarning(final String s) {
        this.m_log.log(BuildLogger.LogLevel.WARNING, s);
    }
    
    public void logInfo(final String s, final Object... array) {
        this.logInfo(String.format(s, array));
    }
    
    public void logInfo(final String s) {
        this.m_log.log(BuildLogger.LogLevel.INFO, s);
    }
    
    public void logInfo() {
        this.logInfo("");
    }
    
    public void logPushVerbosity(final boolean b) {
        this.m_log.pushVerbosity(b);
    }
    
    public void logPopVerbosity() {
        this.m_log.popVerbosity();
    }
    
    public void log(final BuildLogger.LogLevel logLevel, final String s) {
        this.m_log.log(logLevel, s);
    }
    
    public void log(final BuildLogger.LogLevel logLevel, final String s, final Object... array) {
        this.log(logLevel, String.format(s, array));
    }
    
    public Date logInfoAtTime(final String s) {
        return this.logInfoAtTimeWithDuration(s, null);
    }
    
    public Date logInfoAtTimeWithDuration(final String s, final Date date) {
        final Date date2 = new Date();
        String s2 = String.valueOf(s) + " (" + this.m_logDateFormat.format(date2);
        if (date != null) {
            s2 = String.valueOf(s2) + ", time elapsed: " + AmountOfTime.durationBetween(date, date2);
        }
        this.logInfo(String.valueOf(s2) + ")");
        return date2;
    }
    
    public void setGlobalVariable(final String s, final Object o) {
        this.assertHasBuildDirLock();
        final Object o2 = this.m_global_variables.put(s, o);
    }
    
    public <T> T getGlobalVariable(final String s) {
        return this.getGlobalVariable(s, (T)null);
    }
    
    @SuppressWarnings({"unchecked"})
    public <T> T getGlobalVariable(final String s, final T t) {
        return this.m_global_variables.containsKey(s) ? (T)this.m_global_variables.get(s) : t;
    }
    
    public BuildManager addFinalCallback(final BuildManagerCallBack buildManagerCallBack) {
        this.assertHasBuildDirLock();
        this.m_final_callbacks.add(buildManagerCallBack);
        return this;
    }
    
    public <N> N getParameter(final BuildConfOption<N> buildConfOption) {
        String s = this.m_props.getProperty(buildConfOption.getName());
        if (s == null) {
            for (final BuildConfOption buildConfOption2 : buildConfOption.getAliasedOptions()) {
                s = this.m_props.getProperty(buildConfOption2.getName());
                if (s != null) {
                    this.logWarning("Using deprecated build conf option " + buildConfOption2.getName() + ". This conf option has been replaced by " + buildConfOption.getName() + " and will be removed in future releases.\n" + "Plase consider replacing it with " + buildConfOption.getName() + " instead.");
                    break;
                }
            }
        }
        if (s == null && buildConfOption.isMandatory()) {
            throw new BuildConfOption.MaxCompilerBuildConfException(this, "Missing value for mandatory MaxCompilerBuildConf parameter: " + buildConfOption);
        }
        if (s == null && !buildConfOption.getDefaultValue().isPresent()) {
            throw new BuildConfOption.MaxCompilerBuildConfException(this, "No value available for optional MaxCompilerBuildConf parameter: " + buildConfOption);
        }
        return (N)((s == null) ? buildConfOption.getDefaultValue().get() : ((N)buildConfOption.validateConf(s)));
    }
    
    public boolean hasParameter(final BuildConfOption<?> buildConfOption) {
        return this.m_props.getProperty(buildConfOption.getName()) != null;
    }
    
    public <N> void setParameter(final BuildConfOption<N> buildConfOption, final N n) {
        final Object o = this.m_props.setProperty(buildConfOption.getName(), n.toString());
    }
    
    @Deprecated
    public String getParameter(final String s) {
        return this.m_props.getProperty(s);
    }
    
    @Deprecated
    public String getParameter(final String s, final String s2) {
        return this.getParameter((BuildConfOption<String>)MaxCompilerBuildConf.createDummyConf(s, s2));
    }
    
    @Deprecated
    public Integer getParameter(final String s, final Integer n) {
        return this.getParameter((BuildConfOption<Integer>)MaxCompilerBuildConf.createDummyConf(s, n));
    }
    
    @Deprecated
    public Boolean getParameter(final String s, final Boolean b) {
        return this.getParameter((BuildConfOption<Boolean>)MaxCompilerBuildConf.createDummyConf(s, b));
    }
    
    @Deprecated
    public void setParameter(final String s, final String s2) {
        final Object o = this.m_props.setProperty(s, s2);
    }
    
    public boolean isSubBuildManager() {
        return this.m_parent_bm != null;
    }
    
    public BuildManager getParentBuildManager() {
        return this.m_parent_bm;
    }
    
    public Collection<EntityDatabase.EntitySummary> getEntitySummaries() {
        final ArrayList<EntityDatabase.EntitySummary> list = new ArrayList<>();
        list.addAll(this.m_entity_builder.getEntityDatabase().getEntitySummaries());
        final Iterator<BuildManager> iterator = this.m_sub_bms.iterator();
        while (iterator.hasNext()) {
            list.addAll(iterator.next().getEntitySummaries());
        }
        return list;
    }
    
    public void setCacheManager(final CacheManager buildCacheManager) {
        this.m_buildCacheManager = buildCacheManager;
    }
    
    public CacheManager getCacheManager() {
        return this.m_buildCacheManager;
    }
    
    public void setMaxInfo(final MaxInfo maxInfoData) {
        if (this.m_maxInfoData == null) {
            this.m_maxInfoData = maxInfoData;
            MaxInfoCreator.writeMaxInfoFile(this);
        }
    }
    
    public MaxInfo getMaxInfo() {
        return this.m_maxInfoData;
    }
    
    public IpCoreInfo getIpCoreInfo() {
        return this.m_ipCoreInfo;
    }
    
    public static class PrevBuildData
    {
        private boolean m_is_restarted_build;
        private Date m_start_date;
        private BuildParameters m_initial_properties;
        private String m_path_to_prev_mxru;
        private final Map<File, String> m_relative_source_file_paths;
        
        public PrevBuildData() {
            this.m_relative_source_file_paths = new HashMap<File, String>();
        }
        
        static /* synthetic */ void access$3(final PrevBuildData prevBuildData, final boolean is_restarted_build) {
            prevBuildData.m_is_restarted_build = is_restarted_build;
        }
        
        static /* synthetic */ void access$4(final PrevBuildData prevBuildData, final Date start_date) {
            prevBuildData.m_start_date = start_date;
        }
        
        static /* synthetic */ void access$5(final PrevBuildData prevBuildData, final BuildParameters initial_properties) {
            prevBuildData.m_initial_properties = initial_properties;
        }
        
        static /* synthetic */ void access$7(final PrevBuildData prevBuildData, final String path_to_prev_mxru) {
            prevBuildData.m_path_to_prev_mxru = path_to_prev_mxru;
        }
    }
    
    public static class ChooseBuildRootRes
    {
        public String build_name;
        public String build_dir;
    }
    
    public class LostBuildLock extends MaxDCException
    {
        public static final long serialVersionUID = 1L;
        
        private LostBuildLock() {
            super(BuildManager.this, "This build manager has terminated its build and its lock file has been stolen");
        }
    }
}
