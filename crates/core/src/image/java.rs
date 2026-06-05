use jni::objects::GlobalRef;
use jni::sys::jmethodID;
use jni::{InitArgsBuilder, JNIVersion, JavaVM};
use log::{info, warn};
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock}; // Use OnceCell from the crate, not std

pub struct JavaWrapper {
    pub jvm: Option<JavaVM>,
    pub m_bioformats_class: Option<GlobalRef>,
    // The instance of BioFormatsWrapper (GlobalRef so it's not GC'd)
    // Cached Method IDs (valid for the lifetime of the JVM)
    pub m_constructor: Option<jmethodID>,
    pub m_close: Option<jmethodID>,
    pub m_get_image_properties: Option<jmethodID>,
    pub m_read_image_tile: Option<jmethodID>,
    pub m_reserved_ram: u64,
}

unsafe impl Send for JavaWrapper {}
unsafe impl Sync for JavaWrapper {}

pub static JAVA_WRAPPER: OnceLock<JavaWrapper> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Initializes the global Java Runtime environment and stores it in the `JAVA_WRAPPER` OnceLock.
///
/// This function must be called exactly once before any image reading operations take place.
/// It configures the Java Virtual Machine (JVM) with a specific memory limit (in bytes)
/// to prevent the Java heap from consuming all available system memory.
/// JAVA_WRAPPER is used and needed by the image_reader
///
/// # Arguments
/// * `memory_limit` - The maximum heap size allowed for the JVM (e.g., 256 * 1024 * 1024 for 256MB).
///
/// # Errors
/// Returns an error if the JVM fails to start, if the Bio-Formats classes are missing,
/// or if the `JAVA_WRAPPER` has already been initialized.
// We need a Mutex to gate-keep the actual creation logic
pub fn init_java_wrapper(memory_limit: u64) -> Result<(), Box<dyn Error>> {
    // 1. FAST PATH: Check if already initialized (lock-free)
    if JAVA_WRAPPER.get().is_some() {
        return Ok(());
    }

    // 2. SLOW PATH: Only one thread can enter this block at a time
    let _guard = INIT_LOCK.lock().map_err(|e| e.to_string())?;

    // 3. DOUBLE-CHECK: Another thread might have finished while we were waiting for the lock
    if let Some(_) = JAVA_WRAPPER.get() {
        warn!("JAVA_WRAPPER was initialized by another thread while waiting.");
        return Ok(());
    }

    // 4. INITIALIZE: Now we are 100% sure we are the only ones doing this
    let java_wrapper = JavaWrapper::new(memory_limit)?;

    // 5. STORE: Set the global variable
    JAVA_WRAPPER
        .set(java_wrapper)
        .map_err(|_| "Internal Error: OnceLock already full despite double-check")?;

    Ok(())
}

/// Java wrapper
impl JavaWrapper {
    /// Describe this function.
    ///
    /// # Arguments
    ///
    /// - `reserved_ram_bytes` (`u64`) - RAM which should be reseved for the JVM in bytes
    ///
    /// # Returns
    ///
    /// - `Result<Self, Box<dyn Error>>` - Pointer to a JavaWrapper instance or error
    ///
    /// # Errors
    ///
    /// Error if not possible to init JVM
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::...;
    ///
    /// let _ = new();
    /// ```
    pub fn new(reserved_ram_bytes: u64) -> Result<Self, Box<dyn Error>> {
        // 1. Create the base struct
        let mut wrapper = Self {
            jvm: None,
            m_bioformats_class: None,
            m_constructor: None,
            m_close: None,
            m_get_image_properties: None,
            m_read_image_tile: None,
            m_reserved_ram: reserved_ram_bytes,
        };

        // 2. Call init on the local variable
        wrapper.init(reserved_ram_bytes)?;

        // 3. Return the fully initialized wrapper
        Ok(wrapper)
    }

    /// Init the JVM which is used to call Bioformats methods for loading images
    ///
    /// # Arguments
    ///
    /// - `&mut self` (`undefined`) - Describe this parameter.
    /// - `reserved_ram_bytes` (`u64`) - RAM which should be reseved for the JVM in bytes
    ///
    /// # Returns
    ///
    /// - `Result<(), Box<dyn Error>>` - Pointer to a JavaWrapper instance or error
    ///
    /// # Errors
    ///
    /// Error if not possible to init JVM
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::...;
    ///
    /// let _ = init();
    /// ```
    fn init(&mut self, reserved_ram_bytes: u64) -> Result<(), Box<dyn Error>> {
        // 1. Calculate Classpath
        self.set_path();
        let classpath = if cfg!(target_os = "windows") {
            "-Djava.class.path=./;java/bioformats.jar;java".to_string()
        } else if cfg!(target_os = "macos") {
            let exe_path = env::current_exe()?;
            let contents_dir = exe_path
                .parent()
                .and_then(|p| p.parent())
                .ok_or("Could not find .app/Contents directory structure")?;
            let jar_path = contents_dir.join("Java").join("bioformats");
            let other_class_path = contents_dir.join("Java");
            let class_path_arg = format!(
                "-Djava.class.path=./:{}:{}:{}",
                jar_path.display(),
                "java",
                other_class_path.display()
            );
            class_path_arg
        } else {
            "-Djava.class.path=./:java/bioformats.jar:java".to_string()
        };

        // 2. Calculate RAM
        let jvm_ram_mb = (reserved_ram_bytes as f64 / 1_000_000.0).ceil() as u64;
        let ram_arg = format!("-Xmx{}m", jvm_ram_mb);
        let headless_arg = "-Djava.awt.headless=true";

        // 3. Initialize JVM
        // Note: The 'jni' crate will look for jvm.dll/so in the PATH we just updated
        let jvm_args = InitArgsBuilder::new()
            .version(JNIVersion::V8)
            .option(classpath)
            .option(ram_arg)
            .option(headless_arg)
            .build()?;

        let jvm = JavaVM::new(jvm_args)?;

        {
            // Inside your init function's scoped block:
            let mut env = jvm.attach_current_thread()?;

            let class_name = "BioFormatsWrapper";
            let local_cls = env.find_class(class_name)?;

            let global_cls = env.new_global_ref(local_cls)?;
            self.m_bioformats_class = Some(global_cls);

            // 1.  Cache the Method IDs
            // C++: mConstructor = myGlobEnv->GetMethodID(mBioformatsClass, "<init>", "(Ljava/lang/String;)V");
            // We use .into_raw() to store them as simple pointers
            if let Some(ref class_ref) = self.m_bioformats_class {
                self.m_constructor = Some(
                    env.get_method_id(class_ref, "<init>", "(Ljava/lang/String;Z)V")?
                        .into_raw(),
                );

                self.m_close = Some(env.get_method_id(class_ref, "close", "()V")?.into_raw());

                self.m_get_image_properties = Some(
                    env.get_method_id(class_ref, "getImageProperties", "()Ljava/lang/String;")?
                        .into_raw(),
                );

                self.m_read_image_tile = Some(
                    env.get_method_id(
                        class_ref,
                        "readImageTile",
                        "(Ljava/nio/ByteBuffer;IIIIIIIII)V",
                    )?
                    .into_raw(),
                );
            }
        };
        // 'env' is dropped here, so the borrow on 'jvm' is released.

        // In Rust's jni crate, we often call methods by name/sig directly
        // or cache MethodIDs if performance is critical.
        info!("JVM Initialized and BioFormatsWrapper loaded!");
        self.jvm = Some(jvm);
        Ok(())
    }

    /// Set the paths for JNI
    ///
    /// # Arguments
    ///
    /// - `&self` (`undefined`) - Describe this parameter.
    ///
    /// # Returns
    ///
    /// - `(PathBuf, PathBuf)` - Java home path and java lib path
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::...;
    ///
    /// let _ = set_path();
    /// ```
    fn set_path(&self) -> (PathBuf, PathBuf) {
        let (java_home, jvm_lib_path) = if cfg!(target_os = "windows") {
            let home = PathBuf::from("java/jre_win");
            (home.clone(), home.join("bin/server/jvm.dll"))
        } else if cfg!(target_os = "macos") {
            // Placeholder for getAppContentsPath() logic
            let home = PathBuf::from("Contents/Java/jre_macos_arm");
            (home.clone(), home.join("lib/server/libjvm.dylib"))
        } else {
            let home = PathBuf::from("java/jre_linux");
            (home.clone(), home.join("lib/amd64/server/libjvm.so"))
        };

        let java_bin = java_home.join("bin");

        // Set JAVA_HOME
        unsafe {
            env::set_var("JAVA_HOME", &java_home);

            // Update PATH
            if let Ok(current_path) = env::var("PATH") {
                let new_path = format!(
                    "{}{}{}",
                    java_bin.display(),
                    if cfg!(windows) { ";" } else { ":" },
                    current_path
                );
                env::set_var("PATH", new_path);
            }
        }
        (java_home, jvm_lib_path)
    }
}

/// Drops the global refs of the JNI
///
/// # Arguments
///
/// - `&mut self` (`undefined`) - Self
///
/// # Examples
///
/// ```
/// use crate::...;
///
/// let _ = drop();
/// ```
impl Drop for JavaWrapper {
    fn drop(&mut self) {
        // 1. Take the class reference out of the Option
        let class_ref = self.m_bioformats_class.take();
        // 2. Only proceed if we actually have a JVM and a class reference to clean up
        if let (Some(jvm), Some(class)) = (&self.jvm, class_ref) {
            // 3. Attach the thread to provide a JNIEnv for the cleanup
            if let Ok(_env) = jvm.attach_current_thread() {
                // 4. Explicitly drop the GlobalRef while _env is alive
                drop(class);
                // After this line, the JNI DeleteGlobalRef has been called safely.
            }
            // _env drops here, detaching the thread
        }
    }
}
