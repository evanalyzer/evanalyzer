use crate::resources::recommended_jvm_heap_bytes;
use evanalyzer_cfg::core_types::InternalErrors;
use jni::objects::JClass;
use jni::refs::Global;
use jni::sys::jmethodID;
use jni::{InitArgsBuilder, JNIVersion, JavaVM, jni_sig, jni_str};
use log::{info, warn};
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock}; // Use OnceCell from the crate, not std

pub struct JavaWrapper {
    pub jvm: Option<JavaVM>,
    pub m_bioformats_class: Option<Global<JClass<'static>>>,
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

/// Returns the global `JavaWrapper`, initializing it first if this is the
/// first call. The JVM is started lazily (on first actual image read) rather
/// than eagerly at process startup, so launching the app - or running a CLI
/// command that never touches an image - doesn't pay the JVM/Bio-Formats
/// classloading cost up front. `init_java_wrapper` is idempotent and
/// safe to call from multiple threads (see its own doc comment), so this
/// is also what a caller that wants to warm the JVM up in the background
/// (e.g. the GUI, right after showing its window) should call directly.
pub fn ensure_java_wrapper() -> Result<&'static JavaWrapper, InternalErrors> {
    if JAVA_WRAPPER.get().is_none() {
        init_java_wrapper(recommended_jvm_heap_bytes())
            .map_err(|e| InternalErrors::JvmError(e.to_string()))?;
    }
    JAVA_WRAPPER
        .get()
        .ok_or_else(|| InternalErrors::JvmError("JVM failed to initialize".into()))
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
            .version(JNIVersion::V1_8)
            .option(classpath)
            .option(ram_arg)
            .option(headless_arg)
            .build()?;

        let jvm = JavaVM::new(jvm_args)?;

        // jni 0.22's `attach_current_thread` only ever hands out an `Env`
        // reference within a closure (never an owned guard tied to `jvm`),
        // so every JNI call that needs it happens inside this closure; the
        // resolved class ref and cached method IDs are returned out of it.
        let (global_cls, constructor, close, get_image_properties, read_image_tile) = jvm
            .attach_current_thread(|env| -> jni::errors::Result<_> {
                let class_name = jni_str!("BioFormatsWrapper");
                let local_cls = env.find_class(class_name)?;

                // 1.  Cache the Method IDs
                // C++: mConstructor = myGlobEnv->GetMethodID(mBioformatsClass, "<init>", "(Ljava/lang/String;)V");
                // We use .into_raw() to store them as simple pointers
                let constructor = env
                    .get_method_id(
                        &local_cls,
                        jni_str!("<init>"),
                        jni_sig!("(Ljava/lang/String;Z)V"),
                    )?
                    .into_raw();

                let close = env
                    .get_method_id(&local_cls, jni_str!("close"), jni_sig!("()V"))?
                    .into_raw();

                let get_image_properties = env
                    .get_method_id(
                        &local_cls,
                        jni_str!("getImageProperties"),
                        jni_sig!("()Ljava/lang/String;"),
                    )?
                    .into_raw();

                let read_image_tile = env
                    .get_method_id(
                        &local_cls,
                        jni_str!("readImageTile"),
                        jni_sig!("(Ljava/nio/ByteBuffer;IIIIIIIII)V"),
                    )?
                    .into_raw();

                let global_cls = env.new_global_ref(local_cls)?;

                Ok((global_cls, constructor, close, get_image_properties, read_image_tile))
            })?;

        self.m_bioformats_class = Some(global_cls);
        self.m_constructor = Some(constructor);
        self.m_close = Some(close);
        self.m_get_image_properties = Some(get_image_properties);
        self.m_read_image_tile = Some(read_image_tile);

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

// No `Drop` impl is needed here: since jni 0.22, `Global`'s own `Drop` impl
// attaches the current thread internally (via `JavaVM::singleton()`) to call
// `DeleteGlobalRef`, so `m_bioformats_class` cleans itself up when this
// struct's fields are dropped in the normal way.

#[cfg(test)]
mod tests {
    use super::*;

    /// A `JavaWrapper` with no live JVM/JNI state, built via struct literal
    /// (all fields are `pub`) rather than `new()` - this exercises `set_path`'s
    /// pure path-selection logic without paying for (or requiring) a real JVM.
    fn bare_wrapper() -> JavaWrapper {
        JavaWrapper {
            jvm: None,
            m_bioformats_class: None,
            m_constructor: None,
            m_close: None,
            m_get_image_properties: None,
            m_read_image_tile: None,
            m_reserved_ram: 0,
        }
    }

    #[test]
    fn set_path_returns_the_platform_specific_java_home_and_lib_path() {
        let (java_home, jvm_lib_path) = bare_wrapper().set_path();

        if cfg!(target_os = "windows") {
            assert_eq!(java_home, PathBuf::from("java/jre_win"));
            assert_eq!(jvm_lib_path, java_home.join("bin/server/jvm.dll"));
        } else if cfg!(target_os = "macos") {
            assert_eq!(java_home, PathBuf::from("Contents/Java/jre_macos_arm"));
            assert_eq!(jvm_lib_path, java_home.join("lib/server/libjvm.dylib"));
        } else {
            assert_eq!(java_home, PathBuf::from("java/jre_linux"));
            assert_eq!(jvm_lib_path, java_home.join("lib/amd64/server/libjvm.so"));
        }
    }

    #[test]
    fn set_path_is_idempotent_across_repeated_calls() {
        // set_path prepends to the process-global PATH env var as a side
        // effect; calling it more than once (e.g. across several
        // ImageReader::new() calls) must keep returning the same computed
        // paths rather than e.g. accumulating a different value each time.
        let wrapper = bare_wrapper();
        let first = wrapper.set_path();
        let second = wrapper.set_path();
        assert_eq!(first, second);
    }
}
