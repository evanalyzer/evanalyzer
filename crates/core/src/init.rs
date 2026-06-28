use crate::init_java_wrapper;
use crate::resources::recommended_jvm_heap_bytes;
use std::error::Error;

pub struct CoreConfig {
    /// Java heap size in bytes
    pub jvm_heap_size: u64,
}

impl Default for CoreConfig {
    /// Sizes the JVM heap from currently available system RAM (see
    /// [`recommended_jvm_heap_bytes`]) rather than a fixed constant, so a
    /// constrained machine doesn't over-commit and a large one doesn't
    /// under-use what it has.
    fn default() -> Self {
        Self {
            jvm_heap_size: recommended_jvm_heap_bytes(),
        }
    }
}

pub fn init(config: CoreConfig) -> Result<(), Box<dyn Error>> {
    init_java_wrapper(config.jvm_heap_size)?;
    Ok(())
}
