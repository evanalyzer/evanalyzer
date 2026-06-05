use crate::init_java_wrapper;
use std::error::Error;

pub struct CoreConfig {
    /// Java heap size in bytes
    pub jvm_heap_size: u64,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            jvm_heap_size: 1_000_000_000,
        }
    }
}

pub fn init(config: CoreConfig) -> Result<(), Box<dyn Error>> {
    init_java_wrapper(config.jvm_heap_size)?;
    Ok(())
}
