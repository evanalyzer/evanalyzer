#[macro_export]
macro_rules! AssignObjectClass {
    ($val:expr) => {
        ObjectClass::Valid($val as u32)
    };
}

#[macro_export]
macro_rules! object_class_set {
    // Match a comma-separated list of expressions
    ($($val:expr),* $(,)?) => {
        {
            let mut temp_set = std::collections::HashSet::new();
            $(
                temp_set.insert($val);
            )*
            temp_set
        }
    };
}

#[macro_export]
macro_rules! object_class_set_from_u32 {
    // Branch for raw u32 numbers: oc_set![1, 2, 3]
    ( $($val:expr),* $(,)? ) => {
        {
            let mut temp_set = std::collections::HashSet::new();
            $(
                temp_set.insert(ObjectClass::Valid($val));
            )*
            temp_set
        }
    };
}
