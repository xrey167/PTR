macro_rules! impl_trace_value_from {
    ($variant:ident => $type:ty) => {
        impl From<$type> for $crate::TraceValue {
            fn from(value: $type) -> Self {
                Self::$variant(value)
            }
        }
    };
}

pub(crate) use impl_trace_value_from;

#[macro_export]
macro_rules! trace_event {
    ($level:expr, $name:expr $(, $key:expr => $value:expr )* $(,)?) => {{
        let event = $crate::TraceEvent::new($level, $name);
        $(
            let event = event.with_field($key, $value);
        )*
        event
    }};
}
