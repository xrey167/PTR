macro_rules! string_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }
    };
}

pub(crate) use string_newtype;

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
