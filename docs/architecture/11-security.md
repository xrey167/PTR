# Security Model

Security has semantic, runtime and infrastructure layers. Untrusted context never promotes directly to `Known<T>`. Effects require capability and permission checks. Secrets and raw private evidence must be redacted from generic introspection. Artifact identity is content-addressed where possible; storage location is separate from semantic identity.
