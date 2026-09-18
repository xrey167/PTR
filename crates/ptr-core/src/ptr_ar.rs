#[derive(Clone, Debug)]
pub struct PtrArConfig {
    pub layers: usize,
    pub context: usize,
    pub latent_steps_max: u32,
}
impl Default for PtrArConfig {
    fn default() -> Self {
        Self {
            layers: 12,
            context: 8192,
            latent_steps_max: 8,
        }
    }
}
