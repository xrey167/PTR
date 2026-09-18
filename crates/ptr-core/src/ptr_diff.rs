#[derive(Clone, Debug)]
pub struct PtrDiffConfig {
    pub blocks: usize,
    pub canvas_slots: usize,
    pub denoise_steps: u32,
}
impl Default for PtrDiffConfig {
    fn default() -> Self {
        Self {
            blocks: 12,
            canvas_slots: 64,
            denoise_steps: 8,
        }
    }
}
