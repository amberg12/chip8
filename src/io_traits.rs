pub trait IODevice {
    fn poll_keys(&mut self) -> [KeyState; 16];
    fn display(&mut self, display: [[u8; 64]; 32]);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum KeyState {
    Pressed,
    Held,
    Released,
    Up,
}
