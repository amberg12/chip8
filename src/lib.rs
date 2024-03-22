pub mod io_traits;

use std::{cmp::min, time::Instant};

use crate::io_traits::{IODevice, KeyState};
use rand::prelude::*;

pub use crate::io_traits as io;

#[derive(Default)]
struct Registers {
    v0: u8,
    v1: u8,
    v2: u8,
    v3: u8,
    v4: u8,
    v5: u8,
    v6: u8,
    v7: u8,
    v8: u8,
    v9: u8,
    va: u8,
    vb: u8,
    vc: u8,
    vd: u8,
    ve: u8,
    vf: u8,
    pc: u16,
    i: u16,
    sound_timer: u8,
    delay_timer: u8,
}

impl Registers {
    /// Gets the vx register.
    fn get(&self, x: u8) -> u8 {
        match x {
            0x0 => self.v0,
            0x1 => self.v1,
            0x2 => self.v2,
            0x3 => self.v3,
            0x4 => self.v4,
            0x5 => self.v5,
            0x6 => self.v6,
            0x7 => self.v7,
            0x8 => self.v8,
            0x9 => self.v9,
            0xA => self.va,
            0xB => self.vb,
            0xC => self.vc,
            0xD => self.vd,
            0xE => self.ve,
            0xF => self.vf,
            _ => unreachable!(),
        }
    }

    fn set(&mut self, x: u8, val: u8) {
        match x {
            0x0 => self.v0 = val,
            0x1 => self.v1 = val,
            0x2 => self.v2 = val,
            0x3 => self.v3 = val,
            0x4 => self.v4 = val,
            0x5 => self.v5 = val,
            0x6 => self.v6 = val,
            0x7 => self.v7 = val,
            0x8 => self.v8 = val,
            0x9 => self.v9 = val,
            0xA => self.va = val,
            0xB => self.vb = val,
            0xC => self.vc = val,
            0xD => self.vd = val,
            0xE => self.ve = val,
            0xF => self.vf = val,
            _ => unreachable!(),
        }
    }
}

union Instruction {
    byte: (u8, u8),
    word: u16,
}

impl Instruction {
    fn from_bytes(b1: u8, b2: u8) -> Self {
        Self { byte: (b2, b1) }
    }

    fn word(&self) -> u16 {
        // Safety: byte layout of u16 and (u8, u8) are compatable.
        unsafe { self.word }
    }

    fn opcode(&self) -> u8 {
        // Safety: byte layout of u16 and (u8, u8) are compatable.
        (unsafe { self.word } >> 12)
            .try_into()
            .expect("Data should be pushed into 4 rightmost bits.")
    }

    fn x(&self) -> u8 {
        // Safety: byte layout of u16 and (u8, u8) are compatable.
        ((unsafe { self.word } & 0b0000111100000000) >> 8)
            .try_into()
            .expect("Data should be pushed into 4 rightmost bits.")
    }

    fn y(&self) -> u8 {
        // Safety: byte layout of u16 and (u8, u8) are compatable.
        ((unsafe { self.word } & 0b0000000011110000) >> 4)
            .try_into()
            .expect("Data should be pushed into 4 rightmost bits.")
    }

    fn n(&self) -> u8 {
        // Safety: byte layout of u16 and (u8, u8) are compatable.
        (unsafe { self.word } & 0b0000000000001111)
            .try_into()
            .expect("Data should be pushed into 4 rightmost bits.")
    }

    fn nn(&self) -> u8 {
        // Safety: byte layout of u16 and (u8, u8) are compatable.
        (unsafe { self.word } & 0b0000000011111111)
            .try_into()
            .expect("Data should be pushed into 4 rightmost bits.")
    }

    fn nnn(&self) -> u16 {
        // Safety: byte layout of u16 and (u8, u8) are compatable.
        (unsafe { self.word } & 0b0000111111111111)
            .try_into()
            .expect("Data should be pushed into 4 rightmost bits.")
    }
}

const BNNN_OLD: fn(&mut Chip8, &Instruction) -> () = |this_self, current_instruction| {
    this_self.registers.pc = current_instruction.nnn() + this_self.registers.v0 as u16;
};

const BNNN_MODERN: fn(&mut Chip8, &Instruction) -> () = |this_self, current_instruction| {
    this_self.registers.pc =
        current_instruction.nnn() + this_self.registers.get(current_instruction.x()) as u16;
};

const R8XY6_OLD: fn(&mut Chip8, &Instruction) -> () = |this_self, current_instruction| {
    let val = this_self.registers.get(current_instruction.y());
    this_self.registers.set(current_instruction.y(), val >> 1);
    this_self.registers.vf = val & 0b00000001;
};

const R8XY6_MODERN: fn(&mut Chip8, &Instruction) -> () = |this_self, current_instruction| {
    let val = this_self.registers.get(current_instruction.x());
    this_self.registers.set(current_instruction.x(), val >> 1);
    this_self.registers.vf = val & 0b00000001;
};

const R8XYE_OLD: fn(&mut Chip8, &Instruction) -> () = |this_self, current_instruction| {
    let val = this_self.registers.get(current_instruction.y());
    this_self.registers.set(current_instruction.y(), val << 1);
    this_self.registers.vf = (val & 0b10000000) >> 7;
};

const R8XYE_MODERN: fn(&mut Chip8, &Instruction) -> () = |this_self, current_instruction| {
    let val = this_self.registers.get(current_instruction.x());
    this_self.registers.set(current_instruction.x(), val << 1);
    this_self.registers.vf = (val & 0b10000000) >> 7;
};

const FX55_OLD: fn(&mut Chip8, &Instruction) -> () = |this_self, current_instruction| {
    for register in 0..=current_instruction.x() {
        this_self.memory[this_self.registers.i as usize] = this_self.registers.get(register);
        this_self.registers.i += 1;
    }
    this_self.registers.i += 1;
};

const FX55_MODERN: fn(&mut Chip8, &Instruction) -> () = |this_self, current_instruction| {
    for register in 0..=current_instruction.x() {
        this_self.memory[this_self.registers.i as usize + register as usize] =
            this_self.registers.get(register);
    }
};

const FX65_OLD: fn(&mut Chip8, &Instruction) -> () = |this_self, current_instruction| {
    for register in 0..=current_instruction.x() {
        this_self
            .registers
            .set(register, this_self.memory[this_self.registers.i as usize]);
        this_self.registers.i += 1;
    }
    this_self.registers.i += 1;
};

const FX65_MODERN: fn(&mut Chip8, &Instruction) -> () = |this_self, current_instruction| {
    for register in 0..=current_instruction.x() {
        this_self.registers.set(
            register,
            this_self.memory[this_self.registers.i as usize + register as usize],
        );
    }
};

pub struct Chip8Builder {
    program: Option<Vec<u8>>,
    io_device: Option<Box<dyn IODevice>>,
    bnnn: fn(&mut Chip8, &Instruction) -> (),
    r8xy6: fn(&mut Chip8, &Instruction) -> (),
    r8xye: fn(&mut Chip8, &Instruction) -> (),
    fx55: fn(&mut Chip8, &Instruction) -> (),
    fx65: fn(&mut Chip8, &Instruction) -> (),
}

impl Chip8Builder {
    fn new() -> Self {
        Self {
            program: None,
            io_device: None,
            bnnn: BNNN_OLD,
            r8xy6: R8XY6_OLD,
            r8xye: R8XYE_OLD,
            fx55: FX55_OLD,
            fx65: FX65_OLD,
        }
    }

    pub fn with_modern_behaviour(mut self) -> Self {
        self.bnnn = BNNN_MODERN;
        self.r8xy6 = R8XY6_MODERN;
        self.r8xye = R8XYE_MODERN;
        self.fx55 = FX55_MODERN;
        self.fx65 = FX65_MODERN;
        self
    }

    pub fn with_program(mut self, program: Vec<u8>) -> Self {
        self.program = Some(program);
        self
    }

    pub fn with_io_device(mut self, io_device: Box<dyn IODevice>) -> Self {
        self.io_device = Some(io_device);
        self
    }

    /// ## If not set:
    ///
    /// - `program` will panic
    /// - `io_device` will panic
    /// - `bnnn` will use the COSMAC VIP implementation
    /// - `fx55` will use the COSMAC VIP implementation
    /// - `fx65` will use the COSMAC VIP implementation
    /// - `8xy6` will use the COSMAC VIP implementation
    /// - `8xyE` will use the COSMAC VIP implementation
    pub fn build(self) -> Chip8 {
        assert!(self.program.is_some());
        assert!(self.io_device.is_some());

        Chip8::new_internal(
            self.program.unwrap(),
            self.io_device.unwrap(),
            self.bnnn,
            self.r8xy6,
            self.r8xye,
            self.fx55,
            self.fx65,
        )
    }
}

pub struct Chip8 {
    memory: [u8; 0x1000],
    registers: Registers,
    stack: Vec<u16>,
    display: [[u8; 64]; 32],
    io_device: Box<dyn IODevice>,
    clock_timer: Instant,
    bnnn: fn(&mut Chip8, &Instruction) -> (),
    r8xy6: fn(&mut Chip8, &Instruction) -> (),
    r8xye: fn(&mut Chip8, &Instruction) -> (),
    fx55: fn(&mut Chip8, &Instruction) -> (),
    fx65: fn(&mut Chip8, &Instruction) -> (),
}

impl Chip8 {
    pub fn init() -> Chip8Builder {
        Chip8Builder::new()
    }

    fn new_internal(
        program: Vec<u8>,
        io_device: Box<dyn IODevice>,
        bnnn: fn(&mut Chip8, &Instruction),
        r8xy6: fn(&mut Chip8, &Instruction),
        r8xye: fn(&mut Chip8, &Instruction),
        fx55: fn(&mut Chip8, &Instruction),
        fx65: fn(&mut Chip8, &Instruction),
    ) -> Self {
        let mut memory = [0u8; 0x1000];

        for (read_ptr, byte) in program.into_iter().enumerate() {
            *memory
                .get_mut(read_ptr + 0x200)
                .expect("Program should not exceed the size of the emulated ram.") = byte;
        }

        for (offset, byte) in vec![
            0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
            0x20, 0x60, 0x20, 0x20, 0x70, // 1
            0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
            0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
            0x90, 0x90, 0xF0, 0x10, 0x10, // 4
            0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
            0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
            0xF0, 0x10, 0x20, 0x40, 0x40, // 7
            0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
            0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
            0xF0, 0x90, 0xF0, 0x90, 0x90, // A
            0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
            0xF0, 0x80, 0x80, 0x80, 0xF0, // C
            0xE0, 0x90, 0x90, 0x90, 0xE0, // D
            0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
            0xF0, 0x80, 0xF0, 0x80, 0x80, // F
        ]
        .into_iter()
        .enumerate()
        {
            memory[0x50 + offset] = byte;
        }

        Self {
            memory,
            registers: Registers {
                pc: 0x0200,
                ..Default::default()
            },
            stack: Default::default(),
            display: [[0; 64]; 32],
            io_device,
            clock_timer: Instant::now(),
            bnnn,
            r8xy6,
            r8xye,
            fx55,
            fx65,
        }
    }

    pub fn fde_cycle(&mut self) -> u32 {
        // Poll system
        let key_states = self.io_device.poll_keys();

        // Fetch
        let (b1, b2) = (
            self.memory[self.registers.pc as usize],
            self.memory[self.registers.pc as usize + 1],
        );

        let current_instruction = Instruction::from_bytes(b1, b2);
        self.registers.pc += 2;

        if self.clock_timer.elapsed().as_millis() > 16 {
            self.clock_timer = Instant::now();
            self.registers.delay_timer = min(self.registers.delay_timer.wrapping_sub(1), 0);
        }

        self.decode_execute(current_instruction, key_states);
        self.io_device.display(self.display);
        1
    }

    fn decode_execute(&mut self, current_instruction: Instruction, key_states: [KeyState; 16]) {
        match (
            current_instruction.opcode(),
            current_instruction.x(),
            current_instruction.y(),
            current_instruction.n(),
        ) {
            // Clear Screen.
            (0x0, 0x0, 0xE, 0x0) => {
                self.display = [[0; 64]; 32];
            }
            (0x0, 0x0, 0xE, 0xE) => {
                self.registers.pc = self
                    .stack
                    .pop()
                    .expect("Programmer should have placed data on the stack");
            }
            // Execute machine language routine
            (0x0, _, _, _) => {
                // Not applicable to the scope of this emulator.
            }
            (0x1, _, _, _) => {
                self.registers.pc = current_instruction.nnn();
            }
            // Call subroutine at nnn
            (0x2, _, _, _) => {
                self.stack.push(self.registers.pc);
                self.registers.pc = current_instruction.nnn();
            }
            (0x3, _, _, _) => {
                if self.registers.get(current_instruction.x()) == current_instruction.nn() {
                    self.registers.pc += 2;
                }
            }
            (0x4, _, _, _) => {
                if self.registers.get(current_instruction.x()) != current_instruction.nn() {
                    self.registers.pc += 2;
                }
            }
            (0x5, _, _, _) => {
                if self.registers.get(current_instruction.x())
                    == self.registers.get(current_instruction.y())
                {
                    self.registers.pc += 2;
                }
            }
            (0x6, _, _, _) => {
                self.registers
                    .set(current_instruction.x(), current_instruction.nn());
            }
            (0x7, _, _, _) => {
                self.registers.set(
                    current_instruction.x(),
                    self.registers
                        .get(current_instruction.x())
                        .wrapping_add(current_instruction.nn()),
                );
            }
            (0x8, _, _, 0x0) => {
                self.registers.set(
                    current_instruction.x(),
                    self.registers.get(current_instruction.y()),
                );
            }
            (0x8, _, _, 0x1) => {
                self.registers.set(
                    current_instruction.x(),
                    self.registers.get(current_instruction.x())
                        | self.registers.get(current_instruction.y()),
                );
            }
            (0x8, _, _, 0x2) => {
                self.registers.set(
                    current_instruction.x(),
                    self.registers.get(current_instruction.x())
                        & self.registers.get(current_instruction.y()),
                );
            }
            (0x8, _, _, 0x3) => {
                self.registers.set(
                    current_instruction.x(),
                    self.registers.get(current_instruction.x())
                        ^ self.registers.get(current_instruction.y()),
                );
            }
            (0x8, _, _, 0x4) => {
                let a = self.registers.get(current_instruction.x());
                let b = self.registers.get(current_instruction.y());
                self.registers
                    .set(current_instruction.x(), a.wrapping_add(b));
                self.registers.vf = if a as u16 + b as u16 > u8::MAX as u16 {
                    1
                } else {
                    0
                };
            }
            (0x8, _, _, 0x5) => {
                let a = self.registers.get(current_instruction.x());
                let b = self.registers.get(current_instruction.y());
                self.registers
                    .set(current_instruction.x(), a.wrapping_sub(b));
                self.registers.vf = if b > a { 0 } else { 1 };
            }
            (0x8, _, _, 0x6) => {
                let r8xy6 = self.r8xy6.clone();
                (r8xy6)(self, &current_instruction);
            }
            (0x8, _, _, 0x7) => {
                let b = self.registers.get(current_instruction.x());
                let a = self.registers.get(current_instruction.y());
                self.registers
                    .set(current_instruction.x(), a.wrapping_sub(b));
                self.registers.vf = if b > a { 0 } else { 1 };
            }
            (0x8, _, _, 0xE) => {
                let r8xye = self.r8xye.clone();
                (r8xye)(self, &current_instruction);
            }
            (0x9, _, _, _) => {
                if self.registers.get(current_instruction.x())
                    != self.registers.get(current_instruction.y())
                {
                    self.registers.pc += 2;
                }
            }
            (0xA, _, _, _) => {
                self.registers.i = current_instruction.nnn();
            }
            (0xB, _, _, _) => {
                let bnnn = self.bnnn.clone();
                bnnn(self, &current_instruction);
            }
            (0xC, _, _, _) => {
                let random = random::<u8>();
                self.registers
                    .set(current_instruction.x(), random & current_instruction.nn())
            }
            (0xD, _, _, _) => {
                let (x, y) = (
                    self.registers.get(current_instruction.x()) % 64,
                    self.registers.get(current_instruction.y()) % 64,
                );

                for (row, y_coord) in (y..(y + current_instruction.n())).enumerate() {
                    let row_bits = self.memory[row + self.registers.i as usize];

                    for (bit, x_coord) in (x..(x + 8)).enumerate() {
                        let bit = bit as u8;

                        if x_coord > 63 || y_coord > 31 {
                            continue;
                        }

                        if row_bits & (0b10000000 >> bit) > 0 {
                            *self
                                .display
                                .get_mut(y_coord as usize)
                                .expect("Should be valid")
                                .get_mut(x_coord as usize)
                                .expect("Should be valid") ^= 1;
                        } else {
                            //*self
                            //    .display
                            //    .get_mut(y_coord as usize)
                            //    .expect("Should be valid")
                            //    .get_mut(x_coord as usize)
                            //    .expect("Should be valid") = 0;
                        }
                    }
                }
            }
            (0xE, _, 0x9, 0xE) => {
                if key_states[self.registers.get(current_instruction.x()) as usize] != KeyState::Up
                {
                    self.registers.pc += 2;
                }
            }
            (0xE, _, 0xA, 0x1) => {
                if key_states[self.registers.get(current_instruction.x()) as usize] == KeyState::Up
                {
                    self.registers.pc += 2;
                }
            }
            (0xF, _, 0x0, 0x7) => {
                self.registers
                    .set(current_instruction.x(), self.registers.delay_timer);
            }
            (0xF, _, 0x0, 0xA) => {
                if key_states
                    .into_iter()
                    .map(|x| {
                        if let KeyState::Up = x {
                            0
                        } else if let KeyState::Pressed = x {
                            0
                        } else {
                            1
                        }
                    })
                    .sum::<u8>()
                    == 0
                {
                    self.registers.pc -= 2;
                } else {
                    self.registers.set(
                        current_instruction.x(),
                        key_states
                            .into_iter()
                            .enumerate()
                            .find(|(idx, x)| *x == KeyState::Released)
                            .unwrap()
                            .0 as u8,
                    );
                }
            }
            (0xF, _, 0x1, 0x5) => {
                self.registers.delay_timer = self.registers.get(current_instruction.x());
            }
            (0xF, _, 0x1, 0x8) => {
                self.registers.sound_timer = self.registers.get(current_instruction.x());
            }
            (0xF, _, 0x1, 0xE) => {
                self.registers.i += self.registers.get(current_instruction.x()) as u16;
            }
            (0xF, _, 0x2, 0x9) => {
                self.registers.i = current_instruction.x() as u16 + 0x50;
            }
            (0xF, _, 0x3, 0x3) => {
                let val: String = self.registers.get(current_instruction.x()).to_string();
                let decimals = match val.chars().count() {
                    1 => format!("00{}", val),
                    2 => format!("0{}", val),
                    3 => format!("{}", val),
                    _ => val.chars().take(3).collect::<String>(),
                };
                self.memory[self.registers.i as usize] =
                    decimals.chars().take(1).last().unwrap() as u8 & 0b00001111;
                self.memory[self.registers.i as usize + 1] =
                    decimals.chars().take(2).last().unwrap() as u8 & 0b00001111;
                self.memory[self.registers.i as usize + 2] =
                    decimals.chars().take(3).last().unwrap() as u8 & 0b00001111;
            }
            (0xF, _, 0x5, 0x5) => {
                let fx55 = self.fx55.clone();
                fx55(self, &current_instruction);
            }
            (0xF, _, 0x6, 0x5) => {
                let fx65 = self.fx65.clone();
                fx65(self, &current_instruction);
            }
            _ => unreachable!("Instruction: {:x}", current_instruction.word()),
        }
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    struct DummyIO {
        key_state: [KeyState; 16],
    }

    impl DummyIO {
        fn set_key_state(&mut self, new: [KeyState; 16]) {
            self.key_state = new;
        }

        fn new() -> Self {
            Self {
                key_state: [KeyState::Up; 16],
            }
        }
    }

    impl IODevice for DummyIO {
        fn display(&mut self, _display: [[u8; 64]; 32]) {
            // We do not care for having a graphic display in these unit tests.
        }

        fn poll_keys(&mut self) -> [KeyState; 16] {
            self.key_state.clone()
        }
    }

    #[test]
    fn op_00E0() {
        let dummy = DummyIO::new();
        let mut chip8 = Chip8::init()
            .with_program(vec![
                0xA0, 0x50, // Set index register to font start
                0xD0, 0x05, // Draw font letter loaded
                0x00, 0xE0, // Clear Screen
            ])
            .with_io_device(Box::new(dummy))
            .build();

        chip8.fde_cycle();
        chip8.fde_cycle();
        chip8.fde_cycle();

        assert_eq!(chip8.display, [[0; 64]; 32]);
    }

    #[test]
    fn op_1NNN() {
        let dummy = DummyIO::new();
        let mut chip8 = Chip8::init()
            .with_program(vec![
                0x15, 0x00, // Jump to 0x500
                0x11, 0x23, // Jump to 0x123
                0x14, 0x67, // Jump to 0x467
            ])
            .with_io_device(Box::new(dummy))
            .build();

        chip8.fde_cycle();
        assert_eq!(chip8.registers.pc, 0x0500);
        // Jumping means that the other jumps should not be ran, and the chip-8 just runs through
        // empty memory.
        chip8.fde_cycle();
        assert_eq!(chip8.registers.pc, 0x0502);
        chip8.fde_cycle();
        assert_eq!(chip8.registers.pc, 0x0504);
    }

    #[test]
    fn op_3XNN() {
        let dummy = DummyIO::new();
        let mut chip8 = Chip8::init()
            .with_program(vec![
                0x60, 0x01, // Set v0 to 1
                0x30, 0x01, // Skip if v0 == 01
                0x60, 0x00, // Set v0 to 0
                0x31, 0x01, // Skip if v1 == 01
                0x61, 0x11, // Set v1 to 0x11
                0x00, 0x00,
            ])
            .with_io_device(Box::new(dummy))
            .build();

        chip8.fde_cycle();
        chip8.fde_cycle();
        chip8.fde_cycle();
        chip8.fde_cycle();
        assert_eq!(chip8.registers.v0, 0x1);
        assert_eq!(chip8.registers.v1, 0x11)
    }

    #[test]
    fn op_4XNN() {
        let dummy = DummyIO::new();
        let mut chip8 = Chip8::init()
            .with_program(vec![
                0x60, 0x01, // Set v0 to 1
                0x40, 0x01, // Skip if v0 != 01
                0x60, 0x00, // Set v0 to 0
                0x41, 0x01, // Skip if v1 != 01
                0x61, 0x11, // Set v1 to 0x11
                0x00, 0x00,
            ])
            .with_io_device(Box::new(dummy))
            .build();

        chip8.fde_cycle();
        chip8.fde_cycle();
        chip8.fde_cycle();
        chip8.fde_cycle();
        assert_eq!(chip8.registers.v0, 0x0);
        assert_eq!(chip8.registers.v1, 0x0)
    }
    #[test]
    fn op_6XNN() {
        let dummy = DummyIO::new();
        let mut chip8 = Chip8::init()
            .with_program(vec![
                0x60, 0x00, // Set v0 to 00
                0x61, 0x11, // Set v1 to 11
                0x62, 0x22, // Set v2 to 22
            ])
            .with_io_device(Box::new(dummy))
            .build();

        chip8.fde_cycle();
        assert_eq!(chip8.registers.v0, 0x00);
        assert_eq!(chip8.registers.v1, 0x00);
        assert_eq!(chip8.registers.v2, 0x00);

        chip8.fde_cycle();
        assert_eq!(chip8.registers.v0, 0x00);
        assert_eq!(chip8.registers.v1, 0x11);
        assert_eq!(chip8.registers.v2, 0x00);

        chip8.fde_cycle();
        assert_eq!(chip8.registers.v0, 0x00);
        assert_eq!(chip8.registers.v1, 0x11);
        assert_eq!(chip8.registers.v2, 0x22);
    }

    #[test]
    fn op_7XNN() {
        let dummy = DummyIO::new();
        let mut chip8 = Chip8::init()
            .with_program(vec![
                0x60, 0x00, // Set v0 to 00
                0x70, 0xFF, // Add 255 to v0
                0x70, 0x22, // Set v2 to 22
            ])
            .with_io_device(Box::new(dummy))
            .build();

        chip8.fde_cycle();
        chip8.fde_cycle();
        assert_eq!(chip8.registers.v0, 0xFF);
        chip8.fde_cycle();
        // Account for the 255 + 1 = 0
        assert_eq!(chip8.registers.v0, 0x21);
    }

    #[test]
    fn op_8XY0() {
        let dummy = DummyIO::new();

        let mut chip8 = Chip8::init()
            .with_program(vec![
                0x60, 0x50, // Set v0 to 0x50
                0x81, 0x00, // Set v1 to v0
            ])
            .with_io_device(Box::new(dummy))
            .build();

        chip8.fde_cycle();
        chip8.fde_cycle();
        assert_eq!(chip8.registers.v0, chip8.registers.v1);
    }

    #[test]
    fn op_8XY1() {
        let dummy = DummyIO::new();
        let mut chip8 = Chip8::init()
            .with_program(vec![
                0x60, 0b11001010, // Set v0 to 0x50
                0x61, 0b11110000, // Set v0 to 0x50
                0x80, 0x11, // Set v0 to v0 | v1
            ])
            .with_io_device(Box::new(dummy))
            .build();

        chip8.fde_cycle();
        chip8.fde_cycle();
        chip8.fde_cycle();
        assert_eq!(chip8.registers.v0, 0b11111010);
    }

    #[test]
    fn op_ANNN() {
        let dummy = DummyIO::new();
        let mut chip8 = Chip8::init()
            .with_program(vec![
                0xA0, 0x00, // Set I to 00
                0xA1, 0x11, // Set I to 11
                0xA2, 0x22, // Set I to 22
            ])
            .with_io_device(Box::new(dummy))
            .build();

        chip8.fde_cycle();
        assert_eq!(chip8.registers.i, 0x0000);
        chip8.fde_cycle();
        assert_eq!(chip8.registers.i, 0x0111);
        chip8.fde_cycle();
        assert_eq!(chip8.registers.i, 0x0222);
    }

    #[test]
    fn op_DXYN() {
        let dummy = DummyIO::new();
        let mut chip8 = Chip8::init()
            .with_program(vec![
                0x60, 0x3F, // Set v0 to 63
                0x61, 0x00, // Set v1 to 0
                0xA0, 0x5F, // Set I to 0x5F (load the 3)
                0xD0, 0x15, // Draw 0
            ])
            .with_io_device(Box::new(dummy))
            .build();

        chip8.fde_cycle();
        chip8.fde_cycle();
        chip8.fde_cycle();
        chip8.fde_cycle();
        dbg!(chip8.display[0]);
        dbg!(chip8.display[1]);
        dbg!(chip8.display[2]);
        dbg!(chip8.display[3]);
        dbg!(chip8.display[4]);
        assert_eq!(chip8.display[0][63], 1);
        assert_eq!(chip8.display[1][63], 0);
        assert_eq!(chip8.display[2][63], 1);
        assert_eq!(chip8.display[3][63], 0);
        assert_eq!(chip8.display[4][63], 1);
    }
}
