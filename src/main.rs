use chip8::io::*;
use chip8::Chip8;

use macroquad::prelude::*;

struct MacroquadChip8;

#[macro_use]
extern crate clap;
use clap::Parser;

const PX_SIZE: f32 = 10.0;

impl IODevice for MacroquadChip8 {
    fn display(&mut self, display: [[u8; 64]; 32]) {
        for (y, row) in display.into_iter().enumerate() {
            for (x, pixel) in row.into_iter().enumerate() {
                if pixel > 0 {
                    draw_rectangle(
                        x as f32 * PX_SIZE,
                        y as f32 * PX_SIZE,
                        PX_SIZE,
                        PX_SIZE,
                        WHITE,
                    )
                }
            }
        }
    }

    fn poll_keys(&mut self) -> [KeyState; 16] {
        let mut map = [KeyState::Up; 16];

        // These are ordered based on their nibble representation.
        let keycodes = vec![
            KeyCode::X,
            KeyCode::Key1,
            KeyCode::Key2,
            KeyCode::Key3,
            KeyCode::Q,
            KeyCode::W,
            KeyCode::E,
            KeyCode::A,
            KeyCode::S,
            KeyCode::D,
            KeyCode::Z,
            KeyCode::C,
            KeyCode::Key4,
            KeyCode::R,
            KeyCode::F,
            KeyCode::V,
        ];

        for (i, key_code) in keycodes.into_iter().enumerate() {
            map[i] = if is_key_down(key_code) {
                KeyState::Pressed
            } else if is_key_pressed(key_code) {
                KeyState::Pressed
            } else if is_key_released(key_code) {
                KeyState::Released
            } else {
                KeyState::Up
            }
        }
        map
    }
}

#[derive(Parser)]
struct CLI {
    path: std::path::PathBuf,
}

#[macroquad::main("Chip 8")]
async fn main() {
    let args = CLI::parse();

    let io_device = Box::new(MacroquadChip8);
    let program = std::fs::read(args.path).unwrap();

    let mut chip8 = Chip8::init()
        .with_program(program.try_into().unwrap())
        .with_io_device(io_device)
        .with_modern_behaviour()
        .build();

    loop {
        chip8.fde_cycle();
        next_frame().await;
    }
}
