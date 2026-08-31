use flate2::write::ZlibEncoder;
use flate2::Compression;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::process::Command;

use crate::CraftaxWorld;

pub type Rgb = (u8, u8, u8);
pub type RgbRows = Vec<Vec<Rgb>>;
pub type RgbFrame = (u32, u32, RgbRows);

pub const DEFAULT_RENDER_TILE_SIZE: u32 = 16;
pub const UNKNOWN_RGB: Rgb = (30, 34, 38);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RenderMode {
    #[default]
    Auto,
    Symbolic,
    Sprites,
}

pub fn render_rgb_frame_from_world(
    world: &CraftaxWorld,
    tile_size: u32,
    mode: RenderMode,
) -> RgbFrame {
    let resolved = match mode {
        RenderMode::Auto => {
            if crate::sprites::sprites_available() {
                RenderMode::Sprites
            } else {
                RenderMode::Symbolic
            }
        }
        other => other,
    };
    match resolved {
        RenderMode::Sprites => crate::sprites::render_rgb_frame_sprites_from_world(world, tile_size),
        RenderMode::Symbolic | RenderMode::Auto => {
            render_rgb_frame_symbolic_from_world(world, tile_size)
        }
    }
}

fn render_rgb_frame_symbolic_from_world(world: &CraftaxWorld, tile_size: u32) -> RgbFrame {
    let width = (world.width.max(1) as u32) * tile_size;
    let height = (world.height.max(1) as u32) * tile_size;
    let level = usize::try_from(world.player_level.max(0)).unwrap_or(0);
    let mut canvas: RgbRows = vec![vec![UNKNOWN_RGB; width as usize]; height as usize];

    for y in 0..world.height {
        for x in 0..world.width {
            let ux = usize::try_from(x).unwrap_or(0);
            let uy = usize::try_from(y).unwrap_or(0);
            let item = world.item_maps[level][uy][ux].as_str();
            let tile = if item != "none" {
                item
            } else {
                world.maps[level][uy][ux].as_str()
            };
            let rgb = if (x, y) == world.player_pos {
                (250, 204, 21)
            } else if world.entities.iter().any(|entity| {
                entity.mask && entity.level == world.player_level && entity.pos == (x, y)
            }) {
                (209, 213, 219)
            } else {
                symbolic_rgb_for_tile(tile)
            };
            fill_tile(&mut canvas, (x as u32) * tile_size, (y as u32) * tile_size, tile_size, rgb);
        }
    }

    (width, height, canvas)
}

fn symbolic_rgb_for_tile(tile: &str) -> Rgb {
    match tile {
        "grass" | "path" | "fire_grass" | "ice_grass" => (106, 170, 84),
        "sand" | "gravel" => (154, 143, 115),
        "water" => (59, 130, 196),
        "stone" => (119, 119, 119),
        "tree" | "fire_tree" | "ice_shrub" => (35, 107, 46),
        "coal" => (51, 51, 51),
        "iron" => (166, 124, 82),
        "diamond" => (94, 196, 255),
        "sapphire" => (71, 119, 217),
        "ruby" => (217, 79, 79),
        "chest" => (139, 90, 43),
        "crafting_table" => (161, 98, 7),
        "furnace" => (82, 82, 82),
        "plant" => (74, 222, 128),
        "ripe_plant" => (250, 204, 21),
        "ladder_down" | "ladder_up" | "ladder_down_blocked" => (249, 115, 22),
        "wall" => (39, 39, 42),
        "wall_moss" => (63, 63, 70),
        "darkness" => (2, 6, 23),
        "stalagmite" => (168, 162, 158),
        "lava" => (239, 68, 68),
        "fountain" => (56, 189, 248),
        "enchantment_table_fire" | "enchantment_table_ice" => (249, 115, 22),
        "necromancer" => (168, 85, 247),
        "necromancer_vulnerable" => (192, 132, 252),
        "grave" | "grave2" | "grave3" => (113, 113, 122),
        "torch" => (253, 224, 71),
        _ => (209, 213, 219),
    }
}

fn fill_tile(canvas: &mut RgbRows, dest_x: u32, dest_y: u32, tile_size: u32, rgb: Rgb) {
    let canvas_h = canvas.len() as u32;
    let canvas_w = canvas.first().map(|row| row.len() as u32).unwrap_or(0);
    for dy in 0..tile_size {
        for dx in 0..tile_size {
            let y = dest_y + dy;
            let x = dest_x + dx;
            if y < canvas_h && x < canvas_w {
                canvas[y as usize][x as usize] = rgb;
            }
        }
    }
}

pub fn encode_png_rgb(width: u32, height: u32, rows: &RgbRows) -> Vec<u8> {
    let mut raw = Vec::new();
    for row in rows {
        raw.push(0);
        for (red, green, blue) in row {
            raw.extend([*red, *green, *blue]);
        }
    }
    let header = [
        (width >> 24) as u8,
        (width >> 16) as u8,
        (width >> 8) as u8,
        width as u8,
        (height >> 24) as u8,
        (height >> 16) as u8,
        (height >> 8) as u8,
        height as u8,
        8,
        2,
        0,
        0,
        0,
    ];
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
    encoder.write_all(&raw).unwrap();
    let idat = encoder.finish().unwrap();
    let mut out = Vec::new();
    out.extend(b"\x89PNG\r\n\x1a\n");
    out.extend(png_chunk(b"IHDR", &header));
    out.extend(png_chunk(b"IDAT", &idat));
    out.extend(png_chunk(b"IEND", &[]));
    out
}

fn png_chunk(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut chunk = Vec::new();
    chunk.extend((payload.len() as u32).to_be_bytes());
    chunk.extend(kind);
    chunk.extend(payload);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(kind);
    hasher.update(payload);
    chunk.extend(hasher.finalize().to_be_bytes());
    chunk
}

pub fn frame_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn encode_gif_via_ffmpeg(frames: &[RgbFrame], delay_cs: u16) -> Result<Vec<u8>, String> {
    if frames.is_empty() {
        return Err("cannot encode GIF with zero frames".into());
    }
    let fps = (100_u32 / delay_cs.max(1) as u32).clamp(1, 50);
    let temp = tempfile::tempdir().map_err(|err| err.to_string())?;
    for (idx, (width, height, rows)) in frames.iter().enumerate() {
        let path = temp.path().join(format!("frame_{idx:04}.png"));
        std::fs::write(&path, encode_png_rgb(*width, *height, rows)).map_err(|err| err.to_string())?;
    }
    let output = temp.path().join("replay.gif");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            &fps.to_string(),
            "-i",
        ])
        .arg(temp.path().join("frame_%04d.png"))
        .args([
            "-frames:v",
            &frames.len().to_string(),
            "-vf",
            &format!(
                "fps={fps},split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=none"
            ),
        ])
        .arg(&output)
        .status()
        .map_err(|err| err.to_string())?;
    if !status.success() {
        return Err("ffmpeg GIF encode failed".into());
    }
    std::fs::read(&output).map_err(|err| err.to_string())
}
