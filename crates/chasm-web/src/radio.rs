//! Radio voice: makes a squadmate sound like they are on comms.
//!
//! When a radio-capable NPC is beyond the radio distance, the game transport
//! runs every streamed audio chunk through [`RadioFx`]: a telephone-band
//! band-pass (≈300 Hz – 3.2 kHz), light saturation, a bed of static, and a
//! short squelch burst when the transmission opens. Filter and noise state
//! persist across the chunks of one turn, so the streamed slices join without
//! clicks. Pure Rust on 16-bit PCM WAV — no extra engine.

/// RBJ biquad (direct form I) with persistent state.
#[derive(Debug, Clone, Default)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn new(kind: FilterKind, sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * (cutoff / sample_rate).min(0.49);
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let (b0, b1, b2) = match kind {
            FilterKind::LowPass => ((1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0),
            FilterKind::HighPass => ((1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0),
        };
        let a0 = 1.0 + alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            ..Default::default()
        }
    }

    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

#[derive(Debug, Clone, Copy)]
enum FilterKind {
    LowPass,
    HighPass,
}

/// Stateful radio effect for one transmission (one NPC turn).
#[derive(Debug)]
pub struct RadioFx {
    static_level: f32,
    sample_rate: u32,
    filters: Vec<Biquad>,
    noise_state: u32,
    opened: bool,
}

/// The format facts we need from a WAV header.
struct WavInfo {
    channels: u16,
    sample_rate: u32,
    bits: u16,
    data_start: usize,
    data_len: usize,
}

fn parse_wav(bytes: &[u8]) -> Option<WavInfo> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut offset = 12usize;
    let mut format: Option<(u16, u32, u16)> = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?) as usize;
        let body = offset + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            let audio_format = u16::from_le_bytes([bytes[body], bytes[body + 1]]);
            if audio_format != 1 {
                return None;
            }
            let channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
            let sample_rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().ok()?);
            let bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
            format = Some((channels, sample_rate, bits));
        } else if id == b"data" {
            let (channels, sample_rate, bits) = format?;
            // Streaming headers may carry 0xFFFFFFFF; clamp to what is present.
            let data_len = size.min(bytes.len().saturating_sub(body));
            return Some(WavInfo {
                channels,
                sample_rate,
                bits,
                data_start: body,
                data_len,
            });
        }
        offset = body + size + (size & 1);
    }
    None
}

impl RadioFx {
    /// A new transmission. `static_level` is the noise amplitude (0..=0.3).
    pub fn new(static_level: f32) -> Self {
        Self {
            static_level: static_level.clamp(0.0, 0.3),
            sample_rate: 0,
            filters: Vec::new(),
            noise_state: 0x1234_5678,
            opened: false,
        }
    }

    fn noise(&mut self) -> f32 {
        // xorshift32: cheap, deterministic static.
        let mut x = self.noise_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.noise_state = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn ensure_filters(&mut self, sample_rate: u32) {
        if self.sample_rate == sample_rate && !self.filters.is_empty() {
            return;
        }
        let sr = sample_rate as f32;
        self.sample_rate = sample_rate;
        // Two cascaded sections per edge for a steeper, more "comms" band.
        self.filters = vec![
            Biquad::new(FilterKind::HighPass, sr, 320.0, 0.707),
            Biquad::new(FilterKind::HighPass, sr, 320.0, 0.707),
            Biquad::new(FilterKind::LowPass, sr, 3200.0, 0.707),
            Biquad::new(FilterKind::LowPass, sr, 3200.0, 0.707),
        ];
    }

    fn shape(&mut self, sample: f32) -> f32 {
        let mut y = sample;
        for filter in &mut self.filters {
            y = filter.process(y);
        }
        // Saturation: the radio's tiny speaker + clipped mic.
        const DRIVE: f32 = 2.2;
        y = (y * DRIVE).tanh() / DRIVE.tanh();
        let hiss = self.noise() * self.static_level;
        (y * 0.9 + hiss).clamp(-1.0, 1.0)
    }

    /// Processes one mono/stereo 16-bit PCM WAV chunk. Returns `None` for any
    /// other format (the caller then passes the audio through untouched).
    pub fn process_wav(&mut self, wav: &[u8]) -> Option<Vec<u8>> {
        let info = parse_wav(wav)?;
        if info.bits != 16 || info.channels == 0 {
            return None;
        }
        self.ensure_filters(info.sample_rate);
        let channels = info.channels as usize;
        let data = &wav[info.data_start..info.data_start + info.data_len];
        let frames = data.len() / (2 * channels);

        let mut out_samples: Vec<i16> = Vec::with_capacity(frames * channels + 4096);
        // Squelch burst when the channel opens (≈60 ms of shaped noise).
        if !self.opened {
            self.opened = true;
            let burst = (info.sample_rate as usize * 60) / 1000;
            for i in 0..burst {
                let envelope = 1.0 - i as f32 / burst as f32;
                let n = self.noise() * 0.35 * envelope;
                let s = self.shape(n);
                for _ in 0..channels {
                    out_samples.push((s * i16::MAX as f32) as i16);
                }
            }
        }
        for frame in 0..frames {
            // Mix to mono for the filter, then write the same sample to every channel.
            let mut sum = 0.0f32;
            for ch in 0..channels {
                let at = (frame * channels + ch) * 2;
                sum += i16::from_le_bytes([data[at], data[at + 1]]) as f32 / i16::MAX as f32;
            }
            let s = self.shape(sum / channels as f32);
            for _ in 0..channels {
                out_samples.push((s * i16::MAX as f32) as i16);
            }
        }

        let data_len = (out_samples.len() * 2) as u32;
        let byte_rate = info.sample_rate * u32::from(info.channels) * 2;
        let mut out = Vec::with_capacity(44 + data_len as usize);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&info.channels.to_le_bytes());
        out.extend_from_slice(&info.sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&(info.channels * 2).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for sample in out_samples {
            out.extend_from_slice(&sample.to_le_bytes());
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone_wav(freq: f32, sample_rate: u32, millis: u32) -> Vec<u8> {
        let n = (sample_rate * millis / 1000) as usize;
        let mut pcm = Vec::with_capacity(n * 2);
        for i in 0..n {
            let s = (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.5;
            pcm.extend_from_slice(&((s * i16::MAX as f32) as i16).to_le_bytes());
        }
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        wav.extend_from_slice(&pcm);
        wav
    }

    fn rms_tail(wav: &[u8], skip_samples: usize) -> f32 {
        let info = parse_wav(wav).unwrap();
        let data = &wav[info.data_start..info.data_start + info.data_len];
        let samples: Vec<f32> = data
            .chunks_exact(2)
            .skip(skip_samples)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / i16::MAX as f32)
            .collect();
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn band_pass_keeps_voice_band_and_cuts_rumble() {
        let sr = 24_000;
        let mut fx = RadioFx::new(0.0);
        let voice = fx.process_wav(&tone_wav(1000.0, sr, 300)).unwrap();
        let mut fx = RadioFx::new(0.0);
        let rumble = fx.process_wav(&tone_wav(60.0, sr, 300)).unwrap();
        // Skip the squelch burst + filter settling.
        let skip = (sr as usize * 120) / 1000;
        assert!(rms_tail(&voice, skip) > 0.2, "1 kHz should pass");
        assert!(rms_tail(&rumble, skip) < 0.05, "60 Hz should be cut");
    }

    #[test]
    fn squelch_only_on_the_first_chunk_and_lengths_add_up() {
        let sr = 24_000;
        let chunk = tone_wav(800.0, sr, 100);
        let mut fx = RadioFx::new(0.05);
        let first = fx.process_wav(&chunk).unwrap();
        let second = fx.process_wav(&chunk).unwrap();
        let burst_bytes = (sr as usize * 60 / 1000) * 2;
        assert_eq!(first.len(), chunk.len() + burst_bytes);
        assert_eq!(second.len(), chunk.len());
        assert!(RadioFx::new(0.0).process_wav(b"not a wav").is_none());
    }
}
