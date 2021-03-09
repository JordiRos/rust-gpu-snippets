A very simple instanced particles renderer using grr and rust-gpu shaders
Particles have position + scale, you could easily add colour or other attributes via a new buffer, or use instance_id in shader to do any procedural animation there.

shader.rs contains shader code for rust-gpu shader build step
particles.rs is a module you can insert in your code, pretty simple