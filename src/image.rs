use std::path::Path;

fn max_mip_levels_2d(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2() as u32 + 1
}

pub fn load_png(name: &str, grr: &grr::Device, format: grr::Format, downsample: bool) -> anyhow::Result<grr::Image> {
    let img = image::open(&Path::new(&name)).unwrap().to_rgba8();
    let img_width = img.width();
    let img_height = img.height();
    let img_data = img.into_raw();

    unsafe {
        let texture = grr.create_image(
            grr::ImageType::D2 {
                width: img_width,
                height: img_height,
                layers: 1,
                samples: 1,
            },
            format,
            if downsample {
                max_mip_levels_2d(img_width, img_height)
            } else {
                1
            },
        )?;

        grr.copy_host_to_image(
            &img_data,
            texture,
            grr::HostImageCopy {
                host_layout: grr::MemoryLayout {
                    base_format: grr::BaseFormat::RGBA,
                    format_layout: grr::FormatLayout::U8,
                    row_length: img_width,
                    image_height: img_height,
                    alignment: 4,
                },
                image_subresource: grr::SubresourceLayers {
                    level: 0,
                    layers: 0..1,
                },
                image_offset: grr::Offset { x: 0, y: 0, z: 0 },
                image_extent: grr::Extent {
                    width: img_width,
                    height: img_height,
                    depth: 1,
                },
            },
        );

        if downsample {
            grr.generate_mipmaps(texture);
        }

        Ok(texture)
    }
}