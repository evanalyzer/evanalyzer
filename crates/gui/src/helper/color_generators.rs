use evanalyzer_app::extensions::project_ext::ProjectExt;
use evanalyzer_app::ProjectWithRuntime;
use evanalyzer_cfg::core_types::ObjectClass;
use slint::Color;
use std::collections::HashSet;

pub fn get_colors_from_class(
    project: &ProjectWithRuntime,
    transparency: u8,
    classes: &HashSet<ObjectClass>,
) -> Color {
    let mut r = 0u32;
    let mut g = 0u32;
    let mut b = 0u32;
    let mut a = 0u32;

    for class in classes {
        let col = color_from_class(project, transparency, class);
        r += col.red() as u32;
        g += col.green() as u32;
        b += col.blue() as u32;
        a += col.alpha() as u32;
    }

    let count = classes.len() as u32;
    if count == 0 {
        Color::from_rgb_u8(0xFF, 0, 0)
    } else {
        Color::from_argb_u8(
            (a / count) as u8,
            (r / count) as u8,
            (g / count) as u8,
            (b / count) as u8,
        )
    }
}

fn color_from_class(
    project: &ProjectWithRuntime,
    transparency: u8,
    object_class: &ObjectClass,
) -> Color {
    let color = match project.get_class_from_id(object_class) {
        Some(class) => class.color as u32,
        None => 0xff0000 as u32, // Not found
    };
    Color::from_argb_u8(
        transparency,
        ((color >> 16) & 0xFF) as u8,
        ((color >> 8) & 0xFF) as u8,
        (color & 0xFF) as u8,
    )
}

pub fn color_from_rgb(rgb: [f32; 3]) -> Color {
    Color::from_rgb_u8(
        ((rgb[0] * 255.0) as u8) as u8,
        ((rgb[1] * 255.0) as u8) as u8,
        ((rgb[2] * 255.0) as u8) as u8,
    )
}
