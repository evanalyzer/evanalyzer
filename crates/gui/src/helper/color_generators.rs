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

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::classification_settings::Class;

    fn project_with_class(id: u32, hex_color: u32) -> ProjectWithRuntime {
        let mut project = ProjectWithRuntime::default();
        project.classification.classes.push(Class {
            id: ObjectClass::Valid(id),
            color: hex_color,
            name: format!("class-{id}"),
            ..Default::default()
        });
        project
    }

    #[test]
    fn get_colors_from_class_with_no_classes_defaults_to_opaque_red() {
        let project = ProjectWithRuntime::default();
        let color = get_colors_from_class(&project, 128, &HashSet::new());
        assert_eq!((color.red(), color.green(), color.blue(), color.alpha()), (0xFF, 0, 0, 0xFF));
    }

    #[test]
    fn get_colors_from_class_uses_the_registered_classs_color_and_given_transparency() {
        let project = project_with_class(1, 0x00FF00); // pure green
        let classes = HashSet::from([ObjectClass::Valid(1)]);
        let color = get_colors_from_class(&project, 64, &classes);
        assert_eq!((color.red(), color.green(), color.blue(), color.alpha()), (0, 0xFF, 0, 64));
    }

    #[test]
    fn get_colors_from_class_falls_back_to_red_for_an_unknown_class() {
        let project = ProjectWithRuntime::default(); // no classes registered
        let classes = HashSet::from([ObjectClass::Valid(99)]);
        let color = get_colors_from_class(&project, 200, &classes);
        assert_eq!((color.red(), color.green(), color.blue(), color.alpha()), (0xFF, 0, 0, 200));
    }

    #[test]
    fn get_colors_from_class_averages_multiple_classes() {
        let mut project = ProjectWithRuntime::default();
        project.classification.classes.push(Class {
            id: ObjectClass::Valid(1),
            color: 0xFF0000, // red
            ..Default::default()
        });
        project.classification.classes.push(Class {
            id: ObjectClass::Valid(2),
            color: 0x0000FF, // blue
            ..Default::default()
        });
        let classes = HashSet::from([ObjectClass::Valid(1), ObjectClass::Valid(2)]);
        let color = get_colors_from_class(&project, 255, &classes);
        // (255+0)/2 = 127 (integer division), (0+255)/2 = 127
        assert_eq!(color.red(), 127);
        assert_eq!(color.blue(), 127);
        assert_eq!(color.green(), 0);
    }

    #[test]
    fn color_from_rgb_scales_unit_floats_to_u8() {
        let color = color_from_rgb([1.0, 0.0, 0.5]);
        assert_eq!(color.red(), 255);
        assert_eq!(color.green(), 0);
        assert_eq!(color.blue(), 127);
    }
}
