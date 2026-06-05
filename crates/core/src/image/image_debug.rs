//! # image_debug
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-02-03
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use kornia_image::Image;
use kornia_tensor::CpuAllocator;

pub trait ImageDebugExt {
    fn print_window(&self);
}

impl ImageDebugExt for Image<u32, 1, CpuAllocator> {
    /// Prints a sub-region of the image to the console for debugging.
    fn print_window(&self) {
        let img_width = self.width();
        let img_height = self.height();
        let data = self.as_slice();

        let x_start = 0;
        let y_start = 0;
        let width = img_width;
        let height = img_height;

        // Print column headers
        print!("    ");
        for x in x_start..(x_start + width).min(img_width) {
            print!("{:3} ", x);
        }
        println!("\n    {}", "-".repeat(width * 4));

        for y in y_start..(y_start + height).min(img_height) {
            // Print row header
            print!("{:2} |", y);

            for x in x_start..(x_start + width).min(img_width) {
                let val = data[y * img_width + x];
                if val == 0 {
                    // Print 0s as dots to make them stand out
                    print!("  . ");
                } else {
                    print!("{:3} ", val);
                }
            }
            println!();
        }
        println!("-----------------------------------");
    }
}

impl ImageDebugExt for Image<f32, 1, CpuAllocator> {
    /// Prints a sub-region of the image to the console for debugging.
    fn print_window(&self) {
        let img_width = self.width();
        let img_height = self.height();
        let data = self.as_slice();

        let x_start = 0;
        let y_start = 0;
        let width = img_width;
        let height = img_height;

        // We'll use a width of 5 characters per column for floats (e.g., " 1.2 ")
        // If you want more precision, increase 'col_width' and the format string.
        let col_width = 5;

        // Print column headers
        print!("     "); // Extra space for row header
        for x in x_start..(x_start + width).min(img_width) {
            print!("{:>width$} ", x, width = col_width);
        }
        println!("\n    {}", "-".repeat((width + 1) * (col_width + 1)));

        for y in y_start..(y_start + height).min(img_height) {
            // Print row header
            print!("{:3} |", y);

            for x in x_start..(x_start + width).min(img_width) {
                let val = data[y * img_width + x];

                if val == 0.0 {
                    // Centered dot for empty space
                    print!("{:>width$} ", ".", width = col_width);
                } else if val.fract() == 0.0 {
                    // It's a whole number (like a Label ID), print without decimals
                    print!("{:>width$.0} ", val, width = col_width);
                } else {
                    // It's a true float (like EDM distance), print with 1 decimal place
                    print!("{:>width$.1} ", val, width = col_width);
                }
            }
            println!();
        }
        println!("{}", "-".repeat((width + 1) * (col_width + 1)));
    }
}
