use crate::algos::{ImageAlgorithm, PipelineCache, PipelineContext};
use crate::image::ImageContainer;
use evanalyzer_cfg::core_types::CitationMetadata;
use evanalyzer_cfg::core_types::ImageAddress;
use evanalyzer_cfg::core_types::InternalErrors;
use macros::CommandsMeta;
use std::sync::Arc;

/// The mathematical or logical operation to perform between two images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operand {
    /// No operation; typically used as a placeholder.
    None,
    /// Unary operation: Negates the intensities of the primary image.
    Invert,
    /// Arithmetic addition: `A + B`. (Clamped to the maximum pixel value).
    Add,
    /// Arithmetic subtraction: `A - B`. (Clamped to zero).
    Subtract,
    /// Arithmetic multiplication: `A * B`.
    Multiply,
    /// Arithmetic division: `A / B`.
    Divide,
    /// Bitwise AND operation.
    And,
    /// Bitwise OR operation.
    Or,
    /// Bitwise XOR operation.
    Xor,
    /// Per-pixel minimum: `min(A, B)`. (Darkest Pixel).
    Min,
    /// Per-pixel maximum: `max(A, B)`. (Brightest Pixel).
    Max,
    /// Arithmetic mean: `(A + B) / 2`.
    Average,
    /// Absolute difference: `|A - B|`. Useful for change detection.
    DifferenceType,
}

/// A filter that performs pixel-wise mathematical operations between the current
/// pipeline image and a secondary image stored in the cache.
///
/// This command allows for complex image blending, masking, and comparison.
///
/// # Examples
///
/// ```
/// use imagec::backend::algos::{ImageMath, Operand};
/// let subtract_bg = ImageMath {
///     operand: Operand::Subtract,
///     second_image_address: ImageAddress::from("background"),
///     swap_operands: false,
/// };
/// ```
#[derive(CommandsMeta)]
#[cmdsmeta(category = "Preprocessing")]
pub struct ImageMath {
    /// The specific mathematical or logical operator to apply.
    pub operand: Operand,

    /// The address of the second image in the [`ImageCache`] to use for the operation.
    pub second_image_address: ImageAddress,

    /// If false, the calculation is `(Current Image OP Cached Image)`.
    /// If true, the calculation is `(Cached Image OP Current Image)`.
    ///
    /// This is critical for non-commutative operations like Subtraction or Division.
    pub swap_operands: bool,
}

impl ImageAlgorithm for ImageMath {
    /// Executes per-pixel mathematical or logical operations between two images.
    ///
    /// This algorithm supports both unary operations (applied to the current image)
    /// and binary operations (applied between the current image and a cached image).
    ///
    /// # Pipeline Logic
    /// 1. **Unary Check**: If the operation is `Invert`, it processes the current image immediately.
    /// 2. **Cache Retrieval**: For binary operations, it fetches the second image from
    ///    the [`PipelineCache`] using `second_image_address`.
    /// 3. **Commutativity Handling**: If `swap_operands` is true, the cached image becomes
    ///    the primary operand (the "left side" of the equation).
    /// 4. **Pixel Math**: Performs the operation and writes the result back to the context.
    ///
    /// # Errors
    ///
    /// - Returns [`InternalErrors::CacheMiss`] if the second image is not found.
    /// - Returns [`InternalErrors::FormatMismatch`] if the image types (e.g., F32 vs U8) do not match.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        cache: &mut PipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.operand == Operand::None {
            return Ok(());
        }

        // Handle Unary Operations (Invert)
        // This avoids the cache lookup and doesn't require a second image.
        if self.operand == Operand::Invert {
            match (ctx.image.as_ref(), Arc::make_mut(&mut ctx.scratch_pad)) {
                (ImageContainer::F32Gray(input), ImageContainer::F32Gray(output)) => {
                    self.apply_unary_math(input.as_slice(), output.as_slice_mut());
                }
                (ImageContainer::F32Rgb(input), ImageContainer::F32Rgb(output)) => {
                    self.apply_unary_math(input.as_slice(), output.as_slice_mut());
                }
                _ => {
                    return Err(InternalErrors::FormatMismatch {
                        expected: "F32Rgb or F32Gray".into(),
                        found: format!("{:?}", ctx.image),
                    });
                }
            }
            ctx.swap()?;
            return Ok(());
        }

        // Handle Binary Operations
        // Only get the secondary image if we actually need it. Both branches are
        // now a cheap Arc clone (refcount bump), not a pixel copy: when the
        // second image *is* the scratchpad, make_mut below will see the extra
        // strong reference this creates and correctly copy-on-write into a
        // separate output buffer instead of aliasing `second`.
        let second = if self.second_image_address == ImageAddress::Scratchpad {
            Arc::clone(&ctx.scratch_pad)
        } else {
            cache
                .image_cache
                .images
                .get(&self.second_image_address)
                .ok_or_else(|| InternalErrors::Generic("Secondary image not found".to_string()))?
                .clone()
        };
        match (
            ctx.image.as_ref(),
            second.as_ref(),
            Arc::make_mut(&mut ctx.scratch_pad),
        ) {
            (
                ImageContainer::F32Gray(in_a),
                ImageContainer::F32Gray(in_b),
                ImageContainer::F32Gray(out),
            ) => {
                self.apply_binary_math(in_a.as_slice(), in_b.as_slice(), out.as_slice_mut())?;
            }
            (
                ImageContainer::F32Rgb(in_a),
                ImageContainer::F32Rgb(in_b),
                ImageContainer::F32Rgb(out),
            ) => {
                self.apply_binary_math(in_a.as_slice(), in_b.as_slice(), out.as_slice_mut())?;
            }
            _ => {
                return Err(InternalErrors::FormatMismatch {
                    expected: "Matching formats between Input and Secondary image".to_string(),
                    found: "Format mismatch".to_string(),
                });
            }
        }

        ctx.swap()?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Image Math"
    }

    fn cite(&self) -> Option<&'static CitationMetadata> {
        None
    }
}
impl ImageMath {
    /// Separate function for Invert (Unary)
    fn apply_unary_math(&self, primary: &[f32], output: &mut [f32]) {
        primary.iter().zip(output.iter_mut()).for_each(|(&a, out)| {
            *out = 1.0 - a;
        });
    }

    /// Binary operations (Requires two images)
    ///
    /// # Errors
    ///
    /// Returns [`InternalErrors::FormatMismatch`] if `pipeline_data` and
    /// `cache_data` don't have the same pixel count (e.g. `second_image_address`
    /// points at a checkpoint cached from a different-sized processing stage).
    /// `.zip()`ing slices of different lengths silently stops at the shorter
    /// one, which would otherwise leave the tail of `output` untouched -
    /// wrong pixel values with no error, not just a truncated write, since
    /// `output` is the scratch pad and may still hold stale data from a
    /// previous use.
    fn apply_binary_math(
        &self,
        pipeline_data: &[f32],
        cache_data: &[f32],
        output: &mut [f32],
    ) -> Result<(), InternalErrors> {
        if pipeline_data.len() != cache_data.len() || pipeline_data.len() != output.len() {
            return Err(InternalErrors::FormatMismatch {
                expected: format!(
                    "{} pixels (matching the current image)",
                    pipeline_data.len()
                ),
                found: format!("{} pixels in the secondary image", cache_data.len()),
            });
        }

        pipeline_data
            .iter()
            .zip(cache_data.iter())
            .zip(output.iter_mut())
            .for_each(|((&p, &c), out)| {
                // Determine which is 'a' and which is 'b' based on the swap flag
                let (a, b) = if self.swap_operands { (c, p) } else { (p, c) };
                *out = match self.operand {
                    Operand::Add => a + b,
                    Operand::Subtract => a - b, // Important for non-commutative!
                    Operand::Multiply => a * b,
                    Operand::Divide => {
                        if b != 0.0 {
                            a / b
                        } else {
                            a
                        }
                    } // Important!
                    Operand::And => f32::from_bits(a.to_bits() & b.to_bits()),
                    Operand::Or => f32::from_bits(a.to_bits() | b.to_bits()),
                    Operand::Xor => f32::from_bits(a.to_bits() ^ b.to_bits()),
                    Operand::Min => a.min(b),
                    Operand::Max => a.max(b),
                    Operand::Average => (a + b) * 0.5,
                    Operand::DifferenceType => (a - b).abs(),
                    _ => a,
                };
            });
        Ok(())
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::image::PixelSizes;
    use crate::pipeline::pipeline::PipelineImageMeta;

    use super::*;
    use evanalyzer_cfg::core_types::MemoryId;
    use kornia_image::Image;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;
    use std::path::PathBuf;
    use std::sync::Arc;

    // Helper to create a dummy F32Gray image container
    fn create_test_gray(val: f32) -> ImageContainer {
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        let data = vec![val; 4];
        ImageContainer::new_f32_gray_from_image_test(Image::new(size, data, CpuAllocator).unwrap())
    }

    fn run_math_test(op: Operand, val1: f32, val2: f32, swap: bool) -> f32 {
        let mut ctx = PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 2,
                    height: 2,
                },
                full_image_width: ImageSize {
                    width: 2,
                    height: 2,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            create_test_gray(val1).into(),
        )
        .unwrap();
        let mut cache = PipelineCache::default();

        // Mock secondary image in cache
        let second_image_address = ImageAddress::Memory(MemoryId::PipelineContext(1));
        cache.image_cache.images.insert(
            second_image_address.clone(),
            Arc::new(create_test_gray(val2)),
        );

        let cmd = ImageMath {
            operand: op,
            second_image_address,
            swap_operands: swap,
        };

        cmd.execute(&mut ctx, &mut cache).unwrap();

        // After swap, the result is in ctx.image
        if let ImageContainer::F32Gray(img) = ctx.image.as_ref() {
            img.as_slice()[0]
        } else {
            panic!("Wrong output format");
        }
    }

    #[test]
    fn test_basic_arithmetic() {
        assert_eq!(run_math_test(Operand::Add, 0.5, 0.2, false), 0.7);
        assert_eq!(run_math_test(Operand::Subtract, 0.5, 0.2, false), 0.3);
        assert_eq!(run_math_test(Operand::Multiply, 0.5, 0.5, false), 0.25);
        assert_eq!(run_math_test(Operand::Divide, 1.0, 2.0, false), 0.5);
    }

    #[test]
    fn test_logic_and_comparison() {
        assert_eq!(run_math_test(Operand::Min, 0.1, 0.9, false), 0.1);
        assert_eq!(run_math_test(Operand::Max, 0.1, 0.9, false), 0.9);
        assert_eq!(run_math_test(Operand::Average, 0.0, 1.0, false), 0.5);
        assert_eq!(run_math_test(Operand::DifferenceType, 0.2, 0.8, false), 0.6);
    }

    #[test]
    fn test_invert() {
        // Invert ignores the secondary image
        assert!((run_math_test(Operand::Invert, 0.8, 0.0, false) - 0.2).abs() < 1e-6);
        assert!((run_math_test(Operand::Invert, 0.0, 1.0, false) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bitwise_logic() {
        // Test bits: 0.0 (all zeros) and a specific bit pattern
        let val_a = f32::from_bits(0b1010);
        let val_b = f32::from_bits(0b1100);

        // 1010 AND 1100 = 1000 (8)
        assert_eq!(
            run_math_test(Operand::And, val_a, val_b, false),
            f32::from_bits(0b1000)
        );
        // 1010 OR 1100 = 1110 (14)
        assert_eq!(
            run_math_test(Operand::Or, val_a, val_b, false),
            f32::from_bits(0b1110)
        );
        // 1010 XOR 1100 = 0110 (6)
        assert_eq!(
            run_math_test(Operand::Xor, val_a, val_b, false),
            f32::from_bits(0b0110)
        );
    }

    #[test]
    fn test_subtraction_with_swap() {
        // Case 1: No Swap (Pipeline - Cache) -> 0.8 - 0.2 = 0.6
        let result_normal = run_math_test(Operand::Subtract, 0.8, 0.2, false);
        assert!((result_normal - 0.6).abs() < 1e-6);

        // Case 2: Swap Enabled (Cache - Pipeline) -> 0.2 - 0.8 = -0.6
        let result_swapped = run_math_test(Operand::Subtract, 0.8, 0.2, true);
        assert!((result_swapped - (-0.6)).abs() < 1e-6);
    }

    #[test]
    fn test_division_with_swap() {
        // Case 1: No Swap (Pipeline / Cache) -> 10.0 / 2.0 = 5.0
        let result_normal = run_math_test(Operand::Divide, 10.0, 2.0, false);
        assert!((result_normal - 5.0).abs() < 1e-6);

        // Case 2: Swap Enabled (Cache / Pipeline) -> 2.0 / 10.0 = 0.2
        let result_swapped = run_math_test(Operand::Divide, 10.0, 2.0, true);
        assert!((result_swapped - 0.2).abs() < 1e-6);
    }

    #[test]
    fn binary_math_errors_on_mismatched_secondary_image_size_instead_of_silently_truncating() {
        // Regression test: `apply_binary_math` used to `.zip()` the primary,
        // secondary, and output slices together - which silently stops at
        // the shortest of the three instead of erroring, leaving the tail of
        // `output` (the scratch pad, possibly still holding stale data from
        // a previous use) untouched. A cached checkpoint from a
        // different-sized processing stage must now be a clean error
        // instead of wrong pixel values with no signal anything went wrong.
        let mut ctx = PipelineContext::new_from_image(
            PathBuf::default(),
            PipelineImageMeta {
                image_tile_info: crate::ImageTile {
                    offset_x: 0,
                    offset_y: 0,
                    width: 2,
                    height: 2,
                },
                full_image_width: ImageSize {
                    width: 2,
                    height: 2,
                },
                is_rgb: false,
                nr_of_bits: 8,
                pixel_sizes: PixelSizes {
                    px_size_x: 1.0,
                    px_size_y: 1.0,
                    px_size_z: 1.0,
                },
            },
            create_test_gray(0.5).into(),
        )
        .unwrap();
        let mut cache = PipelineCache::default();

        // Secondary image cached at a different pixel count than the
        // current (2x2) pipeline image.
        let smaller_second = ImageContainer::new_f32_gray_from_image_test(
            Image::new(
                ImageSize {
                    width: 1,
                    height: 1,
                },
                vec![0.2],
                CpuAllocator,
            )
            .unwrap(),
        );
        let second_image_address = ImageAddress::Memory(MemoryId::PipelineContext(1));
        cache
            .image_cache
            .images
            .insert(second_image_address.clone(), Arc::new(smaller_second));

        let cmd = ImageMath {
            operand: Operand::Add,
            second_image_address,
            swap_operands: false,
        };

        let result = cmd.execute(&mut ctx, &mut cache);
        assert!(
            matches!(result, Err(InternalErrors::FormatMismatch { .. })),
            "expected a FormatMismatch error for mismatched pixel counts, got {result:?}"
        );
    }

    #[test]
    fn test_commutative_ops_unaffected() {
        // For Add, swapping 0.1 and 0.4 should always result in 0.5
        let res1 = run_math_test(Operand::Add, 0.1, 0.4, false);
        let res2 = run_math_test(Operand::Add, 0.1, 0.4, true);
        assert_eq!(res1, res2);
    }
}
