/// Shapes of morphological `Kernels`.
///
/// Defines the geometry of the kernel used in morphological operations.
/// All kernels are centered at their geometric center.
#[derive(Debug, Clone)]
pub enum KernelShape {
    /// A rectangular box structuring element.
    ///
    /// All pixels within the box are included in the morphological operation.
    ///
    /// # Arguments
    ///
    /// * `size` - The side length of the square kernel (size x size).
    Box {
        /// `size` - The side length of the square kernel (size x size).
        size: usize,
    },

    /// A cross (plus) shaped structuring element.
    ///
    /// Only pixels along the horizontal and vertical center lines are included,
    /// forming a plus/cross pattern.
    ///
    /// # Arguments
    ///
    /// * `size` - The side length of the square kernel (size x size).
    Cross {
        /// `size` - The side length of the square cross kernel (size x size).
        size: usize,
    },

    /// An ellipse (or circle) shaped structuring element.
    ///
    /// Pixels inside the elliptical boundary are included, defined by the
    /// equation: (x^2/ rx^2) + (y^2 / ry^2) <= 1.
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the ellipse.
    /// * `height` - The height of the ellipse.
    Ellipse {
        /// * `width` - The width of the ellipse.
        width: usize,
        /// * `height` - The height of the ellipse.
        height: usize,
    },
}

/// A morphological structuring element.
///
/// The kernel defines the neighborhood structure used in morphological operations
/// (dilate, erode, open, close). It stores a binary mask where 1 indicates pixels
/// included in the operation and 0 indicates excluded pixels.
///
/// # Arguments
///
/// * `data` - A flat vector containing the kernel values (0 or 1).
/// * `width` - The width of the kernel in pixels.
/// * `height` - The height of the kernel in pixels.
///
/// # Example
///
/// ```rust
/// use kornia_imgproc::morphology::{Kernel, KernelShape};
///
/// // Create a 3x3 box kernel
/// let kernel = Kernel::new(KernelShape::Box { size: 3 });
/// assert_eq!(kernel.width(), 3);
/// assert_eq!(kernel.height(), 3);
/// assert_eq!(kernel.pad(), (1, 1));
/// ```
/// Whether a kernel's shape decomposes into independent 1D passes, letting
/// `dilate`/`erode` run an O(k) fast path instead of the full O(k^2) 2D scan.
/// Set once by the constructor that built the mask, not inferred from `data`
/// at use time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Separable {
    /// Not decomposable; scan the full 2D kernel (e.g. Ellipse).
    None,
    /// A full k*1 box: `dilate(f, Box) == dilate(dilate(f, Row), Col)`, since
    /// a box is the Minkowski sum of a row and a column segment.
    Box,
    /// A plus/cross shape: `dilate(f, Cross) == max(dilate(f, Row), dilate(f, Col))`
    /// (erosion: `min` instead of `max`), since a cross is the *union* of its
    /// row and column segments, not their Minkowski sum.
    Cross,
}

pub struct Kernel {
    data: Vec<u8>,
    width: usize,
    height: usize,
    separable: Separable,
}

impl Kernel {
    /// Create a morphological kernel from a shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the kernel (Box, Cross, or Ellipse).
    ///
    /// # Returns
    ///
    /// A [`Kernel`] struct with the appropriate data.
    pub fn new(shape: KernelShape) -> Self {
        match shape {
            KernelShape::Box { size } => box_kernel(size),
            KernelShape::Cross { size } => cross_kernel(size),
            KernelShape::Ellipse { width, height } => ellipse_kernel(width, height),
        }
    }

    /// Get a reference to the kernel data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the width of the kernel.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the height of the kernel.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the padding for the kernel (offset from center).
    pub fn pad(&self) -> (usize, usize) {
        (self.height / 2, self.width / 2)
    }

    /// Whether this kernel's shape supports the O(k) separable fast path.
    pub fn separable(&self) -> Separable {
        self.separable
    }

    /// Builds a kernel with an arbitrary mask and `Separable` tag, bypassing
    /// the normal shape constructors. Used to cross-validate the O(k)
    /// separable fast path against the O(k^2) naive scan: the naive path is
    /// only reachable when `separable() == Separable::None`, so this lets a
    /// test force it for a mask that's actually Box/Cross-shaped.
    #[cfg(test)]
    pub(crate) fn new_test_with_separable(
        data: Vec<u8>,
        width: usize,
        height: usize,
        separable: Separable,
    ) -> Self {
        Self {
            data,
            width,
            height,
            separable,
        }
    }
}

/// Create a box structuring element.
///
/// # Arguments
///
/// * `size` - The size of the kernel (size x size).
///
/// # Returns
///
/// A [`Kernel`] filled with 1s.
pub fn box_kernel(size: usize) -> Kernel {
    let data = vec![1u8; size * size];
    Kernel {
        data,
        width: size,
        height: size,
        separable: Separable::Box,
    }
}

/// Create a cross structuring element.
///
/// # Arguments
///
/// * `size` - The size of the kernel (size x size).
///
/// # Returns
///
/// A [`Kernel`] with 1s along the horizontal and vertical center lines.
pub fn cross_kernel(size: usize) -> Kernel {
    let mut data = vec![0u8; size * size];
    let mid = size / 2;

    // fill horizontal line
    for j in 0..size {
        data[mid * size + j] = 1;
    }

    // fill vertical line
    for i in 0..size {
        data[i * size + mid] = 1;
    }

    Kernel {
        data,
        width: size,
        height: size,
        separable: Separable::Cross,
    }
}

/// Create an ellipse structuring element.
///
/// # Arguments
///
/// * `width` - The width of the ellipse.
/// * `height` - The height of the ellipse.
///
/// # Returns
///
/// A [`Kernel`] with 1s inside the ellipse boundary.
pub fn ellipse_kernel(width: usize, height: usize) -> Kernel {
    let mut data = vec![0u8; width * height];
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let rx = width as f32 / 2.0;
    let ry = height as f32 / 2.0;

    for i in 0..height {
        for j in 0..width {
            let x = j as f32 - cx;
            let y = i as f32 - cy;
            // Ellipse equation: (x^2 / rx^2) + (y^2 / ry^2) <= 1
            if (x * x) / (rx * rx) + (y * y) / (ry * ry) <= 1.0 {
                data[i * width + j] = 1;
            }
        }
    }

    Kernel {
        data,
        width,
        height,
        separable: Separable::None,
    }
}
