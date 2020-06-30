use crate::{Vec3, NEAR_ZERO};

/** # Mat3 - 3x3 Matrix <f32>

 A 3x3 Matrix with 9 elements. Stored internally as `[f32; 9]`.

 #### Column Major

 This means data is stored and retrieved by column, not row. You can retrieve by a tuple like so:

 ```
 # use sawd_glm::Mat3;
 # let my_matrix = Mat3::zero();
 # let column: usize = 0;
 # let row: usize = 0;
 let value: f32 = my_matrix[(column, row)];
 ```

 or if you can do the math ahead of time and retreive by the index:

 ```
 # use sawd_glm::Mat3;
 # let my_matrix = Mat3::zero();
 # let index: usize = 0;
 let value: f32 = my_matrix[index];
 ```

 TODO: This behaviour is not very intuitive, index<usize> should return a slice instead so that `my_matrix[column][row]` is possible.

 #### Matrix Multiplication

 Matrix multiplication is **not** commutative, that means that `A*B ≠ B*A`.

 For example, given a product of multiple matrices that each represent a transformation, such as
 `M = A * B * C`. When you apply this transformation product to a Vec4, `M * V`, you can consider
 the transformations on the right `C` to be applied first, and the left `A` last.

 #### Default

 [Default](https://doc.rust-lang.org/nightly/core/default/trait.Default.html) is implemented for
 ease of use with ECS libraries like [specs](https://docs.rs/specs/0.16.1/specs/) that require
 components to implement Default.

 [`Mat3::default()`](#method.default) is equivalent to [`Mat3::identity()`](#method.identity) and we
 recommend using that function instead to make your code more explicit.

*/

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Mat3 ( pub(crate) [f32; 9] );

impl Mat3 {

    /// Create a new Mat3 from three Vec3 columns
    pub fn new(col1: Vec3, col2: Vec3, col3: Vec3) -> Mat3 {
        Mat3 ( [
            col1[0], col1[1], col1[2],
            col2[0], col2[1], col2[2],
            col3[0], col3[1], col3[2],
        ] )
    }

    /// Create a Mat3 with all elements equal to zero
    pub fn zero() -> Mat3 { Mat3([0.0;9]) }

    /// Create an 3x3 identity Matrix
    pub fn one() -> Mat3 {
        Self::identity()
    }

    /// Create an 3x3 identity Matrix
    pub fn identity() -> Mat3 {
        Mat3 ( [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ] )
    }

    /// Create a Mat3 from a 9 element array
    pub fn mat3(mat: [f32;9]) -> Mat3 { Mat3(mat) }

    /// Receive a copy of the data as an array
    ///
    /// Can also use [`into()`](#method.into)
    ///
    /// For a reference use [`as_ref()`](#method.as_ref) and for a mutable reference use [`as_mut()`](#method.as_mut)
    pub fn data(&self) -> [f32;9] { self.0 }

    pub fn get(&self, col: usize, row: usize) -> &f32 {
        &self[(col, row)]
    }
    pub fn get_mut(&mut self, col: usize, row: usize) -> &mut f32 {
        &mut self[(col, row)]
    }

    /// Test if this Mat3 is equals to another Mat3 for each component up to 1e-6
    pub fn equals(&self, other: Mat3) -> bool {
        self.equals_epsilon(other, NEAR_ZERO)
    }

    /// Test if this Mat3 is equals to another Mat3 for each component up to an epsilon
    pub fn equals_epsilon(&self, other: Mat3, epsilon: f32) -> bool {
        for i in 0..9 {
            if (self[i] - other[i]).abs() > epsilon {
                return false
            }
        }
        true
    }

    /// Receive the *absolute value* of each component in this Mat3
    pub fn abs(&self) -> Mat3 {
        Mat3([
            self[0].abs(), self[1].abs(), self[2].abs(),
            self[3].abs(), self[4].abs(), self[5].abs(),
            self[6].abs(), self[7].abs(), self[8].abs(),
        ])
    }

    /// Receive the aggregate average of each component
    pub fn agg_avg(&self) -> f32 {
        self.agg_sum() / 9.0
    }

    /// Receive the aggregate max of each component
    pub fn agg_max(&self) -> f32 {
        let mut m = self[0];
        for i in 1..9 {
            m = m.max(self[i]);
        }
        m
    }

    /// Receive the aggregate min of each component
    pub fn agg_min(&self) -> f32 {
        let mut m = self[0];
        for i in 1..9 {
            m = m.min(self[i]);
        }
        m
    }

    /// Receive the aggregate product of each component
    pub fn agg_prod(&self) -> f32 {
        let mut s = 1.0;
        for i in 0..9 {
            s *= self[i];
        }
        s
    }

    /// Receive the aggregate sum of each component
    pub fn agg_sum(&self) -> f32 {
        let mut s = 0.0;
        for i in 0..9 {
            s += self[i];
        }
        s
    }

    /// Receive a Mat3 with each component rounded up to the nearest integer
    pub fn ceil(&self) -> Mat3 {
        let mut m = Mat3::zero();
        for i in 0..9 {
            m[i] = self[i].ceil();
        }
        m
    }

    /// Receive a Mat3 clamped at some minimum and some maximum
    pub fn clamp(&self, min: f32, max: f32) -> Mat3 {
        let mut m = Mat3::zero();
        for i in 0..9 {
            m[i] = if self[i] < min { min } else if self[i] > max { max } else { self[i] };
        }
        m
    }

    /// Receive the f32 determinant of this Mat3
    pub fn determinant(&self) -> f32 {
          self[0] * (self[4] * self[8] - self[7] * self[5])
        - self[3] * (self[1] * self[8] - self[7] * self[2])
        + self[6] * (self[1] * self[5] - self[4] * self[2])
    }

    /// Receive a Mat3 with each component rounded down to the nearest integer
    pub fn floor(&self) -> Mat3 {
        let mut m = Mat3::zero();
        for i in 0..9 {
            m[i] = self[i].floor();
        }
        m
    }

    /// Receive a Mat3 with only the fractional portion of each component
    pub fn fract(&self) -> Mat3 {
        let mut m = Mat3::zero();
        for i in 0..9 {
            m[i] = self[i].fract();
        }
        m
    }

    /// Attempt to receive `Some` inverse of this Mat3, or `None` if the determinant is 0
    pub fn inverse(&self) -> Option<Mat3> {
        let det = self.determinant();
        if det == 0.0 {
            None
        } else {
            Some(
                Mat3::mat3([
                    (self[4] * self[8] - self[5] * self[7]) / det,
                    (self[7] * self[2] - self[8] * self[1]) / det,
                    (self[1] * self[5] - self[2] * self[4]) / det,

                    (self[5] * self[6] - self[3] * self[8]) / det,
                    (self[8] * self[0] - self[6] * self[2]) / det,
                    (self[2] * self[3] - self[0] * self[5]) / det,

                    (self[3] * self[7] - self[4] * self[6]) / det,
                    (self[6] * self[1] - self[7] * self[0]) / det,
                    (self[0] * self[4] - self[1] * self[3]) / det,
                ])
            )
        }
    }

    /// Receive the transpose of this Mat3
    pub fn transpose(&self) -> Mat3 {
        Mat3::mat3([
            self[0], self[3], self[6],
            self[1], self[4], self[7],
            self[2], self[5], self[8],
        ])
    }
}

impl std::ops::Index<usize> for Mat3 {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Mat3 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

impl std::ops::Index<(usize,usize)> for Mat3 {
    type Output = f32;

    fn index(&self, index: (usize,usize)) -> &f32 {
        &self.0[index.0 * 3 + index.1]
    }
}

impl std::ops::IndexMut<(usize,usize)> for Mat3 {
    fn index_mut(&mut self, index: (usize,usize)) -> &mut f32 {
        &mut self.0[index.0 * 3 + index.1]
    }
}

impl From<Vec3> for Mat3 {

    /// Converts a Vec3 into a Mat3 by placing the Vec3 components diagonally.
    fn from(f: Vec3) -> Self {
        Mat3 ( [
            f[0],  0.0,  0.0,
            0.0, f[1],  0.0,
            0.0,  0.0, f[2],
        ] )
    }
}

impl std::ops::Mul<Mat3> for Mat3 {
    type Output = Mat3;

    /// Matrix multiplication is **not** commutative, that means that `A*B ≠ B*A`.
    ///
    /// If there is more than one product in a single line, ie `A*B*C`, the product on the far right
    /// is considered to be evaluated first, ie `A*(B*C)`.
    fn mul(self, rhs: Mat3) -> Mat3 {
        let m1 = &self;
        let m2 = &rhs;
        Mat3 ( [
            m1[0]*m2[0]+m1[3]*m2[1]+m1[6]*m2[2], m1[1]*m2[0]+m1[4]*m2[1]+m1[7]*m2[2], m1[2]*m2[0]+m1[5]*m2[1]+m1[8]*m2[2],
            m1[0]*m2[3]+m1[3]*m2[4]+m1[6]*m2[5], m1[1]*m2[3]+m1[4]*m2[4]+m1[7]*m2[5], m1[2]*m2[3]+m1[5]*m2[4]+m1[8]*m2[5],
            m1[0]*m2[6]+m1[3]*m2[7]+m1[6]*m2[8], m1[1]*m2[6]+m1[4]*m2[7]+m1[7]*m2[8], m1[2]*m2[6]+m1[5]*m2[7]+m1[8]*m2[8],
        ] )
    }
}

impl std::ops::Mul<Vec3> for Mat3 {
    type Output = Vec3;

    /// Matrix * Vector = Vector-transformed
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3([
            rhs[0]*self[0] + rhs[1]*self[3] + rhs[2]*self[6],
            rhs[0]*self[1] + rhs[1]*self[4] + rhs[2]*self[7],
            rhs[0]*self[2] + rhs[1]*self[5] + rhs[2]*self[8],
        ])
    }
}

impl Into<[f32; 9]> for Mat3 {
    /// Receive a copy of the data as an array
    ///
    /// Can also use [`data()`](#method.data)
    fn into(self) -> [f32; 9] {
        self.0
    }
}

impl AsRef<[f32; 9]> for Mat3 {
    /// Receive a reference to the internal array
    fn as_ref(&self) -> &[f32; 9] {
        &self.0
    }
}

impl AsMut<[f32; 9]> for Mat3 {
    /// Receive a mutable reference to the internal array
    fn as_mut(&mut self) -> &mut [f32; 9] {
        &mut self.0
    }
}

impl Default for Mat3 {

    /// Default for Mat3 is [`Mat3::identity()`](#method.identity). Consider using that function
    /// instead to be more explicit.
    fn default() -> Self {
        Self::identity()
    }
}
