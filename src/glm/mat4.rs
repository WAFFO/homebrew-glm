use crate::{Vec4, Mat3, NEAR_ZERO};

/** # Mat4 - 4x4 Matrix <f32>

 A 4x4 Matrix with 16 elements. Stored internally as `[f32; 16]`.

 #### Column Major

 This means data is stored and retrieved by column, not row. You can retrieve by a tuple like so:

 ```
 # use sawd_glm::Mat4;
 # let my_matrix = Mat4::zero();
 # let column: usize = 0;
 # let row: usize = 0;
 let value: f32 = my_matrix[(column, row)];
 ```

 or if you can do the math ahead of time and retreive by the index:

 ```
 # use sawd_glm::Mat4;
 # let my_matrix = Mat4::zero();
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

 [`Mat4::default()`](#method.default) is equivalent to [`Mat4::identity()`](#method.identity) and we
 recommend using that function instead to make your code more explicit.

*/

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Mat4 ( pub(crate) [f32; 16] );

impl Mat4 {

    /// Create a new Mat4 with four columns
    pub fn new(col1: Vec4, col2: Vec4, col3: Vec4, col4: Vec4) -> Mat4 {
        Mat4 ( [
            col1[0], col1[1], col1[2], col1[3],
            col2[0], col2[1], col2[2], col2[3],
            col3[0], col3[1], col3[2], col3[3],
            col4[0], col4[1], col4[2], col4[3],
        ] )
    }

    /// Create a Mat4 with all elements equal to zero
    pub fn zero() -> Mat4 { Mat4([0.0;16]) }

    /// Create an 4x4 identity Matrix
    pub fn one() -> Mat4 {
        Self::identity()
    }

    /// Create an 4x4 identity Matrix
    pub fn identity() -> Mat4 {
        Mat4 ( [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] )
    }

    /// Create a Mat4 from a 16 element array
    pub fn mat4(mat: [f32;16]) -> Mat4 { Mat4(mat) }

    /// Receive a copy of the data as an array
    ///
    /// Can also use [`into()`](#method.into)
    ///
    /// For a reference use [`as_ref()`](#method.as_ref) and for a mutable reference use [`as_mut()`](#method.as_mut)
    pub fn data(&self) -> [f32;16] { self.0 }

    pub fn get(&self, col: usize, row: usize) -> &f32 {
        &self[(col, row)]
    }
    pub fn get_mut(&mut self, col: usize, row: usize) -> &mut f32 {
        &mut self[(col, row)]
    }

    /// Test if this Mat4 is equals to another Mat4 for each component up to 1e-6
    pub fn equals(&self, other: Mat4) -> bool {
        self.equals_epsilon(other, NEAR_ZERO)
    }

    /// Test if this Mat4 is equals to another Mat4 for each component up to an epsilon
    pub fn equals_epsilon(&self, other: Mat4, epsilon: f32) -> bool {
        for i in 0..16 {
            if (self[i] - other[i]).abs() > epsilon {
                return false
            }
        }
        true
    }

    /// Receive the *absolute value* of each component in this Mat4
    pub fn abs(&self) -> Mat4 {
        Mat4([
            self[0].abs(), self[1].abs(), self[2].abs(), self[3].abs(),
            self[4].abs(), self[5].abs(), self[6].abs(), self[7].abs(),
            self[8].abs(), self[9].abs(), self[10].abs(), self[11].abs(),
            self[12].abs(), self[13].abs(), self[14].abs(), self[15].abs(),
        ])
    }

    /// Receive the aggregate average of each component
    pub fn agg_avg(&self) -> f32 {
        self.agg_sum() / 16.0
    }

    /// Receive the aggregate max of each component
    pub fn agg_max(&self) -> f32 {
        let mut m = self[0];
        for i in 1..16 {
            m = m.max(self[i]);
        }
        m
    }

    /// Receive the aggregate min of each component
    pub fn agg_min(&self) -> f32 {
        let mut m = self[0];
        for i in 1..16 {
            m = m.min(self[i]);
        }
        m
    }

    /// Receive the aggregate product of each component
    pub fn agg_prod(&self) -> f32 {
        let mut s = 1.0;
        for i in 0..16 {
            s *= self[i];
        }
        s
    }

    /// Receive the aggregate sum of each component
    pub fn agg_sum(&self) -> f32 {
        let mut s = 0.0;
        for i in 0..16 {
            s += self[i];
        }
        s
    }

    /// Receive a Mat4 with each component rounded up to the nearest integer
    pub fn ceil(&self) -> Mat4 {
        let mut m = Mat4::zero();
        for i in 0..16 {
            m[i] = self[i].ceil();
        }
        m
    }

    /// Receive a Mat4 clamped at some minimum and some maximum
    pub fn clamp(&self, min: f32, max: f32) -> Mat4 {
        let mut m = Mat4::zero();
        for i in 0..16 {
            m[i] = if self[i] < min { min } else if self[i] > max { max } else { self[i] };
        }
        m
    }

    /// Receive a Mat4 with each component rounded down to the nearest integer
    pub fn floor(&self) -> Mat4 {
        let mut m = Mat4::zero();
        for i in 0..16 {
            m[i] = self[i].floor();
        }
        m
    }

    /// Receive a Mat4 with only the fractional portion of each component
    pub fn fract(&self) -> Mat4 {
        let mut m = Mat4::zero();
        for i in 0..16 {
            m[i] = self[i].fract();
        }
        m
    }

    /// Receive the transpose of this Mat4
    pub fn transpose(&self) -> Mat4 {
        Mat4::mat4([
            self[0], self[4], self[8], self[12],
            self[1], self[5], self[9], self[13],
            self[2], self[6], self[10], self[14],
            self[3], self[7], self[11], self[15],
        ])
    }
}

impl std::ops::Index<usize> for Mat4 {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Mat4 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

impl std::ops::Index<(usize,usize)> for Mat4 {
    type Output = f32;

    fn index(&self, index: (usize,usize)) -> &f32 {
        &self.0[index.0 * 4 + index.1]
    }
}

impl std::ops::IndexMut<(usize,usize)> for Mat4 {
    fn index_mut(&mut self, index: (usize,usize)) -> &mut f32 {
        &mut self.0[index.0 * 4 + index.1]
    }
}

impl From<Mat3> for Mat4 {

    /// Converts a Mat3 into a Mat4 by placing the Mat3 components in the top left of the matrix.
    fn from(f: Mat3) -> Self {
        Mat4 ( [
            f[0], f[1], f[2],  0.0,
            f[3], f[4], f[5],  0.0,
            f[6], f[7], f[8],  0.0,
            0.0,  0.0,  0.0,  0.0,
        ] )
    }
}

impl From<Vec4> for Mat4 {

    /// Converts a Vec4 into a Mat4 by placing the Vec4 components diagonally.
    fn from(f: Vec4) -> Self {
        Mat4 ( [
            f[0],  0.0,  0.0,  0.0,
            0.0, f[1],  0.0,  0.0,
            0.0,  0.0, f[2],  0.0,
            0.0,  0.0,  0.0, f[3],
        ] )
    }
}

impl std::ops::Mul<Mat4> for Mat4 {
    type Output = Mat4;

    /// Matrix multiplication is **not** commutative, that means that `A*B ≠ B*A`.
    ///
    /// If there is more than one product in a single line, ie `A*B*C`, the product on the far right
    /// is considered to be evaluated first, ie `A*(B*C)`.
    fn mul(self, rhs: Mat4) -> Mat4 {
        let m1 = &self;
        let m2 = &rhs;
        Mat4 ( [
            m1[0]*m2[ 0]+m1[4]*m2[ 1]+m1[ 8]*m2[ 2]+m1[12]*m2[ 3],
            m1[1]*m2[ 0]+m1[5]*m2[ 1]+m1[ 9]*m2[ 2]+m1[13]*m2[ 3],
            m1[2]*m2[ 0]+m1[6]*m2[ 1]+m1[10]*m2[ 2]+m1[14]*m2[ 3],
            m1[3]*m2[ 0]+m1[7]*m2[ 1]+m1[11]*m2[ 2]+m1[15]*m2[ 3],

            m1[0]*m2[ 4]+m1[4]*m2[ 5]+m1[ 8]*m2[ 6]+m1[12]*m2[ 7],
            m1[1]*m2[ 4]+m1[5]*m2[ 5]+m1[ 9]*m2[ 6]+m1[13]*m2[ 7],
            m1[2]*m2[ 4]+m1[6]*m2[ 5]+m1[10]*m2[ 6]+m1[14]*m2[ 7],
            m1[3]*m2[ 4]+m1[7]*m2[ 5]+m1[11]*m2[ 6]+m1[15]*m2[ 7],

            m1[0]*m2[ 8]+m1[4]*m2[ 9]+m1[ 8]*m2[10]+m1[12]*m2[11],
            m1[1]*m2[ 8]+m1[5]*m2[ 9]+m1[ 9]*m2[10]+m1[13]*m2[11],
            m1[2]*m2[ 8]+m1[6]*m2[ 9]+m1[10]*m2[10]+m1[14]*m2[11],
            m1[3]*m2[ 8]+m1[7]*m2[ 9]+m1[11]*m2[10]+m1[15]*m2[11],

            m1[0]*m2[12]+m1[4]*m2[13]+m1[ 8]*m2[14]+m1[12]*m2[15],
            m1[1]*m2[12]+m1[5]*m2[13]+m1[ 9]*m2[14]+m1[13]*m2[15],
            m1[2]*m2[12]+m1[6]*m2[13]+m1[10]*m2[14]+m1[14]*m2[15],
            m1[3]*m2[12]+m1[7]*m2[13]+m1[11]*m2[14]+m1[15]*m2[15],
        ] )
    }
}

impl std::ops::Mul<Vec4> for Mat4 {
    type Output = Vec4;

    /// Matrix * Vector = Vector-transformed
    fn mul(self, rhs: Vec4) -> Vec4 {
        Vec4([
            rhs[0]*self[0] + rhs[1]*self[4] + rhs[2]*self[ 8] + rhs[3]*self[12],
            rhs[0]*self[1] + rhs[1]*self[5] + rhs[2]*self[ 9] + rhs[3]*self[13],
            rhs[0]*self[2] + rhs[1]*self[6] + rhs[2]*self[10] + rhs[3]*self[14],
            rhs[0]*self[3] + rhs[1]*self[7] + rhs[2]*self[11] + rhs[3]*self[15],
        ])
    }
}

impl Into<[f32; 16]> for Mat4 {
    /// Receive a copy of the data as an array
    ///
    /// Can also use [`data()`](#method.data)
    fn into(self) -> [f32; 16] {
        self.0
    }
}

impl AsRef<[f32; 16]> for Mat4 {
    /// Receive a reference to the internal array
    fn as_ref(&self) -> &[f32; 16] {
        &self.0
    }
}

impl AsMut<[f32; 16]> for Mat4 {
    /// Receive a mutable reference to the internal array
    fn as_mut(&mut self) -> &mut [f32; 16] {
        &mut self.0
    }
}

impl Default for Mat4 {

    /// Default for Mat4 is [`Mat4::identity()`](#method.identity). Consider using that function
    /// instead to be more explicit.
    fn default() -> Self {
        Self::identity()
    }
}
