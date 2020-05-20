use crate::{GVec4, GMat3, NEAR_ZERO};
use crate::traits::Scalar;

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
pub struct GMat4<T: Scalar> ( pub(crate) [T; 16] );

pub type Mat4 = GMat4<f32>;

pub type DMat4 = GMat4<f64>;

impl<T: Scalar> GMat4<T> {

    /// Create a new Mat4 with four columns
    pub fn new(col1: GVec4<T>, col2: GVec4<T>, col3: GVec4<T>, col4: GVec4<T>) -> GMat4<T> {
        GMat4 ([
            col1[0], col1[1], col1[2], col1[3],
            col2[0], col2[1], col2[2], col2[3],
            col3[0], col3[1], col3[2], col3[3],
            col4[0], col4[1], col4[2], col4[3],
        ])
    }

    /// Create a Mat4 with all elements equal to zero
    pub fn zero() -> GMat4<T> { GMat4 ([T::zero();16]) }

    /// Create an 4x4 identity Matrix
    pub fn one() -> GMat4<T> {
        Self::identity()
    }

    /// Create an 4x4 identity Matrix
    pub fn identity() -> GMat4<T> {
        GMat4 ([
            T::one(), T::zero(), T::zero(), T::zero(),
            T::zero(), T::one(), T::zero(), T::zero(),
            T::zero(), T::zero(), T::one(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::one(),
        ])
    }

    /// Create a Mat4 from a 16 element array
    pub fn mat4(mat: [T;16]) -> GMat4<T> { GMat4 (mat) }

    /// Receive a copy of the data as an array
    ///
    /// Can also use [`into()`](#method.into)
    ///
    /// For a reference use [`as_ref()`](#method.as_ref) and for a mutable reference use [`as_mut()`](#method.as_mut)
    pub fn data(&self) -> [T;16] { self.0 }

    pub fn get(&self, col: usize, row: usize) -> &T {
        &self[(col, row)]
    }
    pub fn get_mut(&mut self, col: usize, row: usize) -> &mut T {
        &mut self[(col, row)]
    }

    /// Test if this Mat4 is equals to another Mat4 for each component up to 1e-6
    pub fn equals(&self, other: GMat4<T>) -> bool {
        self.equals_epsilon(other, T::cast(NEAR_ZERO))
    }

    /// Test if this Mat4 is equals to another Mat4 for each component up to an epsilon
    pub fn equals_epsilon(&self, other: GMat4<T>, epsilon: T) -> bool {
        for i in 0..16 {
            if (self[i] - other[i]).abs() > epsilon {
                return false
            }
        }
        true
    }

    /// Receive the *absolute value* of each component in this Mat4
    pub fn abs(&self) -> GMat4<T> {
        GMat4 ([
            self[0].abs(), self[1].abs(), self[2].abs(), self[3].abs(),
            self[4].abs(), self[5].abs(), self[6].abs(), self[7].abs(),
            self[8].abs(), self[9].abs(), self[10].abs(), self[11].abs(),
            self[12].abs(), self[13].abs(), self[14].abs(), self[15].abs(),
        ])
    }

    /// Receive the aggregate average of each component
    pub fn agg_avg(&self) -> T {
        self.agg_sum() / T::cast(16.0)
    }

    /// Receive the aggregate max of each component
    pub fn agg_max(&self) -> T {
        let mut m = self[0];
        for i in 1..16 {
            m = m.max(self[i]);
        }
        m
    }

    /// Receive the aggregate min of each component
    pub fn agg_min(&self) -> T {
        let mut m = self[0];
        for i in 1..16 {
            m = m.min(self[i]);
        }
        m
    }

    /// Receive the aggregate product of each component
    pub fn agg_prod(&self) -> T {
        let mut s = T::one();
        for i in 0..16 {
            s *= self[i];
        }
        s
    }

    /// Receive the aggregate sum of each component
    pub fn agg_sum(&self) -> T {
        let mut s = T::zero();
        for i in 0..16 {
            s += self[i];
        }
        s
    }

    /// Receive a Mat4 with each component rounded up to the nearest integer
    pub fn ceil(&self) -> GMat4<T> {
        let mut m = GMat4::zero();
        for i in 0..16 {
            m[i] = self[i].ceil();
        }
        m
    }

    /// Receive a Mat4 clamped at some minimum and some maximum
    pub fn clamp(&self, min: T, max: T) -> GMat4<T> {
        let mut m = GMat4::zero();
        for i in 0..16 {
            m[i] = if self[i] < min { min } else if self[i] > max { max } else { self[i] };
        }
        m
    }
}

impl<T: Scalar> std::ops::Index<usize> for GMat4<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.0[index]
    }
}

impl<T: Scalar> std::ops::IndexMut<usize> for GMat4<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.0[index]
    }
}

impl<T: Scalar> std::ops::Index<(usize,usize)> for GMat4<T> {
    type Output = T;

    fn index(&self, index: (usize,usize)) -> &T {
        &self.0[index.0 * 4 + index.1]
    }
}

impl<T: Scalar> std::ops::IndexMut<(usize,usize)> for GMat4<T> {
    fn index_mut(&mut self, index: (usize,usize)) -> &mut T {
        &mut self.0[index.0 * 4 + index.1]
    }
}

impl<T: Scalar> From<GMat3<T>> for GMat4<T> {

    /// Converts a Mat3 into a Mat4 by placing the Mat3 components in the top left of the matrix.
    fn from(f: GMat3<T>) -> Self {
        GMat4 ([
            f[0], f[1], f[2],  T::zero(),
            f[3], f[4], f[5],  T::zero(),
            f[6], f[7], f[8],  T::zero(),
            T::zero(),  T::zero(),  T::zero(),  T::zero(),
        ])
    }
}

impl<T: Scalar> From<GVec4<T>> for GMat4<T> {

    /// Converts a Vec4 into a Mat4 by placing the Vec4 components diagonally.
    fn from(f: GVec4<T>) -> Self {
        GMat4 ([
            f[0],  T::zero(),  T::zero(),  T::zero(),
            T::zero(), f[1],  T::zero(),  T::zero(),
            T::zero(),  T::zero(), f[2],  T::zero(),
            T::zero(),  T::zero(),  T::zero(), f[3],
        ])
    }
}

impl<T: Scalar> std::ops::Mul<GMat4<T>> for GMat4<T> {
    type Output = GMat4<T>;

    /// Matrix multiplication is **not** commutative, that means that `A*B ≠ B*A`.
    ///
    /// If there is more than one product in a single line, ie `A*B*C`, the product on the far right
    /// is considered to be evaluated first, ie `A*(B*C)`.
    fn mul(self, rhs: GMat4<T>) -> GMat4<T> {
        let m1 = &self;
        let m2 = &rhs;
        GMat4 ([
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
        ])
    }
}

impl<T: Scalar> std::ops::Mul<GVec4<T>> for GMat4<T> {
    type Output = GVec4<T>;

    /// Matrix * Vector = Vector-transformed
    fn mul(self, rhs: GVec4<T>) -> GVec4<T> {
        GVec4 ([
            rhs[0]*self[0] + rhs[1]*self[4] + rhs[2]*self[ 8] + rhs[3]*self[12],
            rhs[0]*self[1] + rhs[1]*self[5] + rhs[2]*self[ 9] + rhs[3]*self[13],
            rhs[0]*self[2] + rhs[1]*self[6] + rhs[2]*self[10] + rhs[3]*self[14],
            rhs[0]*self[3] + rhs[1]*self[7] + rhs[2]*self[11] + rhs[3]*self[15],
        ])
    }
}

impl<T: Scalar> Into<[T; 16]> for GMat4<T> {
    /// Receive a copy of the data as an array
    ///
    /// Can also use [`data()`](#method.data)
    fn into(self) -> [T; 16] {
        self.0
    }
}

impl<T: Scalar> AsRef<[T; 16]> for GMat4<T> {
    /// Receive a reference to the internal array
    fn as_ref(&self) -> &[T; 16] {
        &self.0
    }
}

impl<T: Scalar> AsMut<[T; 16]> for GMat4<T> {
    /// Receive a mutable reference to the internal array
    fn as_mut(&mut self) -> &mut [T; 16] {
        &mut self.0
    }
}

impl<T: Scalar> Default for GMat4<T> {

    /// Default for Mat4 is [`Mat4::identity()`](#method.identity). Consider using that function
    /// instead to be more explicit.
    fn default() -> Self {
        Self::identity()
    }
}
