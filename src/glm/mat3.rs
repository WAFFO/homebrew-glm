use crate::{GVec3, NEAR_ZERO};
use crate::traits::Scalar;

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
pub struct GMat3<T: Scalar> ( pub(crate) [T; 9] );

pub type Mat3 = GMat3<f32>;

pub type DMat3 = GMat3<f64>;

impl<T: Scalar> GMat3<T> {



    /// Create a new Mat3 from three Vec3 columns
    pub fn new(col1: GVec3<T>, col2: GVec3<T>, col3: GVec3<T>) -> GMat3<T> {
        GMat3 ([
            col1[0], col1[1], col1[2],
            col2[0], col2[1], col2[2],
            col3[0], col3[1], col3[2],
        ])
    }

    /// Create a Mat3 with all elements equal to zero
    pub fn zero() -> GMat3<T> { GMat3([T::zero();9]) }

    /// Create an 3x3 identity Matrix
    pub fn one() -> GMat3<T> {
        Self::identity()
    }

    /// Create an 3x3 identity Matrix
    pub fn identity() -> GMat3<T> {
        GMat3 ([
            T::one(), T::zero(), T::zero(),
            T::zero(), T::one(), T::zero(),
            T::zero(), T::zero(), T::one(),
        ])
    }

    /// Create a Mat3 from a 9 element array
    pub fn mat3(mat: [T;9]) -> GMat3<T> { GMat3(mat) }

    /// Receive a copy of the data as an array
    ///
    /// Can also use [`into()`](#method.into)
    ///
    /// For a reference use [`as_ref()`](#method.as_ref) and for a mutable reference use [`as_mut()`](#method.as_mut)
    pub fn data(&self) -> [T;9] { self.0 }

    pub fn get(&self, col: usize, row: usize) -> &T {
        &self[(col, row)]
    }
    pub fn get_mut(&mut self, col: usize, row: usize) -> &mut T {
        &mut self[(col, row)]
    }

    /// Test if this Mat3 is equals to another Mat3 for each component up to 1e-6
    pub fn equals(&self, other: GMat3<T>) -> bool {
        self.equals_epsilon(other, T::cast(NEAR_ZERO))
    }

    /// Test if this Mat3 is equals to another Mat3 for each component up to an epsilon
    pub fn equals_epsilon(&self, other: GMat3<T>, epsilon: T) -> bool {
        for i in 0..9 {
            if (self[i] - other[i]).abs() > epsilon {
                return false
            }
        }
        true
    }

    /// Receive the *absolute value* of each component in this Mat3
    pub fn abs(&self) -> GMat3<T> {
        GMat3([
            self[0].abs(), self[1].abs(), self[2].abs(),
            self[3].abs(), self[4].abs(), self[5].abs(),
            self[6].abs(), self[7].abs(), self[8].abs(),
        ])
    }

    /// Receive the aggregate average of each component
    pub fn agg_avg(&self) -> T {
        self.agg_sum() / T::cast(9.0)
    }

    /// Receive the aggregate max of each component
    pub fn agg_max(&self) -> T {
        let mut m = self[0];
        for i in 1..9 {
            m = m.max(self[i]);
        }
        m
    }

    /// Receive the aggregate min of each component
    pub fn agg_min(&self) -> T {
        let mut m = self[0];
        for i in 1..9 {
            m = m.min(self[i]);
        }
        m
    }

    /// Receive the aggregate product of each component
    pub fn agg_prod(&self) -> T {
        let mut s = T::one();
        for i in 0..9 {
            s *= self[i];
        }
        s
    }

    /// Receive the aggregate sum of each component
    pub fn agg_sum(&self) -> T {
        let mut s = T::zero();
        for i in 0..9 {
            s += self[i];
        }
        s
    }

    /// Receive a Mat3 with each component rounded up to the nearest integer
    pub fn ceil(&self) -> GMat3<T> {
        let mut m = GMat3::zero();
        for i in 0..9 {
            m[i] = self[i].ceil();
        }
        m
    }

    /// Receive a Mat3 clamped at some minimum and some maximum
    pub fn clamp(&self, min: T, max: T) -> GMat3<T> {
        let mut m = GMat3::zero();
        for i in 0..9 {
            m[i] = if self[i] < min { min } else if self[i] > max { max } else { self[i] };
        }
        m
    }
}

impl<T: Scalar> std::ops::Index<usize> for GMat3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.0[index]
    }
}

impl<T: Scalar> std::ops::IndexMut<usize> for GMat3<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.0[index]
    }
}

impl<T: Scalar> std::ops::Index<(usize,usize)> for GMat3<T> {
    type Output = T;

    fn index(&self, index: (usize,usize)) -> &T {
        &self.0[index.0 * 3 + index.1]
    }
}

impl<T: Scalar> std::ops::IndexMut<(usize,usize)> for GMat3<T> {
    fn index_mut(&mut self, index: (usize,usize)) -> &mut T {
        &mut self.0[index.0 * 3 + index.1]
    }
}

impl<T: Scalar> From<GVec3<T>> for GMat3<T> {

    /// Converts a Vec3 into a Mat3 by placing the Vec3 components diagonally.
    fn from(f: GVec3<T>) -> Self {
        GMat3 ([
            f[0],  T::zero(),  T::zero(),
            T::zero(), f[1],  T::zero(),
            T::zero(),  T::zero(), f[2],
        ])
    }
}

impl<T: Scalar> std::ops::Mul<GMat3<T>> for GMat3<T> {
    type Output = GMat3<T>;

    /// Matrix multiplication is **not** commutative, that means that `A*B ≠ B*A`.
    ///
    /// If there is more than one product in a single line, ie `A*B*C`, the product on the far right
    /// is considered to be evaluated first, ie `A*(B*C)`.
    fn mul(self, rhs: GMat3<T>) -> GMat3<T> {
        let m1 = &self;
        let m2 = &rhs;
        GMat3 ([
            m1[0]*m2[0]+m1[3]*m2[1]+m1[6]*m2[2], m1[1]*m2[0]+m1[4]*m2[1]+m1[7]*m2[2], m1[2]*m2[0]+m1[5]*m2[1]+m1[8]*m2[2],
            m1[0]*m2[3]+m1[3]*m2[4]+m1[6]*m2[5], m1[1]*m2[3]+m1[4]*m2[4]+m1[7]*m2[5], m1[2]*m2[3]+m1[5]*m2[4]+m1[8]*m2[5],
            m1[0]*m2[6]+m1[3]*m2[7]+m1[6]*m2[8], m1[1]*m2[6]+m1[4]*m2[7]+m1[7]*m2[8], m1[2]*m2[6]+m1[5]*m2[7]+m1[8]*m2[8],
        ])
    }
}

impl<T: Scalar> std::ops::Mul<GVec3<T>> for GMat3<T> {
    type Output = GVec3<T>;

    /// Matrix * Vector = Vector-transformed
    fn mul(self, rhs: GVec3<T>) -> GVec3<T> {
        GVec3 ([
            rhs[0]*self[0] + rhs[1]*self[3] + rhs[2]*self[6],
            rhs[0]*self[1] + rhs[1]*self[4] + rhs[2]*self[7],
            rhs[0]*self[2] + rhs[1]*self[5] + rhs[2]*self[8],
        ])
    }
}

impl<T: Scalar> Into<[T; 9]> for GMat3<T> {
    /// Receive a copy of the data as an array
    ///
    /// Can also use [`data()`](#method.data)
    fn into(self) -> [T; 9] {
        self.0
    }
}

impl<T: Scalar> AsRef<[T; 9]> for GMat3<T> {
    /// Receive a reference to the internal array
    fn as_ref(&self) -> &[T; 9] {
        &self.0
    }
}

impl<T: Scalar> AsMut<[T; 9]> for GMat3<T> {
    /// Receive a mutable reference to the internal array
    fn as_mut(&mut self) -> &mut [T; 9] {
        &mut self.0
    }
}

impl<T: Scalar> Default for GMat3<T> {

    /// Default for Mat3 is [`Mat3::identity()`](#method.identity). Consider using that function
    /// instead to be more explicit.
    fn default() -> Self {
        Self::identity()
    }
}
