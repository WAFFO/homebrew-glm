
use crate::{NEAR_ZERO, GVec3, GQuat};
use std::fmt::{Display, Formatter, Error};
use crate::traits::Scalar;

/** # Vec4 - 4 Dimensional Vector <f32>

 A Vector with four elements, `x`, `y`, `z`, and `w`, stored internally as `[f32; 4]`.

 #### Default

 [Default](https://doc.rust-lang.org/nightly/core/default/trait.Default.html) is implemented for
 ease of use with ECS libraries like [specs](https://docs.rs/specs/0.16.1/specs/) that require
 components to implement Default.

 [`Vec4::default()`](#method.default) is equivalent to [`Vec4::zero()`](#method.zero) and we
 recommend using that function instead to make your code more explicit.

*/

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GVec4<T: Scalar> ( pub(crate) [T; 4] );

/// # Vec4 - 4 Dimensional Vector <f32>
///
/// aaaaa
pub type Vec4 = GVec4<f32>;

/// Big boy Vec4 <f64>
pub type DVec4 = GVec4<f64>;

impl<T: Scalar> GVec4<T> {

    /// A unit Vec4 representing the positive X axis
    pub const X_AXIS: Vec4 = GVec4 ( [ 1.0, 0.0, 0.0, 0.0 ] );

    /// A unit Vec4 representing the positive Y axis
    pub const Y_AXIS: Vec4 = GVec4 ( [ 0.0, 1.0, 0.0, 0.0 ] );

    /// A unit Vec4 representing the positive Z axis
    pub const Z_AXIS: Vec4 = GVec4 ( [ 0.0, 0.0, 1.0, 0.0 ] );

    /// A unit Vec4 representing the positive W axis
    pub const W_AXIS: Vec4 = GVec4 ( [ 0.0, 0.0, 0.0, 1.0 ] );

    /// Create a new Vec4 with x, y, z, w components
    pub fn new(x: T, y: T, z: T, w: T) -> GVec4<T> { GVec4 ( [x, y, z, w] ) }

    /// Create a unit Vec4 with normalized values and a length of one
    pub fn unit(x: T, y: T, z: T, w: T) -> GVec4<T> { GVec4 ( [x, y, z, w] ).normalize() }

    /// Create an additive identity Vec4
    ///
    /// Equivalent to [`Vec4::new(0.0, 0.0, 0.0, 0.0)`](#method.new)
    pub fn zero() -> GVec4<T> { GVec4 ( [T::zero(), T::zero(), T::zero(), T::zero()] ) }

    /// Create a multiplicative identity Vec4
    ///
    /// Equivalent to [`Vec4::new(1.0, 1.0, 1.0, 1.0)`](#method.new)
    pub fn one()  -> GVec4<T> { GVec4 ( [T::one(), T::one(), T::one(), T::one()] ) }

    /// Create a Vec4 with all the same values
    pub fn all(f: T) -> GVec4<T> { GVec4 ( [f, f, f, f] ) }

    /// Create a Vec4 from a Vec3 and w
    pub fn vec3_w(v: GVec3<T>, w: T) -> GVec4<T> { GVec4 ( [v[0], v[1], v[2], w] ) }

    /// Create a Vec4 from a 4 element array
    pub fn vec4(vec: [T;4]) -> GVec4<T> { GVec4(vec) }

    /// Receive a copy of the array Vec4 represents
    ///
    /// Can also use [`into()`](#method.into)
    ///
    /// For a reference use [`as_ref()`](#method.as_ref) and for a mutable reference use [`as_mut()`](#method.as_mut)
    pub fn data(&self) -> [T;4] { self.0 }

    /// Receive the x value
    pub fn x(&self) -> T { self.0[0] }

    /// Receive the y value
    pub fn y(&self) -> T { self.0[1] }

    /// Receive the z value
    pub fn z(&self) -> T { self.0[2] }

    /// Receive the w value
    pub fn w(&self) -> T { self.0[3] }

    /// Receive the mutable reference for x
    pub fn x_mut(&mut self) -> &mut T { &mut self.0[0] }

    /// Receive the mutable reference for y
    pub fn y_mut(&mut self) -> &mut T { &mut self.0[1] }

    /// Receive the mutable reference for z
    pub fn z_mut(&mut self) -> &mut T { &mut self.0[2] }

    /// Receive the mutable reference for w
    pub fn w_mut(&mut self) -> &mut T { &mut self.0[3] }

    /// Receive a Vec3 made up of the x, y, z components
    pub fn xyz(&self) -> GVec3<T> { GVec3([self[0], self[1], self[2]]) }

    /// Test if this Vec4 is equals to another Vec4 for each component up to 1e-6
    pub fn equals(&self, other: GVec4<T>) -> bool {
        self.equals_epsilon(other, T::cast(NEAR_ZERO))
    }

    /// Test if this Vec4 is equals to another Vec4 for each component up to an epsilon
    pub fn equals_epsilon(&self, other: GVec4<T>, epsilon: T) -> bool {
        (self.x() - other.x()).abs() <= epsilon
            && (self.y() - other.y()).abs() <= epsilon
            && (self.z() - other.z()).abs() <= epsilon
            && (self.w() - other.w()).abs() <= epsilon
    }

    /// Receive the *absolute value* of each component in this Vec4
    pub fn abs(&self) -> GVec4<T> {
        GVec4([self[0].abs(), self[1].abs(), self[2].abs(), self[3].abs()])
    }

    /// Receive the aggregate average of each component
    pub fn agg_avg(&self) -> T {
        (self[0] + self[1] + self[2] + self[3]) / T::cast(4.0)
    }

    /// Receive the aggregate max of each component
    pub fn agg_max(&self) -> T {
        self[0].max(self[1]).max(self[2]).max(self[3])
    }

    /// Receive the aggregate min of each component
    pub fn agg_min(&self) -> T {
        self[0].min(self[1]).min(self[2]).min(self[3])
    }

    /// Receive the aggregate product of each component
    pub fn agg_prod(&self) -> T {
        self[0] * self[1] * self[2] * self[3]
    }

    /// Receive the aggregate sum of each component
    pub fn agg_sum(&self) -> T {
        self[0] + self[1] + self[2] + self[3]
    }

    /// Receive the angle between this Vec4 and another Vec4
    pub fn angle(&self, other: GVec4<T>) -> T {
        self.normalize().dot(other.normalize()).min(T::one()).max(-T::one()).acos()
    }

    /// Receive a Vec4 with each component rounded up to the nearest integer
    pub fn ceil(&self) -> GVec4<T> {
        GVec4([
            self[0].ceil(),
            self[1].ceil(),
            self[2].ceil(),
            self[3].ceil(),
        ])
    }

    /// Receive a Vec4 clamped at some minimum and some maximum
    pub fn clamp(&self, min: T, max: T) -> GVec4<T> {
        GVec4([
            if self[0] < min { min } else if self[0] > max { max } else { self [0] },
            if self[1] < min { min } else if self[1] > max { max } else { self [1] },
            if self[2] < min { min } else if self[2] > max { max } else { self [2] },
            if self[3] < min { min } else if self[3] > max { max } else { self [3] },
        ])
    }

    /// Receive the *dot* product of this Vec4 and another Vec4
    ///
    /// This function is commutative
    pub fn dot(&self, other: GVec4<T>) -> T {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2] + self[3] * other[3]
    }

    /// Receive the magnitude of this Vec4
    ///
    /// This function is equivalent to [`length()`](#method.length)
    pub fn mag(&self) -> T { (
        self.x() * self.x() + self.y() * self.y() + self.z() * self.z() + self.w() * self.w()
    ).sqrt() }

    /// Receive the length of this Vec4
    ///
    /// This function is equivalent to [`mag()`](#method.mag)
    pub fn length(&self) -> T { self.mag() }

    /// Receive a normalization of Vec4
    ///
    /// ```rust
    /// # use sawd_glm::{assert_eq_float, Vec4};
    /// let v = Vec4::new(1.0, 2.0, -0.5, 0.1).normalize();
    /// assert_eq_float!(1.0, v.length());
    /// ```
    pub fn normalize(&self) -> GVec4<T> {
        let mag = self.mag();
        if mag != T::zero() {
            *self / mag
        }
        else {
            *self
        }
    }

    /// Receive a Vec4 bounded by a float, overflows/underflows past the bound to zero
    pub fn bound(&self, bound: T) -> GVec4<T> {
        *self % bound
    }

    /// Receive a Vec4 that is perpendicular to this Vec4
    ///
    /// As there are infinite perpendicular Vectors for any given Vector, the result should be
    /// treated as arbitrary.
    pub fn perpendicular(&self) -> GVec4<T> {
        GVec4::new(self[1], -self[0], self[3], -self[2])
    }

    /// ```
    /// # use sawd_glm::Vec4;
    /// let v = Vec4::new(10.4, -10.4, -10.4, -10.4);
    /// assert!(v.is_perpendicular(v.perpendicular()));
    /// ```
    pub fn is_perpendicular(&self, other: GVec4<T>) -> bool {
        self.dot(other) < T::cast(NEAR_ZERO)
    }
}

impl<T: Scalar> From<GVec3<T>> for GVec4<T> {
    /// Create a Vec4 from a Vec3, w will be 0.0
    ///
    /// Equivalent to `Vec4::vec3_w(v, 0.0);`
    fn from(f: GVec3<T>) -> GVec4<T> { GVec4([ f[0], f[1], f[2], T::zero() ]) }
}

impl<T: Scalar> From<GQuat<T>> for GVec4<T> {
    /// Cast a Quat to a Vec4, no transformations
    fn from(f: GQuat<T>) -> GVec4<T> { GVec4([ f[0], f[1], f[2], f[3] ]) }
}

// Operations
impl<T: Scalar> std::ops::Index<usize> for GVec4<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.0[index]
    }
}


impl<T: Scalar> std::ops::IndexMut<usize> for GVec4<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.0[index]
    }
}

impl<T: Scalar> std::ops::Neg for GVec4<T> {
    type Output = GVec4<T>;

    fn neg(self) -> GVec4<T> {
        GVec4([
            -self.0[0],
            -self.0[1],
            -self.0[2],
            -self.0[3],
        ])
    }
}

impl<T: Scalar> std::ops::Add for GVec4<T> {
    type Output = GVec4<T>;

    fn add(self, other: GVec4<T>) -> GVec4<T> {
        GVec4([
            self.0[0] + other.0[0],
            self.0[1] + other.0[1],
            self.0[2] + other.0[2],
            self.0[3] + other.0[3],
        ])
    }
}

impl<T: Scalar> std::ops::AddAssign for GVec4<T> {
    fn add_assign(&mut self, other: GVec4<T>) {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self.0[3] += other.0[3];
    }
}

impl<T: Scalar> std::ops::Sub for GVec4<T> {
    type Output = GVec4<T>;

    fn sub(self, other: GVec4<T>) -> GVec4<T> {
        GVec4([
            self.0[0] - other.0[0],
            self.0[1] - other.0[1],
            self.0[2] - other.0[2],
            self.0[3] - other.0[3],
        ])
    }
}

impl<T: Scalar> std::ops::SubAssign for GVec4<T> {
    fn sub_assign(&mut self, other: GVec4<T>) {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
        self.0[3] -= other.0[3];
    }
}

impl<T: Scalar> std::ops::Mul<T> for GVec4<T> {
    type Output = GVec4<T>;

    fn mul(self, rhs: T) -> GVec4<T> {
        GVec4([
            self.0[0] * rhs,
            self.0[1] * rhs,
            self.0[2] * rhs,
            self.0[3] * rhs,
        ])
    }
}

impl<T: Scalar> std::ops::MulAssign<T> for GVec4<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
        self.0[3] *= rhs;
    }
}

impl<T: Scalar> std::ops::Div<T> for GVec4<T> {
    type Output = GVec4<T>;

    fn div(self, rhs: T) -> GVec4<T> {
        if rhs == T::zero() { panic!("Cannot divide by zero. (Vec4 / 0.0)"); }
        GVec4([
            self.0[0] / rhs,
            self.0[1] / rhs,
            self.0[2] / rhs,
            self.0[3] / rhs,
        ])
    }
}

impl<T: Scalar> std::ops::DivAssign<T> for GVec4<T> {
    fn div_assign(&mut self, rhs: T) {
        if rhs == T::zero() { panic!("Cannot divide by zero. (Vec4 / 0.0)"); }
        self.0[0] /= rhs;
        self.0[1] /= rhs;
        self.0[2] /= rhs;
        self.0[3] /= rhs;
    }
}

impl<T: Scalar> std::ops::Rem<T> for GVec4<T> {
    type Output = GVec4<T>;

    fn rem(self, rhs: T) -> GVec4<T> {
        if rhs == T::zero() { panic!("Cannot divide by zero. (Vec4 % 0.0)"); }
        GVec4([
            self.0[0] % rhs,
            self.0[1] % rhs,
            self.0[2] % rhs,
            self.0[3] % rhs,
        ])
    }
}

impl<T: Scalar> std::ops::RemAssign<T> for GVec4<T> {

    fn rem_assign(&mut self, rhs: T) {
        if rhs == T::zero() { panic!("Cannot divide by zero. (Vec4 % 0.0)"); }
        self.0[0] %= rhs;
        self.0[1] %= rhs;
        self.0[2] %= rhs;
        self.0[3] %= rhs;
    }
}
// end operations

impl<T: Scalar> Default for GVec4<T> {

    /// Default for Vec4 is [`Vec4::zero()`](#method.zero). Consider using that function instead to
    /// be more explicit.
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: Scalar> Into<[T; 4]> for GVec4<T> {
    /// Receive a copy of the data as an array
    ///
    /// Can also use [`data()`](#method.data)
    fn into(self) -> [T; 4] {
        self.0
    }
}

impl<T: Scalar> AsRef<[T; 4]> for GVec4<T> {
    /// Receive a reference to the internal array
    fn as_ref(&self) -> &[T; 4] {
        &self.0
    }
}

impl<T: Scalar> AsMut<[T; 4]> for GVec4<T> {
    /// Receive a mutable reference to the internal array
    fn as_mut(&mut self) -> &mut [T; 4] {
        &mut self.0
    }
}

impl<T: Scalar> Display for GVec4<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "<{}, {}, {}, {}>", self.x(), self.y(), self.z(), self.w())
    }
}
