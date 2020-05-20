use crate::{NEAR_ZERO, GVec4};
use std::fmt::{Display, Formatter, Error};
use crate::traits::Scalar;

/** # Vec3 - 3 Dimensional Vector <f32>

 A Vector with three elements, `x`, `y`, and `z`, stored internally as `[f32; 3]`.

 #### Default

 [Default](https://doc.rust-lang.org/nightly/core/default/trait.Default.html) is implemented for
 ease of use with ECS libraries like [specs](https://docs.rs/specs/0.16.1/specs/) that require
 components to implement Default.

 [`Vec3::default()`](#method.default) is equivalent to [`Vec3::zero()`](#method.zero) and we
 recommend using that function instead to make your code more explicit.

*/
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GVec3<T: Scalar> ( pub(crate) [T; 3] );

/// # Vec3 - 3 Dimensional Vector <f32>
///
/// aaaaa
pub type Vec3 = GVec3<f32>;

/// Big boy Vec3 <f64>
pub type DVec3 = GVec3<f64>;

impl<T: Scalar> GVec3<T> {

    /// A unit Vec3 representing the positive X axis (in OpenGL this is right)
    pub const X_AXIS: Vec3 = GVec3( [ 1.0, 0.0, 0.0 ] );

    /// A unit Vec3 representing the positive Y axis (in OpenGL this is up)
    pub const Y_AXIS: Vec3 = GVec3( [ 0.0, 1.0, 0.0 ] );

    /// A unit Vec3 representing the positive Z axis (in OpenGL this is towards the screen)
    pub const Z_AXIS: Vec3 = GVec3( [ 0.0, 0.0, 1.0 ] );

    /// Create a new Vec3 with x, y, z components
    pub fn new(x: T, y: T, z: T) -> GVec3<T> { GVec3( [x, y, z] ) }

    /// Create a unit Vec3 with normalized values and a length of one
    pub fn unit(x: T, y: T, z: T) -> GVec3<T> { GVec3( [x, y, z] ).normalize() }

    /// Create an additive identity Vec3
    ///
    /// Equivalent to [`Vec3::new(0.0, 0.0, 0.0)`](#method.new)
    pub fn zero() -> GVec3<T> { GVec3( [T::zero(), T::zero(), T::zero()] ) }

    /// Create a multiplicative identity Vec3
    ///
    /// Equivalent to [`Vec3::new(1.0, 1.0, 1.0)`](#method.new)
    pub fn one()  -> GVec3<T> { GVec3( [T::one(), T::one(), T::one()] ) }

    /// Create a Vec3 with all the same values
    pub fn all(f: T) -> GVec3<T> { GVec3( [f, f, f] ) }

    /// Create a Vec3 from a 3 element array
    pub fn vec3(vec: [T;3]) -> GVec3<T> { GVec3( vec ) }

    /// Receive a copy of the array Vec3 represents
    ///
    /// Can also use [`into()`](#method.into)
    ///
    /// For a reference use [`as_ref()`](#method.as_ref) and for a mutable reference use [`as_mut()`](#method.as_mut)
    pub fn data(&self) -> [T;3] { self.0 }

    /// Receive the x value
    pub fn x(&self) -> T { self.0[0] }

    /// Receive the y value
    pub fn y(&self) -> T { self.0[1] }

    /// Receive the z value
    pub fn z(&self) -> T { self.0[2] }

    /// Receive the mutable reference for x
    pub fn x_mut(&mut self) -> &mut T { &mut self.0[0] }

    /// Receive the mutable reference for y
    pub fn y_mut(&mut self) -> &mut T { &mut self.0[1] }

    /// Receive the mutable reference for z
    pub fn z_mut(&mut self) -> &mut T { &mut self.0[2] }

    /// Create a Vec4 with a value for w
    pub fn vec4(&self, w: T) -> GVec4<T> { GVec4::vec3_w(*self, w) }

    /// Test if this Vec3 is equals to another Vec3 for each component up to 1e-6
    pub fn equals(&self, other: GVec3<T>) -> bool {
        self.equals_epsilon(other, T::cast(NEAR_ZERO))
    }

    /// Test if this Vec3 is equals to another Vec3 for each component up to an epsilon
    pub fn equals_epsilon(&self, other: GVec3<T>, epsilon: T) -> bool {
        (self.x() - other.x()).abs() <= epsilon
            && (self.y() - other.y()).abs() <= epsilon
            && (self.z() - other.z()).abs() <= epsilon
    }

    /// Receive the *absolute value* of each component in this Vec3
    pub fn abs(&self) -> GVec3<T> {
        GVec3([self[0].abs(), self[1].abs(), self[2].abs()])
    }

    /// Receive the aggregate average of each component
    pub fn agg_avg(&self) -> T {
        (self[0] + self[1] + self[2]) / T::cast(3.0)
    }

    /// Receive the aggregate max of each component
    pub fn agg_max(&self) -> T {
        self[0].max(self[1]).max(self[2])
    }

    /// Receive the aggregate min of each component
    pub fn agg_min(&self) -> T {
        self[0].min(self[1]).min(self[2])
    }

    /// Receive the aggregate product of each component
    pub fn agg_prod(&self) -> T {
        self[0] * self[1] * self[2]
    }

    /// Receive the aggregate sum of each component
    pub fn agg_sum(&self) -> T {
        self[0] + self[1] + self[2]
    }

    /// Receive the angle between this Vec3 and another Vec3
    pub fn angle(&self, other: GVec3<T>) -> T {
        let s = self.cross(other).length();
        let c = self.dot(other);
        s.atan2(c)
    }

    /// Receive a Vec3 with each component rounded up to the nearest integer
    pub fn ceil(&self) -> GVec3<T> {
        GVec3([
            self[0].ceil(),
            self[1].ceil(),
            self[2].ceil(),
        ])
    }

    /// Receive a Vec3 clamped at some minimum and some maximum
    pub fn clamp(&self, min: T, max: T) -> GVec3<T> {
        GVec3([
            if self[0] < min { min } else if self[0] > max { max } else { self [0] },
            if self[1] < min { min } else if self[1] > max { max } else { self [1] },
            if self[2] < min { min } else if self[2] > max { max } else { self [2] },
        ])
    }

    /// Receive the *dot* product of this Vec3 and another Vec3
    ///
    /// This function is commutative
    pub fn dot(&self, other: GVec3<T>) -> T {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }

    /// Receive the *cross* product of this Vec3 and another Vec3
    ///
    /// This function is **not** commutative
    pub fn cross(&self, other: GVec3<T>) -> GVec3<T> {
        GVec3([
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ])
    }

    /// Receive the magnitude of this Vec3
    ///
    /// This function is equivalent to [`length()`](#method.length)
    pub fn mag(&self) -> T { (
        self.x() * self.x() + self.y() * self.y() + self.z() * self.z()
    ).sqrt() }

    /// Receive the length of this Vec3
    ///
    /// This function is equivalent to [`mag()`](#method.mag)
    pub fn length(&self) -> T { self.mag() }

    /// Receive a normalization of Vec3
    ///
    /// ```rust
    /// # use sawd_glm::{Vec3, assert_eq_float};
    /// let v = Vec3::new(1.0, 2.0, -0.5).normalize();
    /// assert_eq_float!(1.0, v.length());
    /// ```
    pub fn normalize(&self) -> GVec3<T> {
        let mag = self.mag();
        if mag != T::zero() {
            *self / mag
        }
        else {
            *self
        }
    }

    /// Receive a Vec3 bounded by a float, overflows to zero
    pub fn bound(&self, bound: T) -> GVec3<T> {
        *self % bound
    }

    /// Receive a Vec3 that is perpendicular to this Vec3
    ///
    /// As there are infinite perpendicular Vectors for any given Vector, the result should be
    /// treated as arbitrary.
    pub fn perpendicular(&self) -> GVec3<T> {
        // prevents a trivial solution
        if self[2] < self[0] {
            GVec3::new(self[1], -self[0], T::zero())
        }
        else {
            GVec3::new(T::zero(), -self[2], self[1])
        }
    }

    /// ```
    /// # use sawd_glm::Vec3;
    /// let v = Vec3::all(1.5);
    /// assert!(v.is_perpendicular(v.perpendicular()));
    /// ```
    pub fn is_perpendicular(&self, other: GVec3<T>) -> bool {
        self.dot(other) < T::cast(NEAR_ZERO)
    }
}

// Operations
impl<T: Scalar> std::ops::Index<usize> for GVec3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.0[index]
    }
}


impl<T: Scalar> std::ops::IndexMut<usize> for GVec3<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.0[index]
    }
}

impl<T: Scalar> std::ops::Neg for GVec3<T> {
    type Output = GVec3<T>;

    fn neg(self) -> GVec3<T> {
        GVec3([
            -self.0[0],
            -self.0[1],
            -self.0[2],
        ])
    }
}

impl<T: Scalar> std::ops::Add for GVec3<T> {
    type Output = GVec3<T>;

    fn add(self, other: GVec3<T>) -> GVec3<T> {
        GVec3([
            self.0[0] + other.0[0],
            self.0[1] + other.0[1],
            self.0[2] + other.0[2],
        ])
    }
}

impl<T: Scalar> std::ops::AddAssign for GVec3<T> {
    fn add_assign(&mut self, other: GVec3<T>) {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
    }
}

impl<T: Scalar> std::ops::Sub for GVec3<T> {
    type Output = GVec3<T>;

    fn sub(self, other: GVec3<T>) -> GVec3<T> {
        GVec3([
            self.0[0] - other.0[0],
            self.0[1] - other.0[1],
            self.0[2] - other.0[2],
        ])
    }
}

impl<T: Scalar> std::ops::SubAssign for GVec3<T> {
    fn sub_assign(&mut self, other: GVec3<T>) {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
    }
}

impl<T: Scalar> std::ops::Mul<T> for GVec3<T> {
    type Output = GVec3<T>;

    fn mul(self, rhs: T) -> GVec3<T> {
        GVec3([
            self.0[0] * rhs,
            self.0[1] * rhs,
            self.0[2] * rhs,
        ])
    }
}

impl<T: Scalar> std::ops::MulAssign<T> for GVec3<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
    }
}

impl<T: Scalar> std::ops::Div<T> for GVec3<T> {
    type Output = GVec3<T>;

    fn div(self, rhs: T) -> GVec3<T> {
        if rhs == T::zero() { panic!("Cannot divide by zero. (Vec3 / 0.0)"); }
        GVec3( [
            self.0[0] / rhs,
            self.0[1] / rhs,
            self.0[2] / rhs,
        ])
    }
}

impl<T: Scalar> std::ops::DivAssign<T> for GVec3<T> {
    fn div_assign(&mut self, rhs: T) {
        if rhs == T::zero() { panic!("Cannot divide by zero. (Vec3 / 0.0)"); }
        self.0[0] /= rhs;
        self.0[1] /= rhs;
        self.0[2] /= rhs;
    }
}

impl<T: Scalar> std::ops::Rem<T> for GVec3<T> {
    type Output = GVec3<T>;

    fn rem(self, rhs: T) -> GVec3<T> {
        if rhs == T::zero() { panic!("Cannot divide by zero. (Vec3 % 0.0)"); }
        GVec3([
            self.0[0] % rhs,
            self.0[1] % rhs,
            self.0[2] % rhs,
        ])
    }
}

impl<T: Scalar> std::ops::RemAssign<T> for GVec3<T> {

    fn rem_assign(&mut self, rhs: T) {
        if rhs == T::zero() { panic!("Cannot divide by zero. (Vec3 % 0.0)"); }
        self.0[0] %= rhs;
        self.0[1] %= rhs;
        self.0[2] %= rhs;
    }
}

impl<T: Scalar> Default for GVec3<T> {

    /// Default for Vec3 is [`Vec3::zero()`](#method.zero). Consider using that function instead to
    /// be more explicit.
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: Scalar> Into<[T; 3]> for GVec3<T> {
    /// Receive a copy of the data as an array
    ///
    /// Can also use [`data()`](#method.data)
    fn into(self) -> [T; 3] {
        self.0
    }
}

impl<T: Scalar> AsRef<[T; 3]> for GVec3<T> {
    /// Receive a reference to the internal array
    fn as_ref(&self) -> &[T; 3] {
        &self.0
    }
}

impl<T: Scalar> AsMut<[T; 3]> for GVec3<T> {
    /// Receive a mutable reference to the internal array
    fn as_mut(&mut self) -> &mut [T; 3] {
        &mut self.0
    }
}

impl<T: Scalar> Display for GVec3<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "<{}, {}, {}>", self.x(), self.y(), self.z())
    }
}
