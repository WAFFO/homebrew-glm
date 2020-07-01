
use crate::{NEAR_ZERO, Vec3, Quat};
use std::fmt::{Display, Formatter, Error};

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
pub struct Vec4 ( pub(crate) [f32; 4] );

impl Vec4 {

    /// A unit Vec4 representing the positive X axis
    pub const X_AXIS: Vec4 = Vec4 ( [ 1.0, 0.0, 0.0, 0.0 ] );

    /// A unit Vec4 representing the positive Y axis
    pub const Y_AXIS: Vec4 = Vec4 ( [ 0.0, 1.0, 0.0, 0.0 ] );

    /// A unit Vec4 representing the positive Z axis
    pub const Z_AXIS: Vec4 = Vec4 ( [ 0.0, 0.0, 1.0, 0.0 ] );

    /// A unit Vec4 representing the positive W axis
    pub const W_AXIS: Vec4 = Vec4 ( [ 0.0, 0.0, 0.0, 1.0 ] );

    /// Create a new Vec4 with x, y, z, w components
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 { Vec4 ( [x, y, z, w] ) }

    /// Create a unit Vec4 with normalized values and a length of one
    pub fn unit(x: f32, y: f32, z: f32, w: f32) -> Vec4 { Vec4 ( [x, y, z, w] ).normalize() }

    /// Create an additive identity Vec4
    ///
    /// Equivalent to [`Vec4::new(0.0, 0.0, 0.0, 0.0)`](#method.new)
    pub fn zero() -> Vec4 { Vec4 ( [0.0, 0.0, 0.0, 0.0] ) }

    /// Create a multiplicative identity Vec4
    ///
    /// Equivalent to [`Vec4::new(1.0, 1.0, 1.0, 1.0)`](#method.new)
    pub fn one()  -> Vec4 { Vec4 ( [1.0, 1.0, 1.0, 1.0] ) }

    /// Create a Vec4 with all the same values
    pub fn all(f: f32) -> Vec4 { Vec4 ( [f, f, f, f] ) }

    /// Create a Vec4 from a Vec3 and w
    pub fn vec3_w(v: Vec3, w: f32) -> Vec4 { Vec4 ( [v[0], v[1], v[2], w] ) }

    /// Create a Vec4 from a 4 element array
    pub fn vec4(vec: [f32;4]) -> Vec4 { Vec4(vec) }

    /// Receive a copy of the array Vec4 represents
    ///
    /// Can also use [`into()`](#method.into)
    ///
    /// For a reference use [`as_ref()`](#method.as_ref) and for a mutable reference use [`as_mut()`](#method.as_mut)
    pub fn data(&self) -> [f32;4] { self.0 }

    /// Receive the x value
    pub fn x(&self) -> f32 { self.0[0] }

    /// Receive the y value
    pub fn y(&self) -> f32 { self.0[1] }

    /// Receive the z value
    pub fn z(&self) -> f32 { self.0[2] }

    /// Receive the w value
    pub fn w(&self) -> f32 { self.0[3] }

    /// Receive the mutable reference for x
    pub fn x_mut(&mut self) -> &mut f32 { &mut self.0[0] }

    /// Receive the mutable reference for y
    pub fn y_mut(&mut self) -> &mut f32 { &mut self.0[1] }

    /// Receive the mutable reference for z
    pub fn z_mut(&mut self) -> &mut f32 { &mut self.0[2] }

    /// Receive the mutable reference for w
    pub fn w_mut(&mut self) -> &mut f32 { &mut self.0[3] }

    /// Receive a Vec3 made up of the x, y, z components
    pub fn xyz(&self) -> Vec3 { Vec3([self[0], self[1], self[2]]) }

    /// Test if this Vec4 is equals to another Vec4 for each component up to an epsilon of 1e-6
    pub fn equals(&self, other: Vec4) -> bool {
        self.equals_epsilon(other, NEAR_ZERO)
    }

    /// Test if this Vec4 is equals to another Vec4 for each component up to an epsilon
    pub fn equals_epsilon(&self, other: Vec4, epsilon: f32) -> bool {
        (self.x() - other.x()).abs() <= epsilon
            && (self.y() - other.y()).abs() <= epsilon
            && (self.z() - other.z()).abs() <= epsilon
            && (self.w() - other.w()).abs() <= epsilon
    }

    /// Receive the *absolute value* of each component in this Vec4
    pub fn abs(&self) -> Vec4 {
        Vec4([self[0].abs(), self[1].abs(), self[2].abs(), self[3].abs()])
    }

    /// Receive the aggregate average of each component
    pub fn agg_avg(&self) -> f32 {
        (self[0] + self[1] + self[2] + self[3]) / 4.0
    }

    /// Receive the aggregate max of each component
    pub fn agg_max(&self) -> f32 {
        self[0].max(self[1]).max(self[2]).max(self[3])
    }

    /// Receive the aggregate min of each component
    pub fn agg_min(&self) -> f32 {
        self[0].min(self[1]).min(self[2]).min(self[3])
    }

    /// Receive the aggregate product of each component
    pub fn agg_prod(&self) -> f32 {
        self[0] * self[1] * self[2] * self[3]
    }

    /// Receive the aggregate sum of each component
    pub fn agg_sum(&self) -> f32 {
        self[0] + self[1] + self[2] + self[3]
    }

    /// Receive the angle between this Vec4 and another Vec4
    pub fn angle(&self, other: Vec4) -> f32 {
        self.normalize().dot(other.normalize()).min(1.0).max(-1.0).acos()
    }

    /// Receive a Vec4 with each component rounded up to the nearest integer
    pub fn ceil(&self) -> Vec4 {
        Vec4([
            self[0].ceil(),
            self[1].ceil(),
            self[2].ceil(),
            self[3].ceil(),
        ])
    }

    /// Receive a Vec4 clamped at some minimum and some maximum
    pub fn clamp(&self, min: f32, max: f32) -> Vec4 {
        Vec4([
            if self[0] < min { min } else if self[0] > max { max } else { self [0] },
            if self[1] < min { min } else if self[1] > max { max } else { self [1] },
            if self[2] < min { min } else if self[2] > max { max } else { self [2] },
            if self[3] < min { min } else if self[3] > max { max } else { self [3] },
        ])
    }

    /// Receive the *distance* from this Vec4 to another Vec4
    pub fn distance(&self, other: Vec4) -> f32 {
        let x = other[0] - self[0];
        let y = other[1] - self[1];
        let z = other[2] - self[2];
        let w = other[3] - self[3];
        (x * x + y * y + z * z + w * w).sqrt()
    }

    /// Receive the *distance squared* from this Vec4 to another Vec4
    ///
    /// One less operation than [`distance()`](#method.into)
    pub fn distance2(&self, other: Vec4) -> f32 {
        let x = other[0] - self[0];
        let y = other[1] - self[1];
        let z = other[2] - self[2];
        let w = other[3] - self[3];
        x * x + y * y + z * z + w * w
    }

    /// Receive the *dot* product of this Vec4 and another Vec4
    ///
    /// This function is commutative
    pub fn dot(&self, other: Vec4) -> f32 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2] + self[3] * other[3]
    }

    /// Receive a Vec4 with each component rounded down to the nearest integer
    pub fn floor(&self) -> Vec4 {
        Vec4([
            self[0].floor(),
            self[1].floor(),
            self[2].floor(),
            self[3].floor(),
        ])
    }

    /// Receive a Vec4 with only the fractional portion of each component
    pub fn fract(&self) -> Vec4 {
        Vec4([
            self[0].fract(),
            self[1].fract(),
            self[2].fract(),
            self[3].fract(),
        ])
    }

    /// Receive the magnitude of this Vec4
    ///
    /// This function is equivalent to [`length()`](#method.length)
    pub fn mag(&self) -> f32 { (
        self.x() * self.x() + self.y() * self.y() + self.z() * self.z() + self.w() * self.w()
    ).sqrt() }

    /// Receive the length of this Vec4
    ///
    /// This function is equivalent to [`mag()`](#method.mag)
    pub fn length(&self) -> f32 { self.mag() }

    /// Receive a normalization of Vec4
    ///
    /// ```rust
    /// # use sawd_glm::{assert_eq_float, Vec4};
    /// let v = Vec4::new(1.0, 2.0, -0.5, 0.1).normalize();
    /// assert_eq_float!(1.0, v.length());
    /// ```
    pub fn normalize(&self) -> Vec4 {
        let mag = self.mag();
        if mag != 0.0 {
            *self / mag
        }
        else {
            *self
        }
    }

    /// Receive a Vec4 bounded by a float, overflows/underflows past the bound to zero
    pub fn bound(&self, bound: f32) -> Vec4 {
        *self % bound
    }

    /// Receive a Vec4 that is perpendicular to this Vec4
    ///
    /// As there are infinite perpendicular Vectors for any given Vector, the result should be
    /// treated as arbitrary.
    pub fn perpendicular(&self) -> Vec4 {
        Vec4::new(self[1], -self[0], self[3], -self[2])
    }

    /// ```
    /// # use sawd_glm::Vec4;
    /// let v = Vec4::new(10.4, -10.4, -10.4, -10.4);
    /// assert!(v.is_perpendicular(v.perpendicular()));
    /// ```
    pub fn is_perpendicular(&self, other: Vec4) -> bool {
        self.dot(other) < NEAR_ZERO
    }
}

impl From<Vec3> for Vec4 {
    /// Create a Vec4 from a Vec3, w will be 0.0
    ///
    /// Equivalent to `Vec4::vec3_w(v, 0.0);`
    fn from(f: Vec3) -> Vec4 { Vec4 ( [f[0], f[1], f[2], 0.0] ) }
}

impl From<Quat> for Vec4 {
    /// Cast a Quat to a Vec4, no transformations
    fn from(f: Quat) -> Vec4 { Vec4([ f[0], f[1], f[2], f[3] ]) }
}

impl std::ops::Index<usize> for Vec4 {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}


impl std::ops::IndexMut<usize> for Vec4 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

impl std::ops::Neg for Vec4 {
    type Output = Vec4;

    fn neg(self) -> Vec4 {
        Vec4([
            -self.0[0],
            -self.0[1],
            -self.0[2],
            -self.0[3],
        ])
    }
}

impl std::ops::Add for Vec4 {
    type Output = Vec4;

    fn add(self, other: Vec4) -> Vec4 {
        Vec4 ( [
            self.0[0] + other.0[0],
            self.0[1] + other.0[1],
            self.0[2] + other.0[2],
            self.0[3] + other.0[3],
        ] )
    }
}

impl std::ops::AddAssign for Vec4 {
    fn add_assign(&mut self, other: Vec4) {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self.0[3] += other.0[3];
    }
}

impl std::ops::Sub for Vec4 {
    type Output = Vec4;

    fn sub(self, other: Vec4) -> Vec4 {
        Vec4 ( [
            self.0[0] - other.0[0],
            self.0[1] - other.0[1],
            self.0[2] - other.0[2],
            self.0[3] - other.0[3],
        ] )
    }
}

impl std::ops::SubAssign for Vec4 {
    fn sub_assign(&mut self, other: Vec4) {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
        self.0[3] -= other.0[3];
    }
}

impl std::ops::Mul<f32> for Vec4 {
    type Output = Vec4;

    fn mul(self, rhs: f32) -> Vec4 {
        Vec4 ( [
            self.0[0] * rhs,
            self.0[1] * rhs,
            self.0[2] * rhs,
            self.0[3] * rhs,
        ] )
    }
}

impl std::ops::MulAssign<f32> for Vec4 {
    fn mul_assign(&mut self, rhs: f32) {
        self.0[0] *= rhs;
        self.0[1] *= rhs;
        self.0[2] *= rhs;
        self.0[3] *= rhs;
    }
}

impl std::ops::Div<f32> for Vec4 {
    type Output = Vec4;

    fn div(self, rhs: f32) -> Vec4 {
        if rhs == 0.0 { panic!("Cannot divide by zero. (Vec4 / 0.0)"); }
        Vec4 ( [
            self.0[0] / rhs,
            self.0[1] / rhs,
            self.0[2] / rhs,
            self.0[3] / rhs,
        ] )
    }
}

impl std::ops::DivAssign<f32> for Vec4 {
    fn div_assign(&mut self, rhs: f32) {
        if rhs == 0.0 { panic!("Cannot divide by zero. (Vec4 / 0.0)"); }
        self.0[0] /= rhs;
        self.0[1] /= rhs;
        self.0[2] /= rhs;
        self.0[3] /= rhs;
    }
}

impl std::ops::Rem<f32> for Vec4 {
    type Output = Vec4;

    fn rem(self, rhs: f32) -> Vec4 {
        if rhs == 0.0 { panic!("Cannot divide by zero. (Vec4 % 0.0)"); }
        Vec4 ( [
            self.0[0] % rhs,
            self.0[1] % rhs,
            self.0[2] % rhs,
            self.0[3] % rhs,
        ] )
    }
}

impl std::ops::RemAssign<f32> for Vec4 {

    fn rem_assign(&mut self, rhs: f32) {
        if rhs == 0.0 { panic!("Cannot divide by zero. (Vec4 % 0.0)"); }
        self.0[0] %= rhs;
        self.0[1] %= rhs;
        self.0[2] %= rhs;
        self.0[3] %= rhs;
    }
}

impl Default for Vec4 {

    /// Default for Vec4 is [`Vec4::zero()`](#method.zero). Consider using that function instead to
    /// be more explicit.
    fn default() -> Self {
        Self::zero()
    }
}

impl Into<[f32; 4]> for Vec4 {
    /// Receive a copy of the data as an array
    ///
    /// Can also use [`data()`](#method.data)
    fn into(self) -> [f32; 4] {
        self.0
    }
}

impl AsRef<[f32; 4]> for Vec4 {
    /// Receive a reference to the internal array
    fn as_ref(&self) -> &[f32; 4] {
        &self.0
    }
}

impl AsMut<[f32; 4]> for Vec4 {
    /// Receive a mutable reference to the internal array
    fn as_mut(&mut self) -> &mut [f32; 4] {
        &mut self.0
    }
}

impl Display for Vec4 {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "<{}, {}, {}, {}>", self.x(), self.y(), self.z(), self.w())
    }
}
