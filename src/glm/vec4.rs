
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

    /// Test if this Vec4 is equals to another Vec4 for each component up to 1e-6
    pub fn equals(&self, other: Vec4) -> bool {
        (self.x() - other.x()).abs() <= NEAR_ZERO
            && (self.y() - other.y()).abs() <= NEAR_ZERO
            && (self.z() - other.z()).abs() <= NEAR_ZERO
            && (self.w() - other.w()).abs() <= NEAR_ZERO
    }

    /// Receive the *dot* product of this Vec4 and another Vec4
    ///
    /// This function is commutative
    pub fn dot(&self, other: &Vec4) -> f32 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2] + self[3] * other[3]
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
    /// # use homebrew_glm::{assert_eq_float, Vec4};
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

vector_operations!(Vec4, { 0, 1, 2, 3 });

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
