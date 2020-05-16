use crate::{NEAR_ZERO, Vec4};
use std::fmt::{Display, Formatter, Error};

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
pub struct Vec3 ( pub(crate) [f32; 3] );

impl Vec3 {

    /// A unit Vec3 representing the positive X axis (in OpenGL this is right)
    pub const X_AXIS: Vec3 = Vec3 ( [ 1.0, 0.0, 0.0 ] );

    /// A unit Vec3 representing the positive Y axis (in OpenGL this is up)
    pub const Y_AXIS: Vec3 = Vec3 ( [ 0.0, 1.0, 0.0 ] );

    /// A unit Vec3 representing the positive Z axis (in OpenGL this is towards the screen)
    pub const Z_AXIS: Vec3 = Vec3 ( [ 0.0, 0.0, 1.0 ] );

    /// Create a new Vec3 with x, y, z components
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 { Vec3 ( [x, y, z] ) }

    /// Create a unit Vec3 with normalized values and a length of one
    pub fn unit(x: f32, y: f32, z: f32) -> Vec3 { Vec3 ( [x, y, z] ).normalize() }

    /// Create an additive identity Vec3
    ///
    /// Equivalent to [`Vec3::new(0.0, 0.0, 0.0)`](#method.new)
    pub fn zero() -> Vec3 { Vec3 ( [0.0, 0.0, 0.0] ) }

    /// Create a multiplicative identity Vec3
    ///
    /// Equivalent to [`Vec3::new(1.0, 1.0, 1.0)`](#method.new)
    pub fn one()  -> Vec3 { Vec3 ( [1.0, 1.0, 1.0] ) }

    /// Create a Vec3 with all the same values
    pub fn all(f: f32) -> Vec3 { Vec3 ( [f, f, f] ) }

    /// Create a Vec3 from a 3 element array
    pub fn vec3(vec: [f32;3]) -> Vec3 { Vec3(vec) }

    /// Receive a copy of the array Vec3 represents
    ///
    /// Can also use [`into()`](#method.into)
    ///
    /// For a reference use [`as_ref()`](#method.as_ref) and for a mutable reference use [`as_mut()`](#method.as_mut)
    pub fn data(&self) -> [f32;3] { self.0 }

    /// Receive the x value
    pub fn x(&self) -> f32 { self.0[0] }

    /// Receive the y value
    pub fn y(&self) -> f32 { self.0[1] }

    /// Receive the z value
    pub fn z(&self) -> f32 { self.0[2] }

    /// Receive the mutable reference for x
    pub fn x_mut(&mut self) -> &mut f32 { &mut self.0[0] }

    /// Receive the mutable reference for y
    pub fn y_mut(&mut self) -> &mut f32 { &mut self.0[1] }

    /// Receive the mutable reference for z
    pub fn z_mut(&mut self) -> &mut f32 { &mut self.0[2] }

    /// Create a Vec4 with a value for w
    pub fn vec4(&self, w: f32) -> Vec4 { Vec4::vec3_w(*self, w) }

    /// Test if this Vec3 is equals to another Vec3 for each component up to 1e-6
    pub fn equals(&self, other: Vec3) -> bool {
        self.equals_epsilon(other, NEAR_ZERO)
    }

    /// Test if this Vec3 is equals to another Vec3 for each component up to an epsilon
    pub fn equals_epsilon(&self, other: Vec3, epsilon: f32) -> bool {
        (self.x() - other.x()).abs() <= epsilon
            && (self.y() - other.y()).abs() <= epsilon
            && (self.z() - other.z()).abs() <= epsilon
    }

    /// Receive the *absolute value* of each component in this Vec3
    pub fn abs(&self) -> Vec3 {
        Vec3([self[0].abs(), self[1].abs(), self[2].abs()])
    }

    /// Receive the aggregate average of each component
    pub fn agg_avg(&self) -> f32 {
        (self[0] + self[1] + self[2]) / 3.0
    }

    /// Receive the aggregate max of each component
    pub fn agg_max(&self) -> f32 {
        self[0].max(self[1]).max(self[2])
    }

    /// Receive the aggregate min of each component
    pub fn agg_min(&self) -> f32 {
        self[0].min(self[1]).min(self[2])
    }

    /// Receive the aggregate product of each component
    pub fn agg_prod(&self) -> f32 {
        self[0] * self[1] * self[2]
    }

    /// Receive the aggregate sum of each component
    pub fn agg_sum(&self) -> f32 {
        self[0] + self[1] + self[2]
    }

    /// Receive the angle between this Vec3 and another Vec3
    pub fn angle(&self, other: Vec3) -> f32 {
        let s = self.cross(other).length();
        let c = self.dot(other);
        s.atan2(c)
    }

    /// Receive a Vec3 with each component rounded up to the nearest integer
    pub fn ceil(&self) -> Vec3 {
        Vec3([
            self[0].ceil(),
            self[1].ceil(),
            self[2].ceil(),
        ])
    }

    /// Receive a Vec3 clamped at some minimum and some maximum
    pub fn clamp(&self, min: f32, max: f32) -> Vec3 {
        Vec3([
            if self[0] < min { min } else if self[0] > max { max } else { self [0] },
            if self[1] < min { min } else if self[1] > max { max } else { self [1] },
            if self[2] < min { min } else if self[2] > max { max } else { self [2] },
        ])
    }

    /// Receive the *dot* product of this Vec3 and another Vec3
    ///
    /// This function is commutative
    pub fn dot(&self, other: Vec3) -> f32 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }

    /// Receive the *cross* product of this Vec3 and another Vec3
    ///
    /// This function is **not** commutative
    pub fn cross(&self, other: Vec3) -> Vec3 {
        Vec3 ( [
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ] )
    }

    /// Receive the magnitude of this Vec3
    ///
    /// This function is equivalent to [`length()`](#method.length)
    pub fn mag(&self) -> f32 { (
        self.x() * self.x() + self.y() * self.y() + self.z() * self.z()
    ).sqrt() }

    /// Receive the length of this Vec3
    ///
    /// This function is equivalent to [`mag()`](#method.mag)
    pub fn length(&self) -> f32 { self.mag() }

    /// Receive a normalization of Vec3
    ///
    /// ```rust
    /// # use sawd_glm::{Vec3, assert_eq_float};
    /// let v = Vec3::new(1.0, 2.0, -0.5).normalize();
    /// assert_eq_float!(1.0, v.length());
    /// ```
    pub fn normalize(&self) -> Vec3 {
        let mag = self.mag();
        if mag != 0.0 {
            *self / mag
        }
        else {
            *self
        }
    }

    /// Receive a Vec3 bounded by a float, overflows to zero
    pub fn bound(&self, bound: f32) -> Vec3 {
        *self % bound
    }

    /// Receive a Vec3 that is perpendicular to this Vec3
    ///
    /// As there are infinite perpendicular Vectors for any given Vector, the result should be
    /// treated as arbitrary.
    pub fn perpendicular(&self) -> Vec3 {
        if self[2] < self[0] {
            Vec3::new(self[1],-self[0],0.0)
        }
        else {
            Vec3::new(0.0,-self[2],self[1])
        }
    }

    /// ```
    /// # use sawd_glm::Vec3;
    /// let v = Vec3::all(1.5);
    /// assert!(v.is_perpendicular(v.perpendicular()));
    /// ```
    pub fn is_perpendicular(&self, other: Vec3) -> bool {
        self.dot(other) < NEAR_ZERO
    }
}

vector_operations!(Vec3, { 0, 1, 2 } );

impl Default for Vec3 {

    /// Default for Vec3 is [`Vec3::zero()`](#method.zero). Consider using that function instead to
    /// be more explicit.
    fn default() -> Self {
        Self::zero()
    }
}

impl Into<[f32; 3]> for Vec3 {
    /// Receive a copy of the data as an array
    ///
    /// Can also use [`data()`](#method.data)
    fn into(self) -> [f32; 3] {
        self.0
    }
}

impl AsRef<[f32; 3]> for Vec3 {
    /// Receive a reference to the internal array
    fn as_ref(&self) -> &[f32; 3] {
        &self.0
    }
}

impl AsMut<[f32; 3]> for Vec3 {
    /// Receive a mutable reference to the internal array
    fn as_mut(&mut self) -> &mut [f32; 3] {
        &mut self.0
    }
}

impl Display for Vec3 {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "<{}, {}, {}>", self.x(), self.y(), self.z())
    }
}
