use crate::NEAR_ZERO;
use std::ops::Neg;

/** # Vec3 - 3 Dimensional Vector <f32>

 A Vector with three elements, `x`, `y`, and `z`, stored internally as `[f32; 3]`.

*/

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Vec3 ( pub(crate) [f32; 3] );

impl Vec3 {
    /// Create a new Vec3 with x, y, z components
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 { Vec3 ( [x, y, z] ) }

    /// Create an additive identity Vec3
    ///
    /// Equivalent to `Vec3::new(0.0, 0.0, 0.0)`
    pub fn zero() -> Vec3 { Vec3 ( [0.0, 0.0, 0.0] ) }

    /// Create a multiplicative identity Vec3
    ///
    /// Equivalent to `Vec3::new(1.0, 1.0, 1.0)`
    pub fn one()  -> Vec3 { Vec3 ( [1.0, 1.0, 1.0] ) }

    /// Create a Vec3 with all the same values
    pub fn all(f: f32) -> Vec3 { Vec3 ( [f, f, f] ) }

    /// Create a Vec3 from a 3 element array
    pub fn vec3(vec: [f32;3]) -> Vec3 { Vec3(vec) }

    /// Receive a copy of the array Vec3 represents
    pub fn data(&self) -> [f32;3] { self.0 }

    /// Receive a reference to the array within Vec3
    pub fn data_ref(&self) -> &[f32;3] { &self.0 }

    /// Receive a mutable reference to the array within Vec3
    pub fn data_ref_mut(&mut self) -> &mut [f32;3] { &mut self.0 }

    /// Receive the x value
    pub fn x(&self) -> f32 { self.0[0] }

    /// Receive the y value
    pub fn y(&self) -> f32 { self.0[1] }

    /// Receive the z value
    pub fn z(&self) -> f32 { self.0[2] }

    /// Receive the mutable reference for x
    pub fn x_mut(&mut self) -> &mut f32 { &mut self[0] }

    /// Receive the mutable reference for y
    pub fn y_mut(&mut self) -> &mut f32 { &mut self[1] }

    /// Receive the mutable reference for z
    pub fn z_mut(&mut self) -> &mut f32 { &mut self[2] }

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
    /// This function is analogous to `length()`
    pub fn mag(&self) -> f32 { ( self.x() * self.x() + self.y() * self.y() + self.z() * self.z() ).sqrt() }

    /// Receive the length of this Vec3
    ///
    /// This function is analogous to `mag()`
    pub fn length(&self) -> f32 { self.mag() }

    /// Receive a normalization of Vec3
    ///
    /// ```rust
    /// # use homebrew_glm::{Vec3, assert_eq_float};
    /// let v = Vec3::new(1.0, 2.0, -0.5).normalize();
    /// assert_eq_float!(1.0, v.length())
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

    /// Receive a Vec3 bounded by a float
    pub fn bound(&self, bound: f32) -> Vec3 {
        self.clone() % bound
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
    /// # use homebrew_glm::Vec3;
    /// let v = Vec3::all(1.5);
    /// assert!(v.is_perpendicular(v.perpendicular()));
    /// ```
    pub fn is_perpendicular(&self, other: Vec3) -> bool {
        self.dot(other) < NEAR_ZERO
    }
}

vector_operations!(Vec3, { 0, 1, 2 } );
