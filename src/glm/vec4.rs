
use super::Vec3;

/** # Vec4 - 4 Dimensional Vector <f32>

 A Vector with four elements, `x`, `y`, 'z', and `w`, stored internally as `[f32; 4]`.

*/

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Vec4 ( pub(crate) [f32; 4] );

impl Vec4 {
    /// Create a new Vec4 with x, y, z, w components
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 { Vec4 ( [x, y, z, w] ) }

    /// Create an additive identity Vec4
    ///
    /// Equivalent to `Vec4::new(0.0, 0.0, 0.0, 0.0)`
    pub fn zero() -> Vec4 { Vec4 ( [0.0, 0.0, 0.0, 0.0] ) }

    /// Create a multiplicative identity Vec4
    ///
    /// Equivalent to `Vec4::new(1.0, 1.0, 1.0, 1.0)`
    pub fn one()  -> Vec4 { Vec4 ( [1.0, 1.0, 1.0, 1.0] ) }

    /// Create a Vec4 with all the same values
    pub fn all(f: f32) -> Vec4 { Vec4 ( [f, f, f, f] ) }

    /// Create a Vec4 from a Vec3 and w
    pub fn vec3_w(v: Vec3, w: f32) -> Vec4 { Vec4 ( [v[0], v[1], v[2], w] ) }

    /// Create a Vec4 from a 4 element array
    pub fn vec4(vec: [f32;4]) -> Vec4 { Vec4(vec) }

    /// Receive a copy of the array Vec4 represents
    pub fn data(&self) -> [f32;4] { self.0 }

    /// Receive a reference to the array within Vec4
    pub fn data_ref(&self) -> &[f32;4] { &self.0 }

    /// Receive a mutable reference to the array within Vec4
    pub fn data_ref_mut(&mut self) -> &mut [f32;4] { &mut self.0 }

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

    /// Receive the *dot* product of this Vec4 and another Vec4
    ///
    /// This function is commutative
    pub fn dot(&self, other: &Vec4) -> f32 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2] + self[3] * other[3]
    }

    /// Receive the magnitude of this Vec4
    ///
    /// This function is equivalent to `length()`
    pub fn mag(&self) -> f32 { (
        self.x() * self.x() + self.y() * self.y() + self.z() * self.z() + self.w() * self.w()
    ).sqrt() }

    /// Receive the length of this Vec4
    ///
    /// This function is equivalent to `mag()`
    pub fn length(&self) -> f32 { self.mag() }

    /// Receive a normalization of Vec3
    ///
    /// ```rust
    /// # use homebrew_glm::{assert_eq_float, Vec4};
    /// let v = Vec4::new(1.0, 2.0, -0.5, 0.1).normalize();
    /// assert_eq_float!(1.0, v.length())
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
    fn from(f: Vec3) -> Self {
        Vec4 ( [f[0], f[1], f[2], 0.0] )
    }
}

vector_operations!(Vec4, { 0, 1, 2, 3 });
