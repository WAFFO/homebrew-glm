
use super::Vec3;

/** # Vec4 - 4 Dimensional Vector <f32>

 format: [ x, y, z, w ]

*/

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Vec4 ( pub(crate) [f32; 4] );

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Vec4 { Vec4 ( [x, y, z, w] ) }
    pub fn zero() -> Vec4 { Vec4 ( [0.0, 0.0, 0.0, 0.0] ) }
    pub fn one()  -> Vec4 { Vec4 ( [1.0, 1.0, 1.0, 1.0] ) }
    pub fn all(f: f32) -> Vec4 { Vec4 ( [f, f, f, f] ) }
    pub fn vec3_w(v: Vec3, w: f32) -> Vec4 { Vec4 ( [v[0], v[1], v[2], w] ) }
    pub fn vec4(vec: [f32;4]) -> Vec4 { Vec4(vec) }
    pub fn data(&self) -> [f32;4] { self.0 }
    pub fn data_ref(&self) -> &[f32;4] { &self.0 }
    pub fn data_ref_mut(&mut self) -> &mut [f32;4] { &mut self.0 }
    pub fn x(&self) -> f32 { self.0[0] }
    pub fn y(&self) -> f32 { self.0[1] }
    pub fn z(&self) -> f32 { self.0[2] }
    pub fn w(&self) -> f32 { self.0[3] }
    pub fn xyz(&self) -> Vec3 { Vec3([self[0], self[1], self[2]]) }
    pub fn dot(&self, other: &Vec4) -> f32 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2] + self[3] * other[3]
    }
    /// Note! Cross in 4 dimensions doesn't exist, this is treated as a Vec3, and the 4th component
    /// is set to parameter w!
    pub fn cross(&self, other: &Vec4, w: f32) -> Vec4 {
        Vec4 ( [
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
            w,
        ] )
    }
    pub fn mag(&self) -> f32 { ( self[0].powi(2) + self[1].powi(2) + self[2].powi(2) + self[3].powi(2) ).sqrt() }
    pub fn length(&self) -> f32 { self.mag() }
    pub fn normalize(&self) -> Vec4 {
        let mag = self.mag();
        if mag != 0.0 {
            *self / mag
        }
        else {
            *self
        }
    }
    pub fn bound(&self, bound: f32) -> Vec4 {
        Vec4 ( [
            self[0] % bound,
            self[1] % bound,
            self[2] % bound,
            self[3] % bound,
        ] )
    }
}

impl From<Vec3> for Vec4 {
    fn from(f: Vec3) -> Self {
        Vec4 ( [f[0], f[1], f[2], 0.0] )
    }
}

vector_operations!(Vec4, { 0, 1, 2, 3 });
