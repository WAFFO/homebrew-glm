
use super::Vec3;

/** # Mat3 - 3x3 Matrix <f32>

Look at me, I'm describing things. What a time to be alive. What if this is really long? Like really really really really really really really really really really really really really really really really really really really really really really really really really really really really long?

## What if I break it up?
Look at me, I'm describing things. What a time to be alive. What if this is really long? Like really
really really really really really really really really really really really really really really
really really really really really really really really really really really really really long?
*/

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Mat3 ( pub(crate) [f32; 9] );

impl Mat3 {
    pub fn new(col1: Vec3, col2: Vec3, col3: Vec3) -> Mat3 {
        Mat3 ( [
            col1[0], col1[1], col1[2],
            col2[0], col2[1], col2[2],
            col3[0], col3[1], col3[2],
        ] )
    }
    pub fn zero() -> Mat3 { Mat3([0.0;9]) }
    pub fn identity() -> Mat3 {
        Mat3 ( [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ] )
    }
    pub fn mat3(mat: [f32;9]) -> Mat3 { Mat3(mat) }
    pub fn data(&self) -> [f32;9] { self.0 }
    pub fn data_ref(&self) -> &[f32;9] { &self.0 }
    pub fn data_ref_mut(&mut self) -> &mut [f32;9] { &mut self.0 }
}

impl std::ops::Index<usize> for Mat3 {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Mat3 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

impl std::ops::Index<(usize,usize)> for Mat3 {
    type Output = f32;

    fn index(&self, index: (usize,usize)) -> &f32 {
        &self.0[index.0 * 3 + index.1]
    }
}

impl std::ops::IndexMut<(usize,usize)> for Mat3 {
    fn index_mut(&mut self, index: (usize,usize)) -> &mut f32 {
        &mut self.0[index.0 * 3 + index.1]
    }
}

impl From<Vec3> for Mat3 {
    fn from(f: Vec3) -> Self {
        Mat3 ( [
            f[0],  0.0,  0.0,
            0.0, f[1],  0.0,
            0.0,  0.0, f[2],
        ] )
    }
}

impl std::ops::Mul<Mat3> for Mat3 {
    type Output = Mat3;

    fn mul(self, rhs: Mat3) -> Mat3 {
        let m1 = &self;
        let m2 = &rhs;
        Mat3 ( [
            m1[0]*m2[0]+m1[3]*m2[1]+m1[6]*m2[2], m1[1]*m2[0]+m1[4]*m2[1]+m1[7]*m2[2], m1[2]*m2[0]+m1[5]*m2[1]+m1[8]*m2[2],
            m1[0]*m2[3]+m1[3]*m2[4]+m1[6]*m2[5], m1[1]*m2[3]+m1[4]*m2[4]+m1[7]*m2[5], m1[2]*m2[3]+m1[5]*m2[4]+m1[8]*m2[5],
            m1[0]*m2[6]+m1[3]*m2[7]+m1[6]*m2[8], m1[1]*m2[6]+m1[4]*m2[7]+m1[7]*m2[8], m1[2]*m2[6]+m1[5]*m2[7]+m1[8]*m2[8],
        ] )
    }
}

impl std::ops::Mul<Vec3> for Mat3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3([
            rhs[0]*self[0] + rhs[1]*self[3] + rhs[2]*self[6],
            rhs[0]*self[1] + rhs[1]*self[4] + rhs[2]*self[7],
            rhs[0]*self[2] + rhs[1]*self[5] + rhs[2]*self[8],
        ])
    }
}