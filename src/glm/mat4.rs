
use super::{Vec4, Mat3};

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Mat4 ( pub(crate) [f32; 16] );

impl Mat4 {
    pub fn new(col1: Vec4, col2: Vec4, col3: Vec4, col4: Vec4) -> Mat4 {
        Mat4 ( [
            col1[0], col1[1], col1[2], col1[3],
            col2[0], col2[1], col2[2], col2[3],
            col3[0], col3[1], col3[2], col3[3],
            col4[0], col4[1], col4[2], col4[3],
        ] )
    }
    pub fn zero() -> Mat4 { Mat4([0.0;16]) }
    pub fn identity() -> Mat4 {
        Mat4 ( [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] )
    }
    pub fn mat4(mat: [f32;16]) -> Mat4 { Mat4(mat) }
    pub fn data(&self) -> [f32;16] { self.0 }
    pub fn data_ref(&self) -> &[f32;16] { &self.0 }
    pub fn data_ref_mut(&mut self) -> &mut [f32;16] { &mut self.0 }
    pub fn get(&self, col: usize, row: usize) -> &f32 {
        &self[(col, row)]
    }
    pub fn get_mut(&mut self, col: usize, row: usize) -> &mut f32 {
        &mut self[(col, row)]
    }
}

impl std::ops::Index<usize> for Mat4 {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Mat4 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

impl std::ops::Index<(usize,usize)> for Mat4 {
    type Output = f32;

    fn index(&self, index: (usize,usize)) -> &f32 {
        &self.0[index.0 * 4 + index.1]
    }
}

impl std::ops::IndexMut<(usize,usize)> for Mat4 {
    fn index_mut(&mut self, index: (usize,usize)) -> &mut f32 {
        &mut self.0[index.0 * 4 + index.1]
    }
}

impl From<Mat3> for Mat4 {
    fn from(f: Mat3) -> Self {
        Mat4 ( [
            f[0], f[1], f[2],  0.0,
            f[3], f[4], f[5],  0.0,
            f[6], f[7], f[8],  0.0,
            0.0,  0.0,  0.0,  0.0,
        ] )
    }
}

impl From<Vec4> for Mat4 {
    fn from(f: Vec4) -> Self {
        Mat4 ( [
            f[0],  0.0,  0.0,  0.0,
            0.0, f[1],  0.0,  0.0,
            0.0,  0.0, f[2],  0.0,
            0.0,  0.0,  0.0, f[3],
        ] )
    }
}

impl std::ops::Mul<Mat4> for Mat4 {
    type Output = Mat4;

    fn mul(self, rhs: Mat4) -> Mat4 {
        let m1 = &self;
        let m2 = &rhs;
        Mat4 ( [
            m1[0]*m2[ 0]+m1[4]*m2[ 1]+m1[ 8]*m2[ 2]+m1[12]*m2[ 3],
            m1[1]*m2[ 0]+m1[5]*m2[ 1]+m1[ 9]*m2[ 2]+m1[13]*m2[ 3],
            m1[2]*m2[ 0]+m1[6]*m2[ 1]+m1[10]*m2[ 2]+m1[14]*m2[ 3],
            m1[3]*m2[ 0]+m1[7]*m2[ 1]+m1[11]*m2[ 2]+m1[15]*m2[ 3],

            m1[0]*m2[ 4]+m1[4]*m2[ 5]+m1[ 8]*m2[ 6]+m1[12]*m2[ 7],
            m1[1]*m2[ 4]+m1[5]*m2[ 5]+m1[ 9]*m2[ 6]+m1[13]*m2[ 7],
            m1[2]*m2[ 4]+m1[6]*m2[ 5]+m1[10]*m2[ 6]+m1[14]*m2[ 7],
            m1[3]*m2[ 4]+m1[7]*m2[ 5]+m1[11]*m2[ 6]+m1[15]*m2[ 7],

            m1[0]*m2[ 8]+m1[4]*m2[ 9]+m1[ 8]*m2[10]+m1[12]*m2[11],
            m1[1]*m2[ 8]+m1[5]*m2[ 9]+m1[ 9]*m2[10]+m1[13]*m2[11],
            m1[2]*m2[ 8]+m1[6]*m2[ 9]+m1[10]*m2[10]+m1[14]*m2[11],
            m1[3]*m2[ 8]+m1[7]*m2[ 9]+m1[11]*m2[10]+m1[15]*m2[11],

            m1[0]*m2[12]+m1[4]*m2[13]+m1[ 8]*m2[14]+m1[12]*m2[15],
            m1[1]*m2[12]+m1[5]*m2[13]+m1[ 9]*m2[14]+m1[13]*m2[15],
            m1[2]*m2[12]+m1[6]*m2[13]+m1[10]*m2[14]+m1[14]*m2[15],
            m1[3]*m2[12]+m1[7]*m2[13]+m1[11]*m2[14]+m1[15]*m2[15],
        ] )
    }
}

impl std::ops::Mul<Vec4> for Mat4 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Vec4 {
        Vec4([
            rhs[0]*self[0] + rhs[1]*self[4] + rhs[2]*self[ 8] + rhs[3]*self[12],
            rhs[0]*self[1] + rhs[1]*self[5] + rhs[2]*self[ 9] + rhs[3]*self[13],
            rhs[0]*self[2] + rhs[1]*self[6] + rhs[2]*self[10] + rhs[3]*self[14],
            rhs[0]*self[3] + rhs[1]*self[7] + rhs[2]*self[11] + rhs[3]*self[15],
        ])
    }
}