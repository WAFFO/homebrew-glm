


#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Vec3 ( pub(crate) [f32; 3] );

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 { Vec3 ( [x, y, z] ) }
    pub fn zero() -> Vec3 { Vec3 ( [0.0, 0.0, 0.0] ) }
    pub fn one()  -> Vec3 { Vec3 ( [1.0, 1.0, 1.0] ) }
    pub fn all(f: f32) -> Vec3 { Vec3 ( [f, f, f] ) }
    pub fn vec3(vec: [f32;3]) -> Vec3 { Vec3(vec) }
    pub fn data(&self) -> [f32;3] { self.0 }
    pub fn data_ref(&self) -> &[f32;3] { &self.0 }
    pub fn data_ref_mut(&mut self) -> &mut [f32;3] { &mut self.0 }
    pub fn x(&self) -> f32 { self.0[0] }
    pub fn y(&self) -> f32 { self.0[1] }
    pub fn z(&self) -> f32 { self.0[2] }
    pub fn dot(&self, other: &Vec3) -> f32 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 ( [
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ] )
    }
    pub fn mag(&self) -> f32 { ( self[0].powi(2) + self[1].powi(2) + self[2].powi(2) ).sqrt() }
    pub fn length(&self) -> f32 { self.mag() }
    pub fn normalize(&self) -> Vec3 {
        let mag = self.mag();
        if mag != 0.0 {
            *self / mag
        }
        else {
            *self
        }
    }
    pub fn bound(&self, bound: f32) -> Vec3 {
        Vec3 ( [
            self[0] % bound,
            self[1] % bound,
            self[2] % bound,
        ] )
    }
    pub fn perpendicular(&self) -> Vec3 {
        if self[2]<self[0] {
            Vec3::new(self[1],-self[0],0.0)
        }
        else {
            Vec3::new(0.0,-self[2],self[1])
        }
    }
}

impl std::ops::Index<usize> for Vec3 {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3 ( [
            self[0] + other[0],
            self[1] + other[1],
            self[2] + other[2],
        ] )
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        *self = Vec3 ( [
            self[0] + other[0],
            self[1] + other[1],
            self[2] + other[2],
        ] )
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 ( [
            self[0] - other[0],
            self[1] - other[1],
            self[2] - other[2],
        ] )
    }
}

impl std::ops::SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Vec3) {
        *self = Vec3 ( [
            self[0] - other[0],
            self[1] - other[1],
            self[2] - other[2],
        ] )
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f32) -> Vec3 {
        Vec3 ( [
            self[0] * rhs,
            self[1] * rhs,
            self[2] * rhs,
        ] )
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f32) -> Vec3 {
        if rhs == 0.0 { panic!("Cannot divide by zero. (Vec3 / 0.0)"); }
        Vec3 ( [
            self[0] / rhs,
            self[1] / rhs,
            self[2] / rhs,
        ] )
    }
}

impl std::ops::Rem<f32> for Vec3 {
    type Output = Vec3;

    fn rem(self, rhs: f32) -> Vec3 {
        if rhs == 0.0 { panic!("Cannot divide by zero. (Vec3 % 0.0)"); }
        Vec3 ( [
            self[0] % rhs,
            self[1] % rhs,
            self[2] % rhs,
        ] )
    }
}

impl std::ops::RemAssign<f32> for Vec3 {

    fn rem_assign(&mut self, rhs: f32) {
        if rhs == 0.0 { panic!("Cannot divide by zero. (Vec3 % 0.0)"); }
        *self = Vec3 ( [
            self[0] % rhs,
            self[1] % rhs,
            self[2] % rhs,
        ] )
    }
}