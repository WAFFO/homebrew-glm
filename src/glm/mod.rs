
pub(crate) mod vec3;
pub(crate) mod vec4;
pub(crate) mod mat3;
pub(crate) mod mat4;
pub(crate) mod quat;

pub use vec3::Vec3;
pub use vec4::Vec4;
pub use mat3::Mat3;
pub use mat4::Mat4;
pub use quat::Quat;

pub const NEAR_ZERO: f32 = 0.000001;
pub const D_NEAR_ZERO: f64 = 0.000001;

#[doc(hidden)]
#[macro_export]
macro_rules! assert_eq_float {
    ($left:expr, $right:expr) => {
        assert!(( $left - $right ).abs() < 0.000001);
    }
}

pub fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4::new(x, y, z, w)
}

pub fn mat3(col1: Vec3, col2: Vec3, col3: Vec3) -> Mat3 {
    Mat3::new(col1, col2, col3)
}

pub fn mat4(col1: Vec4, col2: Vec4, col3: Vec4, col4: Vec4) -> Mat4 {
    Mat4::new(col1, col2, col3, col4)
}

pub fn perspective(aspect: f32, fov: f32, near: f32, far: f32) -> Mat4 {
    let xy_max = near * fov.to_radians();

    let depth = far - near;
    let q = -(far + near) / depth;
    let qn = -2.0 * (far * near) / depth;

    let w = 2.0 * near / xy_max;
    let w = w / aspect;
    let h = 2.0 * near / xy_max;

    let mut m = Mat4::zero();

    m[0] =   w;
    m[5] =   h;
    m[10] =  q;
    m[11] = -1.0;
    m[14] = qn;

    m
}


pub fn look_at(pos: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let zaxis: Vec3 = (pos - target).normalize();
    let xaxis: Vec3 = up.cross(target).normalize();
    let yaxis: Vec3 = zaxis.cross(xaxis).normalize();

    Mat4::new(
        Vec4::new(xaxis[0], yaxis[0], zaxis[0], 0.0),
        Vec4::new(yaxis[1], yaxis[1], zaxis[1], 0.0),
        Vec4::new(zaxis[2], yaxis[2], zaxis[2], 0.0),
        Vec4::new(xaxis.dot(pos) * -1.0, yaxis.dot(pos) * -1.0, zaxis.dot(pos) * -1.0, 1.0),
    )
}

pub fn translate(t: Vec3) -> Mat4 {
    Mat4([
        1.0,  0.0,  0.0, 0.0,
        0.0,  1.0,  0.0, 0.0,
        0.0,  0.0,  1.0, 0.0,
        t[0], t[1], t[2], 1.0,
    ])
}

pub fn rotate(q: Quat) -> Mat4 {
    q.mat4()
}

pub fn rotate_x(f: f32) -> Mat4 {
    Mat4([
        1.0, 0.0, 0.0, 0.0,
        0.0, f.cos(), -f.sin(), 0.0,
        0.0, f.sin(), f.cos(), 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
}

pub fn rotate_y(f: f32) -> Mat4 {
    Mat4([
        f.cos(), 0.0, -f.sin(), 0.0,
        0.0, 1.0, 0.0, 0.0,
        f.sin(), 0.0, f.cos(), 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
}

pub fn rotate_z(f: f32) -> Mat4 {
    Mat4([
        f.cos(), -f.sin(), 0.0, 0.0,
        f.sin(), f.cos(), 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
}

pub fn scale(s: Vec3) -> Mat4 {
    Mat4([
        s[0],  0.0,  0.0, 0.0,
        0.0, s[1],  0.0, 0.0,
        0.0,  0.0, s[2], 0.0,
        0.0,  0.0,  0.0, 1.0,
    ])
}