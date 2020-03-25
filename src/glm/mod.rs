
pub(crate) mod vec3;
pub(crate) mod vec4;
pub(crate) mod mat3;
pub(crate) mod mat4;
pub(crate) mod quat;

use vec3::Vec3;
use vec4::Vec4;
use mat3::Mat3;
use mat4::Mat4;
use quat::Quat;

pub const NEAR_ZERO: f32 = 0.000001;
pub const D_NEAR_ZERO: f64 = 0.000001;

#[doc(hidden)]
#[macro_export]
macro_rules! assert_eq_float {
    ($left:expr, $right:expr) => {
        assert!(( $left - $right ).abs() < 0.000001);
    }
}

/// Create a new Vec3 with x, y, z components
///
/// May also use [`Vec3::new(x, y, z)`](./struct.Vec3.html#method.new)
pub fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// Create a new Vec4 with x, y, z, w components
///
/// May also use [`Vec4::new(x, y, z, w)`](./struct.Vec4.html#method.new)
pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4::new(x, y, z, w)
}

/// Create a new Mat3 with 3 Vec3 columns
///
/// May also use [`Mat3::new(col1, col2, col3)`](./struct.Mat3.html#method.new)
pub fn mat3(col1: Vec3, col2: Vec3, col3: Vec3) -> Mat3 {
    Mat3::new(col1, col2, col3)
}

/// Create a new Mat4 with 4 Vec4 columns
///
/// May also use [`Mat4::new(col1, col2, col3, col4)`](./struct.Mat4.html#method.new)
pub fn mat4(col1: Vec4, col2: Vec4, col3: Vec4, col4: Vec4) -> Mat4 {
    Mat4::new(col1, col2, col3, col4)
}

/// Build a *Projection Matrix* that transforms vertices from eye space to the clip space
///
/// - `fovy`: field of view in *radians* along the y axis
/// - `aspect`: aspect ratio of the view
/// - `near`: minimum distance something is visible
/// - `far`: maximum distance something is visible
///
/// ## Where is this typically used?
///
/// Consider this GLSL code:
///
/// ```glsl
/// gl_Position = u_projection * view * v_position;
/// ```
///
/// The projection matrix is typically the left most matrix, it transforms the result of the current
/// vertex multiplied by the [view](./fn.lookAt.html) matrix, into `gl_Position`.
///
/// Note:`gl_Position` is a special variable GLSL expects to be clip space coordinates.
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.4/api/a00151.html#ga283629a5ac7fb9037795435daf22560f](https://glm.g-truc.net/0.9.4/api/a00151.html#ga283629a5ac7fb9037795435daf22560f)
pub fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let xy_max = near * fovy;

    let depth = far - near;
    let q = -(far + near) / depth;
    let qn = -2.0 * (far * near) / depth;

    let w = 2.0 * near / xy_max;
    let w = w / aspect;
    let h = 2.0 * near / xy_max;

    let mut m = Mat4::zero();

    m[0] = w;
    m[5] = h;
    m[10] = q;
    m[11] = -1.0;
    m[14] = qn;

    m
}

/// Build a *View Matrix* that transforms vertices from world space to eye space
///
/// The matrix is derived from an eye point, a reference point indicating the center of
/// the scene, and an UP vector
///
/// - `pos`: eye point in world coordinates
/// - `target`: reference point in world coordinates that will be in the center of the screen
/// - `up`: which direction is *up* in your world, this is typically [`Vec3::new(0.0, 1.0, 0.0)`](#method.new)
///
/// ## Where is this typically used?
///
/// Consider this GLSL code:
///
/// ```glsl
/// gl_Position = u_projection * view * v_position;
/// ```
///
/// The view matrix is typically used to multiply the current vertex being iterated on. This matrix
/// rotates the world space around based one what is in front of the eye and what is not. The result
/// of this is often times referred to as the eye space.
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.4/api/a00151.html#gae2dca3785b6d5796e876114af58a60a1](https://glm.g-truc.net/0.9.4/api/a00151.html#gae2dca3785b6d5796e876114af58a60a1)
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

/// Build a *Translation Matrix* that transforms vectors in the world space
///
/// - `t`: [Vec3](./struct.Vec3.html) containing model world  position
///
/// ## Where is this typically used?
///
/// This is typically used when building the model of an entity.
///
/// ```text
/// model = translation * rotation * scale;
/// ```
///
/// The translation matrix is often the left most matrix, that way it's the last matrix to be
/// multiplied. Each matrix is in reference to the origin (0,0,0) and if we were to translate before
/// [rotating](./fn.rotate.html), we would rotate around the origin instead of in place.
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.4/api/a00151.html#ga8925161ecc1767957900c5ca8b922dc4](https://glm.g-truc.net/0.9.4/api/a00151.html#ga8925161ecc1767957900c5ca8b922dc4)
pub fn translate(t: Vec3) -> Mat4 {
    Mat4([
        1.0,  0.0,  0.0, 0.0,
        0.0,  1.0,  0.0, 0.0,
        0.0,  0.0,  1.0, 0.0,
        t[0], t[1], t[2], 1.0,
    ])
}

/// Build a *Rotation Matrix* that transforms vectors in the world space
///
/// - `q`: [Quat](./struct.Quat.html) containing model rotation
///
/// ## Where is this typically used?
///
/// This is typically used when building the model of an entity.
///
/// ```text
/// model = translation * rotation * scale;
/// ```
///
/// The rotation matrix is to the right of the [translation](./fn.translate.html) matrix, so that we may rotate in place
/// around the origin (0,0,0) first. Unless you want the model to stretch along a certain axis no
/// matter which way it's rotated, the rotation matrix should always come after (to the left) of the
/// [scale](./fn.scale.html) matrix.
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.4/api/a00151.html#gaacb9cbe8f93a8fef9dc3e25559df19c0](https://glm.g-truc.net/0.9.4/api/a00151.html#gaacb9cbe8f93a8fef9dc3e25559df19c0)
pub fn rotate(q: Quat) -> Mat4 {
    q.mat4()
}

/// Build an *X Axis Rotation Matrix* that transforms vectors in the world space
///
/// - `f`: Float angle in radians to rotate around the X axis (X axis is positive to the right)
///
/// ## Where is this typically used?
///
/// This is typically used when building the model of an entity.
///
/// ```text
/// model = translation * rotate_x * rotate_y * rotate_z * scale;
/// ```
///
/// The order in when each axis is rotated depends entirely on your use case, but order does matter.
/// For example, something rotated about the X axis 90 degrees clockwise and then rotated around
/// the Z axis 180 degrees, is different from the same rotations in opposite order.
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.3/api/a00199.html#gaaadca0c077515d56955f3c662a3a3c7f](https://glm.g-truc.net/0.9.3/api/a00199.html#gaaadca0c077515d56955f3c662a3a3c7f)
pub fn rotate_x(f: f32) -> Mat4 {
    Mat4([
        1.0, 0.0, 0.0, 0.0,
        0.0, f.cos(), -f.sin(), 0.0,
        0.0, f.sin(), f.cos(), 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
}

/// Build an *Y Axis Rotation Matrix* that transforms vectors in the world space
///
/// - `f`: Float angle in radians to rotate around the Y axis (Y axis is positive pointing up)
///
/// ## Where is this typically used?
///
/// This is typically used when building the model of an entity.
///
/// ```text
/// model = translation * rotate_x * rotate_y * rotate_z * scale;
/// ```
///
/// The order in when each axis is rotated depends entirely on your use case, but order does matter.
/// For example, something rotated about the X axis 90 degrees clockwise and then rotated around
/// the Z axis 180 degrees, is different from the same rotations in opposite order.
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.3/api/a00199.html#gacffa0ae7f32f4e2ee7bc1dc0ed290d45](https://glm.g-truc.net/0.9.3/api/a00199.html#gacffa0ae7f32f4e2ee7bc1dc0ed290d45)
pub fn rotate_y(f: f32) -> Mat4 {
    Mat4([
        f.cos(), 0.0, -f.sin(), 0.0,
        0.0, 1.0, 0.0, 0.0,
        f.sin(), 0.0, f.cos(), 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
}

/// Build an *Z Axis Rotation Matrix* that transforms vectors in the world space
///
/// - `f`: Float angle in radians to rotate around the Z axis (Z axis is positive towards the screen)
///
/// ## Where is this typically used?
///
/// This is typically used when building the model of an entity.
///
/// ```text
/// model = translation * rotate_x * rotate_y * rotate_z * scale;
/// ```
///
/// The order in when each axis is rotated depends entirely on your use case, but order does matter.
/// For example, something rotated about the X axis 90 degrees clockwise and then rotated around
/// the Z axis 180 degrees, is different from the same rotations in opposite order.
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.3/api/a00199.html#ga105c77751b4ab56c491334655751e0af](https://glm.g-truc.net/0.9.3/api/a00199.html#ga105c77751b4ab56c491334655751e0af)
pub fn rotate_z(f: f32) -> Mat4 {
    Mat4([
        f.cos(), -f.sin(), 0.0, 0.0,
        f.sin(), f.cos(), 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
}

/// Build a *Scale Matrix* that transforms vectors in the world space
///
/// - `s`: Vec3 representing the scale of the model on each axis
///
/// ## Where is this typically used?
///
/// This is typically used when building the model of an entity.
///
/// ```text
/// model = translation * rotation * scale;
/// ```
///
/// The scale matrix should be the right most matrix, so that the model is scaled first, otherwise
/// we're scaling other transformations. Unless you want the model to stretch along a certain axis
/// no matter which way it's [rotated](./fn.rotate.html), the scale matrix should always come first (the far right).
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.4/api/a00151.html#ga223e08009f1cab54651200b81e91981c](https://glm.g-truc.net/0.9.4/api/a00151.html#ga223e08009f1cab54651200b81e91981c)
pub fn scale(s: Vec3) -> Mat4 {
    Mat4([
        s[0],  0.0,  0.0, 0.0,
        0.0, s[1],  0.0, 0.0,
        0.0,  0.0, s[2], 0.0,
        0.0,  0.0,  0.0, 1.0,
    ])
}