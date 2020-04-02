
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
        assert!(( $left - $right ).abs() < std::f32::EPSILON);
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

    Mat4::mat4([
        w, 0.0, 0.0, 0.0,
        0.0, h, 0.0, 0.0,
        0.0, 0.0, q, -1.0,
        0.0, 0.0, qn, 0.0,
    ])
}

/// Build a *View Matrix* that transforms vertices from world space to eye space
///
/// The matrix is derived from an eye point, a reference point indicating the center of
/// the scene, and an UP vector.
///
/// - `pos`: eye point in world coordinates
/// - `target`: reference point in world coordinates that will be in the center of the screen
/// - `up`: which direction is *up* in your world, this is typically [`Vec3::Y_AXIS`](./struct.Vec3.html#associatedconstant.Y_AXIS)
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
    let zaxis: Vec3 = (target - pos).normalize();
    let xaxis = zaxis.cross(up).normalize();
    let yaxis = xaxis.cross(zaxis);

    Mat4::mat4([
        xaxis.x(), yaxis.x(), -zaxis.x(), 0.0,
        xaxis.y(), yaxis.y(), -zaxis.y(), 0.0,
        xaxis.z(), yaxis.z(), -zaxis.z(), 0.0,
        -pos.dot(xaxis), -pos.dot(yaxis), pos.dot(zaxis), 1.0,
    ])
}

/// Build a *Translation Matrix* that transforms vectors in the world space
///
/// - `t`: [Vec3](./struct.Vec3.html) containing model world  position
///
/// ## Where is this typically used?
///
/// This is typically used when building the model of an entity.
///
/// ```
///  # use homebrew_glm::{translate, rotate, scale, Vec3, Mat4, Quat};
///  # let my_position = Vec3::zero();
///  # let my_rotation = Quat::identity();
///  # let my_scale = Vec3::one();
///  let model: Mat4 = translate(my_position) * rotate(my_rotation) * scale(my_scale);
/// ```
///
/// The translation matrix is often the left most matrix. Matrix mulitiplication is **not**
/// commutative, you can think of the translation transformation happening last. Each matrix is in
/// reference to the origin (0,0,0) and if we were to translate before [rotating](./fn.rotate.html),
/// we would rotate around the origin instead of in place.
///
/// ## Example
///
/// In the code snippet below, we are translating a vector, which really is equivalent to adding a
/// vectors components. This is how we set the position of points in the world space.
///
/// ```
/// # use homebrew_glm::{translate, Vec3, Vec4};
/// let vec = Vec4::new(2.0, 2.0, 2.0, 1.0);
/// let translate = translate(Vec3::new(1.0, -2.0, 0.5));
/// let expected = Vec4::new(3.0, 0.0, 2.5, 1.0);
/// assert!(expected.equals( translate * vec ));
/// ```
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.4/api/a00151.html#ga8925161ecc1767957900c5ca8b922dc4](https://glm.g-truc.net/0.9.4/api/a00151.html#ga8925161ecc1767957900c5ca8b922dc4)
pub fn translate(t: Vec3) -> Mat4 {
    Mat4::mat4([
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
/// ```
///  # use homebrew_glm::{translate, rotate, scale, Vec3, Mat4, Quat};
///  # let my_position = Vec3::zero();
///  # let my_rotation = Quat::identity();
///  # let my_scale = Vec3::one();
///  let model: Mat4 = translate(my_position) * rotate(my_rotation) * scale(my_scale);
/// ```
///
/// The rotation matrix is to the right of the [translation](./fn.translate.html) matrix, so that we
/// may rotate in place around the origin (0,0,0) before we translate. Unless you want the model to
/// stretch along a certain axis no matter which way it's rotated, the rotation matrix should always
/// come after (to the left) of the [scale](./fn.scale.html) matrix.
///
/// ## Example
///
/// In the code snippet below, we are rotating a vector that is pointing in the positive Z direction
/// a quater turn counter clockwise around an axis-vector pointing in the negative X and positive Y
/// direction. The expected resulting vector is pointing in the positive X and positive Y direction.
///
/// ```
/// # use homebrew_glm::{rotate, Vec3, Vec4, Quat};
/// # use std::f32::consts::PI;
/// let vec = Vec4::new(0.0, 0.0, 1.0, 0.0);
/// let rotation = rotate(Quat::from_angle_axis(PI/2.0, Vec3::new(-1.0, 1.0, 0.0).normalize()));
/// let expected = Vec4::new(1.0, 1.0, 0.0, 0.0).normalize();
/// assert!(expected.equals( rotation * vec ));
/// ```
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
/// ```
///  # use homebrew_glm::{translate, rotate_x, rotate_y, rotate_z, scale, Vec3, Mat4, Quat};
///  # let my_position = Vec3::zero();
///  # let my_rotation = Vec3::zero();
///  # let my_scale = Vec3::one();
///  let model: Mat4 =
///      translate(my_position)
///          * rotate_x(my_rotation.x())
///          * rotate_y(my_rotation.x())
///          * rotate_z(my_rotation.x())
///          * scale(my_scale);
/// ```
///
/// The order in when each axis is rotated depends entirely on your use case, but order does matter.
/// For example, something rotated about the X axis 90 degrees clockwise and then rotated around
/// the Z axis 180 degrees, is different from the same rotations in opposite order.
///
/// ## Example
///
/// In the code snippet below, we are rotating a vector that is pointing in the positive Z direction
/// a quater turn counter clockwise around the X axis. The expected resulting vector is pointing in
/// the negative Y direction.
///
/// ```
/// # use homebrew_glm::{rotate_x, Vec4};
/// # use std::f32::consts::PI;
/// let vec = Vec4::new(0.0, 0.0, 1.0, 0.0);
/// let rotation = rotate_x(PI/2.0);
/// let expected = Vec4::new(0.0, -1.0, 0.0, 0.0);
/// assert!(expected.equals( rotation * vec ));
/// ```
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.3/api/a00199.html#gaaadca0c077515d56955f3c662a3a3c7f](https://glm.g-truc.net/0.9.3/api/a00199.html#gaaadca0c077515d56955f3c662a3a3c7f)
pub fn rotate_x(f: f32) -> Mat4 {
    Mat4::mat4([
        1.0, 0.0, 0.0, 0.0,
        0.0, f.cos(), f.sin(), 0.0,
        0.0, -f.sin(), f.cos(), 0.0,
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
/// ```
///  # use homebrew_glm::{translate, rotate_x, rotate_y, rotate_z, scale, Vec3, Mat4, Quat};
///  # let my_position = Vec3::zero();
///  # let my_rotation = Vec3::zero();
///  # let my_scale = Vec3::one();
///  let model: Mat4 =
///      translate(my_position)
///          * rotate_x(my_rotation.x())
///          * rotate_y(my_rotation.x())
///          * rotate_z(my_rotation.x())
///          * scale(my_scale);
/// ```
///
/// The order in when each axis is rotated depends entirely on your use case, but order does matter.
/// For example, something rotated about the X axis 90 degrees clockwise and then rotated around
/// the Z axis 180 degrees, is different from the same rotations in opposite order.
///
/// ## Example
///
/// In the code snippet below, we are rotating a vector that is pointing in the positive Z direction
/// a quater turn counter clockwise around the Y axis. The expected resulting vector is pointing in
/// the positive X direction.
///
/// ```
/// # use homebrew_glm::{rotate_y, Vec4};
/// # use std::f32::consts::PI;
/// let vec = Vec4::new(0.0, 0.0, 1.0, 0.0);
/// let rotation = rotate_y(PI/2.0);
/// let expected = Vec4::new(1.0, 0.0, 0.0, 0.0);
/// assert!(expected.equals( rotation * vec ));
/// ```
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.3/api/a00199.html#gacffa0ae7f32f4e2ee7bc1dc0ed290d45](https://glm.g-truc.net/0.9.3/api/a00199.html#gacffa0ae7f32f4e2ee7bc1dc0ed290d45)
pub fn rotate_y(f: f32) -> Mat4 {
    Mat4::mat4([
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
/// ```
///  # use homebrew_glm::{translate, rotate_x, rotate_y, rotate_z, scale, Vec3, Mat4, Quat};
///  # let my_position = Vec3::zero();
///  # let my_rotation = Vec3::zero();
///  # let my_scale = Vec3::one();
///  let model: Mat4 =
///      translate(my_position)
///          * rotate_x(my_rotation.x())
///          * rotate_y(my_rotation.x())
///          * rotate_z(my_rotation.x())
///          * scale(my_scale);
/// ```
///
/// The order in when each axis is rotated depends entirely on your use case, but order does matter.
/// For example, something rotated about the X axis 90 degrees clockwise and then rotated around
/// the Z axis 180 degrees, is different from the same rotations in opposite order.
///
/// ## Example
///
/// In the code snippet below, we are rotating a vector that is pointing in the positive X direction
/// a quater turn counter clockwise around the Z axis. The expected resulting vector is pointing in
/// the positive Y direction.
///
/// ```
/// # use homebrew_glm::{rotate_z, Vec4};
/// # use std::f32::consts::PI;
/// let vec = Vec4::new(1.0, 0.0, 0.0, 0.0);
/// let rotation = rotate_z(PI/2.0);
/// let expected = Vec4::new(0.0, 1.0, 0.0, 0.0);
/// assert!(expected.equals( rotation * vec ));
/// ```
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.3/api/a00199.html#ga105c77751b4ab56c491334655751e0af](https://glm.g-truc.net/0.9.3/api/a00199.html#ga105c77751b4ab56c491334655751e0af)
pub fn rotate_z(f: f32) -> Mat4 {
    Mat4::mat4([
        f.cos(), f.sin(), 0.0, 0.0,
        -f.sin(), f.cos(), 0.0, 0.0,
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
/// ```
///  # use homebrew_glm::{translate, rotate, scale, Vec3, Mat4, Quat};
///  # let my_position = Vec3::zero();
///  # let my_rotation = Quat::identity();
///  # let my_scale = Vec3::one();
///  let model: Mat4 = translate(my_position) * rotate(my_rotation) * scale(my_scale);
/// ```
///
/// The scale matrix should be the right most matrix, so that the model is scaled first, otherwise
/// we're scaling other transformations. Unless you want the model to stretch along a certain axis
/// no matter which way it's [rotated](./fn.rotate.html), the scale matrix should always come first
/// (the far right).
///
/// ## Example
///
/// In the code snippet below, we are scaling a vector by it's individual components. It's
/// equivalent to multiplying each component by a different scalar.
///
/// ```
/// # use homebrew_glm::{scale, Vec3, Vec4};
/// let vec = Vec4::new(2.0, 2.0, 2.0, 0.0);
/// let scale = scale(Vec3::new(1.0, -2.0, 0.5));
/// let expected = Vec4::new(2.0, -4.0, 1.0, 0.0);
/// assert!(expected.equals( scale * vec ));
/// ```
///
/// ## GLM equivalent function
///
/// GLM documentation: [https://glm.g-truc.net/0.9.4/api/a00151.html#ga223e08009f1cab54651200b81e91981c](https://glm.g-truc.net/0.9.4/api/a00151.html#ga223e08009f1cab54651200b81e91981c)
pub fn scale(s: Vec3) -> Mat4 {
    Mat4::mat4([
        s[0],  0.0,  0.0, 0.0,
        0.0, s[1],  0.0, 0.0,
        0.0,  0.0, s[2], 0.0,
        0.0,  0.0,  0.0, 1.0,
    ])
}