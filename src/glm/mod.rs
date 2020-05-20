
pub(crate) mod vec3;
pub(crate) mod vec4;
pub(crate) mod mat3;
pub(crate) mod mat4;
pub(crate) mod quat;

use vec3::GVec3;
use vec4::GVec4;
use mat3::GMat3;
use mat4::GMat4;
use quat::GQuat;
use crate::traits::Scalar;

pub const NEAR_ZERO: f64 = 0.000001;

#[doc(hidden)]
#[macro_export]
macro_rules! assert_eq_float {
    ($left:expr, $right:expr) => {
        assert!(( $left - $right ).abs() < std::f32::EPSILON);
    }
}

impl<T: Scalar> GMat4<T> {
    /// Build a *Projection Matrix* that transforms vertices from eye space to the clip space
    ///
    /// - `left`: value furthest on the left on the X axis
    /// - `right`: value furthest on the right on the X axis
    /// - `bottom`: value furthest down on the Y axis
    /// - `top`: value furthest up on the Y axis
    /// - `near`: value closest to the viewer on the Z axis
    /// - `far`: value furthest from the viewer on the Z axis
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
    /// ## Note about OpenGL behavior
    ///
    /// Much to the confusion of many programmers, OpenGL interprets the positive Z axis to point
    /// **away** from the screen, and thus is actually left-handed. This is confusing as for the rest of
    /// this library, and the offical GLM library, we treat positive Z as being towards the screen. This
    /// is done to uphold the right hand rule across tranformations. For our purposes it's safe to
    /// continue thinking of positive Z as pointing towards the screen, just know that when we translate
    /// into clip space, we flip the Z axis.
    ///
    /// ## GLM equivalent function
    ///
    /// `perspective()`: [https://glm.g-truc.net/0.9.4/api/a00151.html#ga283629a5ac7fb9037795435daf22560f](https://glm.g-truc.net/0.9.4/api/a00151.html#ga283629a5ac7fb9037795435daf22560f)
    pub fn perspective(left: T, right: T, bottom: T, top: T, near: T, far: T) -> GMat4<T> {
        let w = (T::cast(2.0) * near) / (right - left);
        let h = (T::cast(2.0) * near) / (top - bottom);
        let wq = (right + left) / (right - left);
        let hq = (top + bottom) / (top - bottom);
        let q = -(far + near) / (far - near);
        let qn = -T::cast(2.0) * (far * near) / (far - near);

        GMat4::mat4([
            w, T::zero(), T::zero(), T::zero(),
            T::zero(), h, T::zero(), T::zero(),
            wq, hq, q, -T::one(),
            T::zero(), T::zero(), qn, T::zero(),
        ])
    }

    /// Build a *Projection Matrix* that transforms vertices from eye space to the clip space
    ///
    /// - `fov_y`: field of view in **radians** along the Y axis
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
    /// ## Note about OpenGL behavior
    ///
    /// Much to the confusion of many programmers, OpenGL interprets the positive Z axis to point
    /// **away** from the screen, and thus is actually left-handed. This is confusing as for the rest of
    /// this library, and the offical GLM library, we treat positive Z as being towards the screen. This
    /// is done to uphold the right hand rule across tranformations. For our purposes it's safe to
    /// continue thinking of positive Z as pointing towards the screen, just know that when we translate
    /// into clip space, we flip the Z axis.
    ///
    /// ## GLM equivalent function
    ///
    /// `perspectiveFov()`: [https://glm.g-truc.net/0.9.4/api/a00151.html#gac2bbb4ae38c7cc549feefae5406517d7](https://glm.g-truc.net/0.9.4/api/a00151.html#gac2bbb4ae38c7cc549feefae5406517d7)
    pub fn perspective_fov(fov_y: T, aspect: T, near: T, far: T) -> GMat4<T> {
        let xy_max = near * fov_y;

        let depth = far - near;
        let q = -(far + near) / depth;
        let qn = -T::cast(2.0) * (far * near) / depth;

        let w = (T::cast(2.0) * near / xy_max) / aspect;
        let h = T::cast(2.0) * near / xy_max;

        GMat4::mat4([
            w, T::zero(), T::zero(), T::zero(),
            T::zero(), h, T::zero(), T::zero(),
            T::zero(), T::zero(), q, -T::one(),
            T::zero(), T::zero(), qn, T::zero(),
        ])
    }

    /// Build a *Projection Matrix* that transforms vertices from eye space to the clip space
    ///
    /// - `fov_y`: field of view in **radians** along the Y axis
    /// - `aspect`: aspect ratio of the view
    /// - `near`: minimum distance something is visible
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
    /// ## Note about OpenGL behavior
    ///
    /// Much to the confusion of many programmers, OpenGL interprets the positive Z axis to point
    /// **away** from the screen, and thus is actually left-handed. This is confusing as for the rest of
    /// this library, and the offical GLM library, we treat positive Z as being towards the screen. This
    /// is done to uphold the right hand rule across tranformations. For our purposes it's safe to
    /// continue thinking of positive Z as pointing towards the screen, just know that when we translate
    /// into clip space, we flip the Z axis.
    ///
    /// ## GLM equivalent function
    ///
    /// `infinitePerspective()`: [https://glm.g-truc.net/0.9.4/api/a00151.html#ga414f3cfe1af5619acebd5c28cf6bd45c](https://glm.g-truc.net/0.9.4/api/a00151.html#ga414f3cfe1af5619acebd5c28cf6bd45c)
    pub fn perspective_infinite(fov_y: T, aspect: T, near: T) -> GMat4<T> {
        let xy_max = near * fov_y;

        let qn = -T::cast(2.0) * near;

        let w = (T::cast(2.0) * near / xy_max) / aspect;
        let h = T::cast(2.0) * near / xy_max;

        GMat4::mat4([
            w, T::zero(), T::zero(), T::zero(),
            T::zero(), h, T::zero(), T::zero(),
            T::zero(), T::zero(), -T::one(), -T::one(),
            T::zero(), T::zero(), qn, T::zero(),
        ])
    }

    /// Build a Orthographic *Projection Matrix* that transforms vertices from eye space to the clip
    /// space
    ///
    /// - `left`: value furthest on the left on the X axis
    /// - `right`: value furthest on the right on the X axis
    /// - `bottom`: value furthest down on the Y axis
    /// - `top`: value furthest up on the Y axis
    /// - `near`: value closest to the viewer on the Z axis
    /// - `far`: value furthest from the viewer on the Z axis
    ///
    /// ## Where is this typically used?
    ///
    /// An orthographic projection matrix is different from a regular perspective matrix that uses a
    /// frustum. A regular perspective matrix will create the illusion of objects being smaller the
    /// further away they are from the viewpoint, and objects larger when closer to the viewpoint.
    /// Instead of a frustum, orthographic projection uses a rectangular prism, nothing grows in size
    /// based on the Z axis, near or far the size is consistent.
    ///
    /// This is useful more often for 3D tools where it's beneficial for a user to judge the size of
    /// objects without their distance affecting the size.
    ///
    /// ## Note about OpenGL behavior
    ///
    /// Much to the confusion of many programmers, OpenGL interprets the positive Z axis to point
    /// **away** from the screen, and thus is actually left-handed. This is confusing as for the rest of
    /// this library, and the offical GLM library, we treat positive Z as being towards the screen. This
    /// is done to uphold the right hand rule across tranformations. For our purposes it's safe to
    /// continue thinking of positive Z as pointing towards the screen, just know that when we translate
    /// into clip space, we flip the Z axis.
    ///
    /// ## GLM equivalent function
    ///
    /// `ortho()`: [https://glm.g-truc.net/0.9.4/api/a00151.html#gaf039a9f8d24e4bf39d30b7d692c1b8c3](https://glm.g-truc.net/0.9.4/api/a00151.html#gaf039a9f8d24e4bf39d30b7d692c1b8c3)
    pub fn perspective_ortho(left: T, right: T, bottom: T, top: T, near: T, far: T) -> GMat4<T> {
        let w = T::one() / (right - left);
        let h = T::one() / (top - bottom);
        let p = T::one() / (far - near);

        let x = (right + left) * -w;
        let y = (top + bottom) * -h;
        let z = (far + near) * -p;

        let w = T::cast(2.0) * w;
        let h = T::cast(2.0) * h;
        let p = -T::cast(2.0) * p;

        GMat4::mat4([
            w, T::zero(), T::zero(), T::zero(),
            T::zero(), h, T::zero(), T::zero(),
            T::zero(), T::zero(), p, T::zero(),
            x, y, z, T::one(),
        ])
    }

    /// Build a *View Matrix* that transforms vertices from world space to eye space
    ///
    /// The matrix is derived from an eye point, a reference point indicating the center of
    /// the scene, and an up vector.
    ///
    /// - `pos`: eye point in world coordinates
    /// - `target`: reference point in world coordinates that will be in the center of the screen
    /// - `up`: which direction is up in your world, this is typically [`Vec3::Y_AXIS`](./struct.Vec3.html#associatedconstant.Y_AXIS)
    ///
    /// ## Where is this typically used?
    ///
    /// Consider this GLSL code:
    ///
    /// ```glsl
    /// gl_Position = u_projection * view * v_position;
    /// ```
    ///
    /// The view matrix is typically used to multiply the current vertex, being iterated on,
    /// `v_position` in the example. This matrix rotates the world space around based on the eye
    /// position and where it's looking. The result of this is often times referred to as the eye space.
    ///
    /// ## GLM equivalent function
    ///
    /// `lookAt()`: [https://glm.g-truc.net/0.9.4/api/a00151.html#gae2dca3785b6d5796e876114af58a60a1](https://glm.g-truc.net/0.9.4/api/a00151.html#gae2dca3785b6d5796e876114af58a60a1)
    pub fn look_at(pos: GVec3<T>, target: GVec3<T>, up: GVec3<T>) -> GMat4<T> {
        let zaxis: GVec3<T> = (target - pos).normalize();
        let xaxis = zaxis.cross(up).normalize();
        let yaxis = xaxis.cross(zaxis);

        GMat4::mat4([
            xaxis.x(), yaxis.x(), -zaxis.x(), T::zero(),
            xaxis.y(), yaxis.y(), -zaxis.y(), T::zero(),
            xaxis.z(), yaxis.z(), -zaxis.z(), T::zero(),
            pos.dot(xaxis) * -T::one(), pos.dot(yaxis) * -T::one(), pos.dot(zaxis) * -T::one(), T::one(),
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
    ///  # use sawd_glm::{Vec3, Mat4, Quat};
    ///  # let my_position = Vec3::zero();
    ///  # let my_rotation = Quat::identity();
    ///  # let my_scale = Vec3::one();
    ///  let model: Mat4 = Mat4::translate(my_position) * Mat4::rotate(my_rotation) * Mat4::scale(my_scale);
    /// ```
    ///
    /// The translation matrix is often the left most matrix. Matrix mulitiplication is **not**
    /// commutative, you can think of the translation transformation happening last. Each matrix is in
    /// reference to the origin (0,0,0) and if we were to translate before [rotating](./fn.rotate.html),
    /// we would rotate around the origin instead of in place.
    ///
    /// ## Example
    ///
    /// In the code snippet below, I am translating a vector, which really is equivalent to adding a
    /// vectors components. This is how we set the position of points in the world space.
    ///
    /// ```
    /// # use sawd_glm::{Vec3, Vec4, Mat4};
    /// let vec = Vec4::new(2.0, 2.0, 2.0, 1.0);
    /// let translate = Mat4::translate(Vec3::new(1.0, -2.0, 0.5));
    /// let expected = Vec4::new(3.0, 0.0, 2.5, 1.0);
    /// assert!(expected.equals( translate * vec ));
    /// ```
    ///
    /// ## GLM equivalent function
    ///
    /// `translate()`: [https://glm.g-truc.net/0.9.4/api/a00151.html#ga8925161ecc1767957900c5ca8b922dc4](https://glm.g-truc.net/0.9.4/api/a00151.html#ga8925161ecc1767957900c5ca8b922dc4)
    pub fn translate(t: GVec3<T>) -> GMat4<T> {
        GMat4::mat4([
            T::one(), T::zero(), T::zero(), T::zero(),
            T::zero(), T::one(), T::zero(), T::zero(),
            T::zero(), T::zero(), T::one(), T::zero(),
            t[0], t[1], t[2], T::one(),
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
    ///  # use sawd_glm::{Vec3, Mat4, Quat};
    ///  # let my_position = Vec3::zero();
    ///  # let my_rotation = Quat::identity();
    ///  # let my_scale = Vec3::one();
    ///  let model: Mat4 = Mat4::translate(my_position) * Mat4::rotate(my_rotation) * Mat4::scale(my_scale);
    /// ```
    ///
    /// The rotation matrix is to the right of the [translation](./fn.translate.html) matrix, so that we
    /// may rotate in place around the origin (0,0,0) before we translate. The rotation matrix should
    /// always come after (to the left of) the [scale](./fn.scale.html) matrix, otherwise the model to
    /// stretch along a certain axis no matter which way it's rotated.
    ///
    /// ## Example
    ///
    /// In the code snippet below, I am rotating a vector that is pointing in the positive Z direction
    /// a quarter turn counterclockwise around an axis-vector pointing in the negative X and positive Y
    /// direction. The expected resulting vector is pointing in the positive X and positive Y direction.
    ///
    /// ```
    /// # use sawd_glm::{Vec3, Vec4, Mat4, Quat};
    /// # use std::f32::consts::PI;
    /// let vec = Vec4::new(0.0, 0.0, 1.0, 0.0);
    /// let rotation = Mat4::rotate(Quat::from_angle_axis(PI/2.0, Vec3::new(-1.0, 1.0, 0.0).normalize()));
    /// let expected = Vec4::new(1.0, 1.0, 0.0, 0.0).normalize();
    /// assert!(expected.equals( rotation * vec ));
    /// ```
    ///
    /// ## GLM equivalent function
    ///
    /// `rotate()`: [https://glm.g-truc.net/0.9.4/api/a00151.html#gaacb9cbe8f93a8fef9dc3e25559df19c0](https://glm.g-truc.net/0.9.4/api/a00151.html#gaacb9cbe8f93a8fef9dc3e25559df19c0)
    pub fn rotate(q: GQuat<T>) -> GMat4<T> {
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
    ///  # use sawd_glm::{Vec3, Mat4, Quat};
    ///  # let my_position = Vec3::zero();
    ///  # let my_rotation = Vec3::zero();
    ///  # let my_scale = Vec3::one();
    ///  let model: Mat4 =
    ///      Mat4::translate(my_position)
    ///          * Mat4::rotate_x(my_rotation.x())
    ///          * Mat4::rotate_y(my_rotation.x())
    ///          * Mat4::rotate_z(my_rotation.x())
    ///          * Mat4::scale(my_scale);
    /// ```
    ///
    /// The order in when each axis is rotated depends entirely on your use case, but order does matter.
    /// For example, something rotated about the X axis 90 degrees clockwise and then rotated around
    /// the Z axis 180 degrees, is different from the same rotations in opposite order.
    ///
    /// ## Example
    ///
    /// In the code snippet below, I am rotating a vector that is pointing in the positive Z direction
    /// a quarter turn counterclockwise around the X axis. The expected resulting vector is pointing in
    /// the negative Y direction.
    ///
    /// ```
    /// # use sawd_glm::{Vec4, Mat4};
    /// # use std::f32::consts::PI;
    /// let vec = Vec4::new(0.0, 0.0, 1.0, 0.0);
    /// let rotation = Mat4::rotate_x(PI/2.0);
    /// let expected = Vec4::new(0.0, -1.0, 0.0, 0.0);
    /// assert!(expected.equals( rotation * vec ));
    /// ```
    ///
    /// ## GLM equivalent function
    ///
    /// `rotateX()`: [https://glm.g-truc.net/0.9.3/api/a00199.html#gaaadca0c077515d56955f3c662a3a3c7f](https://glm.g-truc.net/0.9.3/api/a00199.html#gaaadca0c077515d56955f3c662a3a3c7f)
    pub fn rotate_x(f: T) -> GMat4<T> {
        GMat4::mat4([
            T::one(), T::zero(), T::zero(), T::zero(),
            T::zero(), f.cos(), f.sin(), T::zero(),
            T::zero(), -f.sin(), f.cos(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::one(),
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
    ///  # use sawd_glm::{Vec3, Mat4, Quat};
    ///  # let my_position = Vec3::zero();
    ///  # let my_rotation = Vec3::zero();
    ///  # let my_scale = Vec3::one();
    ///  let model: Mat4 =
    ///      Mat4::translate(my_position)
    ///          * Mat4::rotate_x(my_rotation.x())
    ///          * Mat4::rotate_y(my_rotation.x())
    ///          * Mat4::rotate_z(my_rotation.x())
    ///          * Mat4::scale(my_scale);
    /// ```
    ///
    /// The order in when each axis is rotated depends entirely on your use case, but order does matter.
    /// For example, something rotated about the X axis 90 degrees clockwise and then rotated around
    /// the Z axis 180 degrees, is different from the same rotations in opposite order.
    ///
    /// ## Example
    ///
    /// In the code snippet below, I am rotating a vector that is pointing in the positive Z direction
    /// a quarter turn counterclockwise around the Y axis. The expected resulting vector is pointing in
    /// the positive X direction.
    ///
    /// ```
    /// # use sawd_glm::{Vec4, Mat4};
    /// # use std::f32::consts::PI;
    /// let vec = Vec4::new(0.0, 0.0, 1.0, 0.0);
    /// let rotation = Mat4::rotate_y(PI/2.0);
    /// let expected = Vec4::new(1.0, 0.0, 0.0, 0.0);
    /// assert!(expected.equals( rotation * vec ));
    /// ```
    ///
    /// ## GLM equivalent function
    ///
    /// `rotateY()`: [https://glm.g-truc.net/0.9.3/api/a00199.html#gacffa0ae7f32f4e2ee7bc1dc0ed290d45](https://glm.g-truc.net/0.9.3/api/a00199.html#gacffa0ae7f32f4e2ee7bc1dc0ed290d45)
    pub fn rotate_y(f: T) -> GMat4<T> {
        GMat4::mat4([
            f.cos(), T::zero(), -f.sin(), T::zero(),
            T::zero(), T::one(), T::zero(), T::zero(),
            f.sin(), T::zero(), f.cos(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::one(),
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
    ///  # use sawd_glm::{Vec3, Mat4, Quat};
    ///  # let my_position = Vec3::zero();
    ///  # let my_rotation = Vec3::zero();
    ///  # let my_scale = Vec3::one();
    ///  let model: Mat4 =
    ///      Mat4::translate(my_position)
    ///          * Mat4::rotate_x(my_rotation.x())
    ///          * Mat4::rotate_y(my_rotation.x())
    ///          * Mat4::rotate_z(my_rotation.x())
    ///          * Mat4::scale(my_scale);
    /// ```
    ///
    /// The order in when each axis is rotated depends entirely on your use case, but order does matter.
    /// For example, something rotated about the X axis 90 degrees clockwise and then rotated around
    /// the Z axis 180 degrees, is different from the same rotations in opposite order.
    ///
    /// ## Example
    ///
    /// In the code snippet below, I am rotating a vector that is pointing in the positive X direction
    /// a quarter turn counterclockwise around the Z axis. The expected resulting vector is pointing in
    /// the positive Y direction.
    ///
    /// ```
    /// # use sawd_glm::{Vec4, Mat4};
    /// # use std::f32::consts::PI;
    /// let vec = Vec4::new(1.0, 0.0, 0.0, 0.0);
    /// let rotation = Mat4::rotate_z(PI/2.0);
    /// let expected = Vec4::new(0.0, 1.0, 0.0, 0.0);
    /// assert!(expected.equals( rotation * vec ));
    /// ```
    ///
    /// ## GLM equivalent function
    ///
    /// `rotateZ()`: [https://glm.g-truc.net/0.9.3/api/a00199.html#ga105c77751b4ab56c491334655751e0af](https://glm.g-truc.net/0.9.3/api/a00199.html#ga105c77751b4ab56c491334655751e0af)
    pub fn rotate_z(f: T) -> GMat4<T> {
        GMat4::mat4([
            f.cos(), f.sin(), T::zero(), T::zero(),
            -f.sin(), f.cos(), T::zero(), T::zero(),
            T::zero(), T::zero(), T::one(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::one(),
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
    ///  # use sawd_glm::{Vec3, Mat4, Quat};
    ///  # let my_position = Vec3::zero();
    ///  # let my_rotation = Quat::identity();
    ///  # let my_scale = Vec3::one();
    ///  let model: Mat4 = Mat4::translate(my_position) * Mat4::rotate(my_rotation) * Mat4::scale(my_scale);
    /// ```
    ///
    /// The scale matrix should be the right most matrix, so that the model is scaled first, otherwise
    /// we're scaling other transformations. Unless you want the model to stretch along a certain axis
    /// no matter which way it's [rotated](./fn.rotate.html), the scale matrix should always come first
    /// (the far right).
    ///
    /// ## Example
    ///
    /// In the code snippet below, I am scaling a vector by it's individual components. It's
    /// equivalent to multiplying each component by a different scalar.
    ///
    /// ```
    /// # use sawd_glm::{Vec3, Vec4, Mat4};
    /// let vec = Vec4::new(2.0, 2.0, 2.0, 0.0);
    /// let scale = Mat4::scale(Vec3::new(1.0, -2.0, 0.5));
    /// let expected = Vec4::new(2.0, -4.0, 1.0, 0.0);
    /// assert!(expected.equals( scale * vec ));
    /// ```
    ///
    /// ## GLM equivalent function
    ///
    /// `scale()`: [https://glm.g-truc.net/0.9.4/api/a00151.html#ga223e08009f1cab54651200b81e91981c](https://glm.g-truc.net/0.9.4/api/a00151.html#ga223e08009f1cab54651200b81e91981c)
    pub fn scale(s: GVec3<T>) -> GMat4<T> {
        GMat4::mat4([
            s[0], T::zero(), T::zero(), T::zero(),
            T::zero(), s[1], T::zero(), T::zero(),
            T::zero(), T::zero(), s[2], T::zero(),
            T::zero(), T::zero(), T::zero(), T::one(),
        ])
    }
}