use std::fmt::{Display, Formatter, Error};

use crate::{NEAR_ZERO, Mat4, Vec3, Mat3, look_at, Vec4};

/** # Quat - Quaternion <f32>

  Quaternions are four component vectors that represent rotations or orientations.

  Quat will at all times try stay a unit quaternion, unless you multiply or divide a quaternion by a
  scalar, at which point it will no longer be a unit quaternion.

  ## How to use Quaternions

  There are lots of videos about the math behind quaternions, but I've found far fewer resources on
  how to actually *use* them. This particular fact has always bothered me so I'd like to explain how
  to use them here to the best of my ability.

  #### Creating a Quaternion

  The most common way to create a quaternion and probably the easiest way to think of quaternions is
  with [`Quat::from_angle_axis(angle: f32, axis: Vec3)`](#method.from_angle_axis). Where `axis` is a
  unit [`Vec3`](struct.Vec3.html) representing the axis of rotation, and angle is the rotation
  around `axis` in radians. This creates a [`Quat`](./struct.Quat.html) that represents that
  specific orientation.

  ```
  # use sawd_glm::{Quat, Vec3};
  # use std::f32::consts::PI;
  let axis = Vec3::new(1.0, 1.0, 0.0).normalize();
  let rotation = Quat::from_angle_axis(PI/2.0, axis);
  ```

  #### Rotating a Vec3 with a Quaternion

  Ultimately the point of quaternions or any other 3D transformation is to transform vertices in the
  third dimension.

  Given a [`Vec3`](./struct.Vec3.html) we can rotate it one of two ways:
  - multiplying with a [`Quat`](#)
  - converting [`Quat`](#) into a [`rotation matrix`](./fn.rotate.html), converting [`Vec3`](./struct.Vec3.html)
  into a [`Vec4`](./struct.Vec4.html), and multiplying those two

  ```
  # use sawd_glm::{Quat, Vec3, Vec4, rotate};
  # use std::f32::consts::PI;
  # let axis = Vec3::new(1.0, 1.0, 0.0).normalize();
  # let rotation = Quat::from_angle_axis(PI/2.0, axis);
  let vertex = Vec3::new(10.0, 10.0, 10.0);

  // Rotate via multiplication
  let rotated_vertex_mul: Vec3 = rotation * vertex;

  // Rotate via Mat4 multiplication (like with a model matrix)
  let model = rotate(rotation);
  let vertex_4d = Vec4::vec3_w(vertex, 0.0);
  let rotated_vertex_mat: Vec3 = (model * vertex_4d).xyz();

  assert!(rotated_vertex_mul.equals(rotated_vertex_mat));
  ```

  These are obviously two very different use cases, so use what works best depending on what types
  are on hand at the time.

  #### Updating a Quaternion

  It's entirely possible to just store an angle and an axis and create a quaternion on each frame,
  but the best quality of quaternions is their ability to compose rotations without fear of gimbal
  lock.

  Multiplying two quaternions compounds two rotations in a similar way to vector addition, but it
  is not commutative like vector addition. It's a relatively cheap operation to perform. The
  following example uses [`Quat::from_two_axis()`](#method.from_two_axis); take a look at its
  documentation. Note how the order of rotations affect the result.

  ```
  # use sawd_glm::{Quat, Vec3};
  # use std::f32::consts::PI;
  let rotation_z2y = Quat::from_two_axis(Vec3::Z_AXIS, Vec3::Y_AXIS);
  let rotation_y2x = Quat::from_two_axis(Vec3::Y_AXIS, Vec3::X_AXIS);
  let rotation_a = rotation_z2y * rotation_y2x;
  let rotation_b = rotation_y2x * rotation_z2y;

  let vertex = Vec3::new(1.0, 2.0, 3.0);
  let vertex_rotated_a = rotation_a * vertex;
  let vertex_rotated_b = rotation_b * vertex;

  assert!(vertex_rotated_a.equals(Vec3::new(2.0, 3.0, 1.0)));
  assert!(vertex_rotated_b.equals(Vec3::new(3.0, -1.0, -2.0)));
  ```

  Notice how the resulting vectors are just rearranged and their signs changed? That is because
  we're only doing quarter turn rotations. Follow the [right hand rule](https://en.wikipedia.org/wiki/Right-hand_rule)
  yourself (with your thumb to the right as the X axis, index finger pointing up as the Y axis, and
  your middle finger pointing towards you as the Z axis) and see if you can come to the same values
  as I did.

  What this shows is if your current orientation is represented by a [`Quat`](#), and you want to
  rotate the object by another [`Quat`](#), all you have to do is multiply this orientation by a
  rotation. Alternatively you can use [`.rotate()`](#method.rotate), if you're used to that syntax.

  #### Retrieving and Representing a Rotation with Quat

  The most common use case for quaternions is storing them as a represention of a rotation for a
  particular instance. When it comes to building a model for this instance, we retrieve the
  position, scale, and its rotation.

  To retieve a rotation matrix from a Quaternion, use either [`quat.mat4()`](#method.mat4)
  or this crate's glm function [`rotate()`](./fn.rotate.html).
  ```
  # use sawd_glm::{translate, rotate, scale, Vec3, Mat4, Quat};
  # let my_position = Vec3::zero();
  # let my_rotation = Quat::identity();
  # let my_scale = Vec3::one();
  let model: Mat4 = translate(my_position) * rotate(my_rotation) * scale(my_scale);
  ```

  ## Default

  [Default](https://doc.rust-lang.org/nightly/core/default/trait.Default.html) is implemented for
  ease of use with Entity Component System libraries like [specs](https://docs.rs/specs/0.16.1/specs/)
  that require components to implement Default.

  [`Quat::default()`](#method.default) is equivalent to [`Quat::identity()`](#method.identity) and
  I recommend using that function instead to make your code more explicit.
*/

// note: layout is [ x, y, z, w]
//              or [ i, j, k, w]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Quat (pub(crate) [f32; 4]);


impl Quat {
    /// Create a quaternion with explicitly set x, y, z, and w components
    ///
    /// Normally you don't want to create your own, as Quat should be normalized whenever possible.
    /// If you do use this function I recommend using [`.normalize()`](#method.normalize).
    ///
    /// Consider instead [`unit()`](#method.unit), which is equivalent to
    /// `Quat::new(x, y, z, w).normalize()`
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Quat { Quat ( [ x, y, z, w] ) }

    /// Create a unit quaternion based on x, y, z, and w components
    pub fn unit(x: f32, y: f32, z: f32, w: f32) -> Quat { Quat ( [ x, y, z, w] ).normalize() }

    /// Create a unit identity quaternion
    pub fn identity() -> Quat { Quat ( [0.0, 0.0, 0.0, 1.0] ) }

    /// Receive the x value
    pub fn x(&self) -> f32 { self.0[0] }

    /// Receive the y value
    pub fn y(&self) -> f32 { self.0[1] }

    /// Receive the z value
    pub fn z(&self) -> f32 { self.0[2] }

    /// Receive the w value
    pub fn w(&self) -> f32 { self.0[3] }

    /// Receive the 'vector' portion of the quaternion
    ///
    /// Please note that this is **not** the same as the axis you are rotating around
    pub fn xyz(&self) -> Vec3 { Vec3([self.0[0], self.0[1], self.0[2]]) }

    /// Test if this [`Quat`](./struct.Quat.html) is equals to another [`Quat`](./struct.Quat.html)
    /// for each component up to 1e-6
    pub fn equals(&self, other: Quat) -> bool {
        Vec4::from(*self).equals(Vec4::from(other))
        || Vec4::from(*self * -1.0).equals(Vec4::from(other))
    }

    /// Test if this [`Quat`](./struct.Quat.html) is equals to another [`Quat`](./struct.Vec3.html)
    /// for each component up to an epsilon
    pub fn equals_epsilon(&self, other: Quat, epsilon: f32) -> bool {
        Vec4::from(*self).equals_epsilon(Vec4::from(other), epsilon)
            || Vec4::from(*self * -1.0).equals_epsilon(Vec4::from(other), epsilon)
    }

    /// Receive the *absolute value* of each component in this Quat
    pub fn abs(&self) -> Quat {
        Quat([self[0].abs(), self[1].abs(), self[2].abs(), self[3].abs()])
    }

    /// Receive the aggregate average of each component
    pub fn agg_avg(&self) -> f32 {
        (self[0] + self[1] + self[2] + self[3]) / 4.0
    }

    /// Receive the aggregate max of each component
    pub fn agg_max(&self) -> f32 {
        self[0].max(self[1]).max(self[2]).max(self[3])
    }

    /// Receive the aggregate min of each component
    pub fn agg_min(&self) -> f32 {
        self[0].min(self[1]).min(self[2]).min(self[3])
    }

    /// Receive the aggregate product of each component
    pub fn agg_prod(&self) -> f32 {
        self[0] * self[1] * self[2] * self[3]
    }

    /// Receive the aggregate sum of each component
    pub fn agg_sum(&self) -> f32 {
        self[0] + self[1] + self[2] + self[3]
    }

    /// Receive the magnitude of this Quat, should always be 1.0
    pub fn length(&self) -> f32 { ( self[0].powi(2) + self[1].powi(2) + self[2].powi(2) + self[3].powi(2) ).sqrt() }

    /// Receive a normalization of this Quat
    ///
    /// ```rust
    /// # use sawd_glm::{assert_eq_float, Quat};
    /// let v = Quat::new(1.0, 2.0, -0.5, 0.1).normalize();
    /// assert_eq_float!(1.0, v.length());
    /// ```
    pub fn normalize(&self) -> Quat {
        let mag = self.length();
        if mag != 0.0 {
            *self / mag
        }
        else {
            *self
        }
    }

    /// Receive the conjugate of this Quat
    pub fn conjugate(&self) -> Quat {
        Quat ( [
            -self.x(), // x
            -self.y(), // y
            -self.z(), // z
             self.w(), // w
        ] )
    }

    /// Receive the inverse of this Quat
    ///
    /// `quat * -1.0` is not the inverse, it's still the same quaternion!
    pub fn inverse(&self) -> Quat {
        let inv_norm = 1.0 / (
            self.w() * self.w() +
            self.x() * self.x() +
            self.y() * self.y() +
            self.z() * self.z() );
        self.conjugate() * inv_norm
    }

    /// Receive the Mat3 rotation matrix representing this Quat
    pub fn mat3(&self) -> Mat3 {
        // calculate coefficients
        let x2 = self.x() + self.x();
        let y2 = self.y() + self.y();
        let z2 = self.z() + self.z();
        let xx = self.x() * x2;
        let xy = self.x() * y2;
        let xz = self.x() * z2;
        let yy = self.y() * y2;
        let yz = self.y() * z2;
        let zz = self.z() * z2;
        let wx = self.w() * x2;
        let wy = self.w() * y2;
        let wz = self.w() * z2;
        Mat3([
            1.0 - yy - zz,       xy + wz,       xz - wy,
                  xy - wz, 1.0 - xx - zz,       yz + wx,
                  xz + wy,       yz - wx, 1.0 - xx - yy,
        ])
    }

    /// Receive the Mat4 rotation matrix representing this Quat
    pub fn mat4(&self) -> Mat4 {
        // calculate coefficients
        let x2 = self.x() + self.x();
        let y2 = self.y() + self.y();
        let z2 = self.z() + self.z();
        let xx = self.x() * x2;
        let xy = self.x() * y2;
        let xz = self.x() * z2;
        let yy = self.y() * y2;
        let yz = self.y() * z2;
        let zz = self.z() * z2;
        let wx = self.w() * x2;
        let wy = self.w() * y2;
        let wz = self.w() * z2;
        Mat4([
            1.0 - yy - zz,       xy + wz,       xz - wy, 0.0,
                  xy - wz, 1.0 - xx - zz,       yz + wx, 0.0,
                  xz + wy,       yz - wx, 1.0 - xx - yy, 0.0,
                      0.0,           0.0,           0.0, 1.0,
        ])

    }

    /// Rotate this Quat by the rotation of another Quat
    pub fn rotate(&self, rotation: Quat) -> Quat {
        *self * rotation
    }

    /// Receive the rotation at a particular point in time in a Spherical Linear Interpolation
    ///
    /// This function calls `.acos()` which can be expensive, so reserve this for special occasions!
    /// Consider using its cheaper cousin [`lerp()`](#method.lerp) if you know for sure the
    /// difference in rotation is small.
    pub fn slerp(&self, mut to: Quat, t: f32) -> Quat {
        let scale0: f32;
        let scale1: f32;

        // calc dot
        let mut dot = self.x() * to.x() + self.y() * to.y() + self.z() * to.z() + self.w() * to.w();

        // adjust signs (if necessary)
        if dot < 0.0 {
            dot = -dot;

            to = to * -1.0;
        }

        // calculate coefficients
        if 1.0 - dot > NEAR_ZERO {
            // standard case (slerp)
            let omega = dot.acos();
            let sinom = omega.sin();
            scale0 = ((1.0 - t) * omega).sin() / sinom;
            scale1 = (t * omega).sin() / sinom;
        } else {
            // "self" and "to" quaternions are very close
            //  ... so we can do a linear interpolation
            scale0 = 1.0 - t;
            scale1 = t;
        }

        // calculate to values
        Quat::new(
            scale0 * self.x() + scale1 * to.x(),
            scale0 * self.y() + scale1 * to.y(),
            scale0 * self.z() + scale1 * to.z(),
            scale0 * self.w() + scale1 * to.w(),
        )
    }

    /// Receive the rotation at a particular point in time in a Linear Interpolation
    pub fn lerp(&self, mut to: Quat, t: f32) -> Quat {

        // calc cosine
        let cosom = self.x() * to.x() + self.y() * to.y() + self.z() * to.z() + self.w() * to.w();

        // adjust signs (if necessary)
        if cosom < 0.0 {
            to = to * -1.0;
        }

        // linear interpolation
        let scale0 = 1.0 - t;
        let scale1 = t;

        // calculate to values
        Quat::new(
            scale0 * self.x() + scale1 * to.x(),
            scale0 * self.y() + scale1 * to.y(),
            scale0 * self.z() + scale1 * to.z(),
            scale0 * self.w() + scale1 * to.w(),
        )
    }

    /// Build a rotation from one axis to another. Both `from` and `to` must be unit vectors!
    ///
    /// This resulting quaternion represents the rotation necessary to rotate the `from` [`Vec3`](./struct.Vec3.html)
    /// to the `to` [`Vec3`](./struct.Vec3.html).
    ///
    /// ```
    /// # use sawd_glm::{Vec3, Quat};
    /// let from = Vec3::new(1.0, 0.0, 0.0);
    /// let to = Vec3::new(0.0, 1.0, 1.0).normalize();
    /// let rotation = Quat::from_two_axis(from, to);
    /// assert!(to.equals(rotation * from));
    /// ```
    pub fn from_two_axis(from: Vec3, to: Vec3) -> Quat {
        let mut tx: f32;
        let mut ty: f32;
        let mut tz: f32;
        let dist: f32;

        // get dot product of two vectors
        let cost = from.x() * to.x() + from.y() * to.y() + from.z() * to.z();

        // check if parallel
        if cost > 0.99999 {
            return Quat::new(1.0, 0.0, 0.0, 0.0)
        }
        else if cost < -0.99999 {     // check if opposite
            // check if we can use cross product of from vector with [1, 0, 0]
            tx = 0.0;
            ty = from.x();
            tz = -from.y();

            let len = (ty*ty + tz*tz).sqrt();

            if len < NEAR_ZERO {
                // nope! we need cross product of from vector with [0, 1, 0]
                tx = -from.z();
                ty = 0.0;
                tz = from.x();
            }

            // normalize
            dist = 1.0 / (tx*tx + ty*ty + tz*tz).sqrt();

            tx *= dist;
            ty *= dist;
            tz *= dist;

            return Quat::new(0.0, tx, ty, tz)
        }

        // ... else we can just cross two vectors
        tx = from.y() * to.z() - from.z() * to.y();
        ty = from.z() * to.x() - from.x() * to.z();
        tz = from.x() * to.y() - from.y() * to.x();

        dist = 1.0 / (tx*tx + ty*ty + tz*tz).sqrt();

        tx *= dist;
        ty *= dist;
        tz *= dist;

        // we have to use half-angle formulae (sin^2 t = ( 1 - cos (2t) ) /2)
        let ss = (0.5 * (1.0 - cost)).sqrt();

        tx *= ss;
        ty *= ss;
        tz *= ss;

        // scale the axis to get the normalized quaternion
        // cos^2 t = ( 1 + cos (2t) ) / 2
        // w part is cosine of half the rotation angle
        Quat::new(tx, ty, tz, (0.5 * (1.0 + cost)).sqrt())
    }

    /// Build a Quat based entirely on Euler axis rotations
    ///
    /// Rotation order is Z -> Y -> X
    pub fn from_euler_xyz_rotation(x_rotation: f32, y_rotation: f32, z_rotation: f32) -> Quat {
        let cr = (z_rotation/2.0).cos();
        let cp = (y_rotation/2.0).cos();
        let cy = (x_rotation/2.0).cos();
        let sr = (z_rotation/2.0).sin();
        let sp = (y_rotation/2.0).sin();
        let sy = (x_rotation/2.0).sin();
        let cycp = cy * cp;
        let sysp = sy * sp;
        let cysp = cy * sp;
        let sycp = sy * cp;
        Quat::unit(
            sycp * cr - cysp * sr,
            cysp * cr + sycp * sr,
            cycp * sr - sysp * cr,
            cycp * cr + sysp * sr,
        )
    }

    /// Build a Quat from an axis or rotation, and a float of radians around that axis
    ///
    /// `axis` must be normalized!
    pub fn from_angle_axis(angle: f32, axis: Vec3) -> Quat {
        // scalar
        let scale = (angle / 2.0).sin();

        Quat ([
            axis[0] * scale,
            axis[1] * scale,
            axis[2] * scale,
            (angle / 2.0).cos(),
        ])
    }

    /// Convert this Quat into a [`Vec3`](./struct.Vec3.html) containing rotations for the X, Y, and
    /// Z axes. Rotation order is ZYX.
    ///
    /// Note: Euler is very strange, especially around PI/2. Quaternions are highly recommended.
    pub fn to_euler_xyz_rotation(&self) -> Vec3 {
        let r11 = 2.0*(self.y()*self.z() + self.w()*self.x());
        let r12 = self.w()*self.w() - self.x()*self.x() - self.y()*self.y() + self.z()*self.z();
        let r21 = -2.0*(self.x()*self.z() - self.w()*self.y());
        let r31 = 2.0*(self.x()*self.y() + self.w()*self.z());
        let r32 = self.w()*self.w() + self.x()*self.x() - self.y()*self.y() - self.z()*self.z();

        // simple clamp
        let r21 = if r21 > 1.0 { 1.0 } else if r21 < -1.0 { -1.0 } else { r21 };

        Vec3::new(
            r11.atan2(r12),
            r21.asin(),
            r31.atan2(r32),
        )
    }

    /// Convert this Quat into a scalar and Vec3 tuple. Scalar is radians around the Vec3 axis.
    pub fn to_angle_axis(&self) -> (f32, Vec3) {

        let angle = 2.0 * self.w().acos();
        let scale = (angle / 2.0).sin();

        // if it's not pretty much zero
        if scale > NEAR_ZERO
        {
            ( angle, Vec3::new(self.x() / scale, self.y() / scale, self.z() / scale) )
        }
        else {
            ( 0.0, Vec3::new(0.0, 0.0, 1.0) )
        }
    }

    /// Scale the radians of rotation around the inner axis
    pub fn scale_angle(&self, s: f32) -> Quat {
        let (angle, vec) = self.to_angle_axis();
        Self::from_angle_axis(angle * s, vec)
    }

    /// Set the radians of rotation around the inner axis
    pub fn set_angle(&self, s: f32) -> Quat {
        let (_, vec) = self.to_angle_axis();
        Self::from_angle_axis(s, vec)
    }

    /// Create an orientation looking from a `pos` to a `target` with an up vector to prevent
    /// rolling
    ///
    /// See also: [`look_at()`](./fn.lookAt.html)
    pub fn look_at(pos: Vec3, target: Vec3, up: Vec3) -> Quat {
        Quat::from(look_at(pos, target, up))
    }
}

//------------------------------------------------------------------------------------------------//
// OPERATORS                                                                                      //
//------------------------------------------------------------------------------------------------//
impl std::ops::Mul<Quat> for Quat {
    type Output = Quat;

    /// Rotate a quaternion by another rotation
    fn mul(self, rhs: Quat) -> Quat {
        Quat ( [
            self.x() * rhs.w() + self.w() * rhs.x() + self.y() * rhs.z() - self.z() * rhs.y(), // x
            self.y() * rhs.w() + self.w() * rhs.y() + self.z() * rhs.x() - self.x() * rhs.z(), // y
            self.z() * rhs.w() + self.w() * rhs.z() + self.x() * rhs.y() - self.y() * rhs.x(), // z
            self.w() * rhs.w() - self.x() * rhs.x() - self.y() * rhs.y() - self.z() * rhs.z(), // w
        ] ).normalize()
    }
}

impl std::ops::MulAssign<Quat> for Quat {

    /// Rotate a quaternion by another rotation
    fn mul_assign(&mut self, rhs: Quat) {
        *self = *self * rhs;
    }
}

// Quat * Vec3 = Vec3 rotated by Quat
impl std::ops::Mul<Vec3> for Quat {
    type Output = Vec3;

    /// Rotate a [`Vec3`](./struct.Vec3.html) by this quaternion
    fn mul(self, rhs: Vec3) -> Vec3 {
        let q_xyz: Vec3 = self.xyz();
        let t: Vec3 = q_xyz.cross(rhs) * 2.0;
        let u: Vec3 = q_xyz.cross(t);
        rhs + (t * self.w()) + u
    }
}

impl std::ops::Mul<f32> for Quat {
    type Output = Quat;

    /// Scale this quaternion, warning: likely to no longer be a unit quaternion after this
    fn mul(self, rhs: f32) -> Quat {
        // may not be a unit quaternion after this
        Quat::new(self.x() * rhs, self.y() * rhs, self.z() * rhs, self.w() * rhs)
    }
}

impl std::ops::MulAssign<f32> for Quat {
    /// Scale this quaternion, warning: likely to no longer be a unit quaternion after this
    fn mul_assign(&mut self, rhs: f32) {
        // may not be a unit quaternion after this
        self[0] *= rhs;
        self[1] *= rhs;
        self[2] *= rhs;
        self[3] *= rhs;
    }
}

impl std::ops::Div<f32> for Quat {
    type Output = Quat;


    /// Scale this quaternion, warning: likely to no longer be a unit quaternion after this
    fn div(self, rhs: f32) -> Quat {
        if rhs == 0.0 { panic!("Cannot divide by zero. (Quat / 0.0)"); }
        Quat ( [
            self[0] / rhs,
            self[1] / rhs,
            self[2] / rhs,
            self[3] / rhs,
        ] )
    }
}

impl std::ops::DivAssign<f32> for Quat {

    /// Scale this quaternion, warning: likely to no longer be a unit quaternion after this
    fn div_assign(&mut self, rhs: f32) {
        // may not be a unit quaternion after this
        self[0] /= rhs;
        self[1] /= rhs;
        self[2] /= rhs;
        self[3] /= rhs;
    }
}

impl std::ops::Index<usize> for Quat {
    type Output = f32;

    /// Obtain a reference to a component of this Quat, order is x, y, z, w.
    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Quat {
    /// Obtain a mutable reference to a component of this Quat, order is x, y, z, w.
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

//------------------------------------------------------------------------------------------------//
// FROM                                                                                           //
//------------------------------------------------------------------------------------------------//

impl From<Mat4> for Quat {
    /// Convert from a [`Mat4`](./struct.Mat4.html) rotation matrix to [`Quat`](./struct.Quat.html)
    ///
    /// ```
    /// # use sawd_glm::{Vec3, Quat};
    /// let q = Quat::from_angle_axis(0.4572, Vec3::new(-1.4, 2.3, 10.1));
    ///
    /// assert!(q.equals(Quat::from(q.mat4())));
    /// ```
    fn from(m: Mat4) -> Self {
        let tr: f32 = m[0] + m[5] + m[10];

        // check the diagonal
        if tr > 0.0 {
            let s: f32 = (tr + 1.0).sqrt();
            let si: f32 = 0.5 / s;
            Quat([
                (m[6] - m[9]) * si,
                (m[8] - m[2]) * si,
                (m[1] - m[4]) * si,
                s / 2.0,
            ])
        }
        // diagonal is negative
        else {
            let mut q: [f32; 4] = [0.0; 4];
            let mut i: usize;
            let j: usize;
            let k: usize;
            let nxt: [usize; 3] = [1, 2, 0];

            i = 0;
            if m[5] > m[0] { i = 1; }
            if m[10] > m[(i,i)] { i = 2; }
            j = nxt[i];
            k = nxt[j];
            let mut s: f32 = ((m[(i,i)] - (m[(j,j)] + m[(k,k)])) + 1.0).sqrt();
            q[i] = s * 0.5;
            if s != 0.0 { s = 0.5 / s; }
            q[3] = (m[(j,k)] - m[(k,j)]) * s;
            q[j] = (m[(i,j)] + m[(j,i)]) * s;
            q[k] = (m[(i,k)] + m[(k,i)]) * s;
            Quat([ q[0], q[1], q[2], q[3] ])
        }
    }
}

impl From<Vec4> for Quat {
    /// Cast a Vec4 to a Quat, no transformations
    fn from(f: Vec4) -> Self {
        Quat::new(f[0], f[1], f[2], f[3])
    }
}

//impl From<Vec3> for Quat {
//    fn from(f: Vec3) -> Self {
//        Quat::from_euler_ypr(f.x(), f.y(), f.z())
//    }
//}

//------------------------------------------------------------------------------------------------//
// OTHER                                                                                          //
//------------------------------------------------------------------------------------------------//
impl Default for Quat {
    /// Creates the default value for [`Quat`](./struct.Quat.html), [`Quat::identity()`]([`Quat`](./struct.Quat.html#method.identity)
    fn default() -> Self {
        Quat::identity()
    }
}

impl Display for Quat {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "({} + {} i + {} j + {} k)", self.w(), self.x(), self.y(), self.z())
    }
}