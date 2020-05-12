extern crate sawd_glm as glm;

use glm::{Vec3, look_at, Quat};

use std::f32::consts::PI;

#[test]
pub fn from_angle_axis_test() {
    let vec = Vec3::unit(1.0, 2.0, 3.0);
    let angle = PI/6.0;

    let quat = Quat::from_angle_axis(angle, vec);

    let (_, axis) = quat.to_angle_axis();

    println!("vec: {}, quat: {}, axis: {}", vec, quat, axis);
    assert!(vec.equals(axis));
}

#[test]
pub fn look_at_test() {
    let position = Vec3::new(0.0, 0.0, 0.0);
    let target = Vec3::new(1.0, 1.0, 1.0);

    let mat = look_at(position, target, Vec3::Y_AXIS);
    let quat = Quat::look_at(position, target, Vec3::Y_AXIS);

    assert!(quat.mat4().equals(mat));

    let vertex = Vec3::new(10.0, -6.5, 6.0);
    let quat_rotation = quat * vertex;
    let mat_rotation = (mat * vertex.vec4(0.0)).xyz();

    assert!(quat_rotation.equals(mat_rotation));
}

#[test]
pub fn from_euler_xyz_rotation_test() {
    let rotation_z2y = Quat::from_two_axis(Vec3::Z_AXIS, Vec3::Y_AXIS);
    let rotation_y2x = Quat::from_two_axis(Vec3::Y_AXIS, Vec3::X_AXIS);

    let rotation_2a = rotation_z2y * rotation_y2x;

    let rotation_xyz = Quat::from_euler_xyz_rotation(0.0, -PI/2.0, -PI/2.0);

    println!("rotation_2a: {}, rotation_xyz: {}", rotation_2a, rotation_xyz);
    assert!(rotation_2a.equals(rotation_xyz));

    let rotation_2a = rotation_y2x * rotation_z2y;

    let rotation_xyz = Quat::from_euler_xyz_rotation(-PI/2.0, 0.0, -PI/2.0);

    println!("rotation_2a: {}, rotation_xyz: {}", rotation_2a, rotation_xyz);
    assert!(rotation_2a.equals(rotation_xyz));
}

#[test]
pub fn to_euler_xyz_rotation_test() {
    use std::f32::consts::PI;
    let euler = Vec3::new(PI/3.0, PI/3.0, PI/4.0);

    println!("left: {}, right: {}", euler, Quat::from_euler_xyz_rotation(euler.x(), euler.y(), euler.z()).to_euler_xyz_rotation());
    println!("quat: {}", Quat::from_euler_xyz_rotation(euler.x(), euler.y(), euler.z()));

    assert!(euler.equals(
        Quat::from_euler_xyz_rotation(
            euler.x(),
            euler.y(),
            euler.z(),
        ).to_euler_xyz_rotation()
    ));
}