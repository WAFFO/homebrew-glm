extern crate sawd_glm;

use sawd_glm::{Vec3, Vec4};

#[test]
fn index() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(3.0, v[2]);
    let u = &v;
    assert_eq!(2.0, u[1]);
}

#[test]
fn index_mut() {
    let mut v = Vec3::new(1.0, 2.0, 3.0);
    v[2] = 6.0;
    assert_eq!(6.0, v[2]);
    let u = &mut v;
    u[1] = 9.9;
    assert_eq!(9.9, u[1]);
}

#[test]
fn neg() {
    let v = Vec3::new(0.0, 1.0, -2.0);

    assert_eq!(Vec3::new(0.0, -1.0, 2.0), -v);
}

#[test]
fn add() {
    let a = Vec3::new(0.0, 1.0, 2.0);
    let b = Vec3::new(-1.0, 0.0, 1.0);

    assert_eq!(Vec3::new(-1.0, 1.0, 3.0), a + b);

    let mut c = Vec3::new(0.0, 0.0, 0.0);
    c += b;

    assert_eq!(b, c);
}

#[test]
fn sub() {
    let a = Vec3::new(0.0, 1.0, 2.0);
    let b = Vec3::new(-1.0, 0.0, 1.0);

    assert_eq!(Vec3::all(1.0), a - b);

    let mut c = Vec3::new(0.0, 0.0, 0.0);
    c -= b;

    assert_eq!(-b, c);
}

#[test]
fn refraction() {
    let i = Vec3::all(1.5);
    let n = Vec3::Y_AXIS;

    println!("refraction: {}", i.refraction(n, 1.0));
    println!("refraction: {}", i.refraction(n, 0.5));
    // assert_eq!(Vec3::new(1.0, -1.0, 1.0), i.reflection(n));
}

#[test]
fn swizzle() {
    let v = Vec3::new(1.0, 2.0, 3.0);

    assert_eq!(Vec3::new(1.0, 1.0, 1.0), v.xxx());
    assert_eq!(Vec3::new(3.0, 2.0, 1.0), v.zyx());
    assert_eq!(v, v.xyz());

    assert_eq!(Vec4::new(1.0, 1.0, 1.0, 1.0), v.xxxx());
    assert_eq!(Vec4::new(3.0, 2.0, 3.0, 1.0), v.zyzx());
}