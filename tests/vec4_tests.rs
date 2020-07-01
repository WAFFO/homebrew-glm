extern crate sawd_glm;

use sawd_glm::{Vec3, Vec4};

#[test]
fn index() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(4.0, v[3]);
    let u = &v;
    assert_eq!(2.0, u[1]);
}

#[test]
fn index_mut() {
    let mut v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    v[2] = 6.0;
    assert_eq!(6.0, v[2]);
    let u = &mut v;
    u[1] = 9.9;
    assert_eq!(9.9, u[1]);
}

#[test]
fn neg() {
    let v = Vec4::new(0.0, 1.0, -2.0, 4.0);

    assert_eq!(Vec4::new(0.0, -1.0, 2.0, -4.0), -v);
}

#[test]
fn add() {
    let a = Vec4::new(0.0, 1.0, 2.0, 4.0);
    let b = Vec4::new(-1.0, 0.0, 1.0, 4.0);

    assert_eq!(Vec4::new(-1.0, 1.0, 3.0, 8.0), a + b);

    let mut c = Vec4::new(0.0, 0.0, 0.0, 0.0);
    c += b;

    assert_eq!(b, c);
}

#[test]
fn sub() {
    let a = Vec4::new(0.0, 1.0, 2.0, 4.0);
    let b = Vec4::new(-1.0, 0.0, 1.0, 3.0);

    assert_eq!(Vec4::all(1.0), a - b);

    let mut c = Vec4::new(0.0, 0.0, 0.0, 0.0);
    c -= b;

    assert_eq!(-b, c);
}


#[test]
fn swizzle() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);

    assert_eq!(Vec3::new(1.0, 1.0, 1.0), v.xxx());
    assert_eq!(Vec3::new(3.0, 2.0, 1.0), v.zyx());
    assert_eq!(Vec3::new(4.0, 3.0, 2.0), v.wzy());

    assert_eq!(Vec4::new(1.0, 1.0, 1.0, 1.0), v.xxxx());
    assert_eq!(Vec4::new(3.0, 2.0, 3.0, 1.0), v.zyzx());
    assert_eq!(Vec4::new(3.0, 2.0, 3.0, 4.0), v.zyzw());
    assert_eq!(Vec4::new(4.0, 3.0, 2.0, 1.0), v.wzyx());
    assert_eq!(v, v.xyzw());
}