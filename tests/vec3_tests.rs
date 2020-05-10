extern crate homebrew_glm;

use homebrew_glm::Vec3;

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