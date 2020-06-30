use sawd_glm::Mat3;

#[test]
fn inverse() {
    let mat = Mat3::mat3([
        1.0, 2.0, 3.0,
        4.0, 1.0, 6.0,
        7.0, 8.0, 1.0,
    ]);

    let inv = mat.inverse().unwrap();

    assert!(Mat3::identity().equals(mat * inv));
}