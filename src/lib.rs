
/*! # Homebrew-glm - Lightweight, Easy to Use, Graphics Focused Linear Algebra Library

Description of stuff goes here.

And here.
And here.

Maybe here too.

## Header 2

Content 2.

*/

#[macro_use]
pub mod macros;

pub(crate) mod glm;

pub use glm::*;
pub use glm::vec3::Vec3;
pub use glm::vec4::Vec4;
pub use glm::mat3::Mat3;
pub use glm::mat4::Mat4;
pub use glm::quat::Quat;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
