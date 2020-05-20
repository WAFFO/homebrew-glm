
/*! # SaWD-glm - Simple and Well Documented, Graphics Focused Linear Algebra Library

  This library is based on the C OpenGL library [GLM](https://glm.g-truc.net/). That means:

  - Matrices are [Column-Major](https://en.wikipedia.org/wiki/Row-_and_column-major_order)
    (data is stored by column, not row)
  - Transformations flow from right to left (vectors need to be on the right of a matrix)
  - Math follows the [right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule)
  - Y points up, and Z points towards the screen
    (at least until the projection matrix, see [perspective_fov](./fn.perspective_fov.html))

  ## Goals for this library

  This library was created with these 4 goals in mind:

  #### • Straight Forward

  This library should be _simple_ and _easy to use_. Types aren't heavily abstracted, so any
  compilation errors encountered will be easy to understand, and what types a function requires
  can't be misinterpreted. By default, each type has the copy trait, and no functions take a
  reference, removing the small road-bumps a developer might encounter by forgetting to pass by
  reference. This way we leverage the power of Rust and let the compiler optimize the memory
  allocation.

  #### • IDE Friendly

  Important functionality should not be hidden behind macros, aliases, or layers of types. I want
  IDEs to be able to easily identify the outcome of operators and what functions are available to
  them.

  #### • Complete Documentation

  This goes beyond just giving descriptions for functions, by explaining the finer details of
  complicated functionality and application. Giving examples where necessary.

  #### • Lightweight

  Little to no dependencies, what you see here is what you get. Code is fast and efficient. The only
  crate we use is [`num-traits`](https://docs.rs/num-traits/0.2.11/num_traits/index.html) which is
  an official rust crate that provides the generic trait [`Float`](https://docs.rs/num-traits/0.2.11/num_traits/float/trait.Float.html).

*/

extern crate num_traits;

pub mod traits;
pub(crate) mod glm;

pub use glm::*;
pub use glm::vec3::{GVec3, Vec3};
pub use glm::vec4::{GVec4, Vec4};
pub use glm::mat3::{GMat3, Mat3};
pub use glm::mat4::{GMat4, Mat4};
pub use glm::quat::{GQuat, Quat};
