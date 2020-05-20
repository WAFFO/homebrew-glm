use crate::{GVec3, GVec4, Mat3, Mat4, GMat4, GMat3};
use std::fmt::{Debug, Display};
use std::ops::{Add, Sub, AddAssign, SubAssign, Neg, Mul, MulAssign, Div, DivAssign, Rem, RemAssign};

/// Trait for scalars: f32 or f64
pub trait Scalar:
    Debug +
    Display +
    Copy +
    Clone +
    PartialEq +
    PartialOrd +
    Neg<Output = Self> +
    Add<Output = Self> +
    AddAssign +
    Sub<Output = Self> +
    SubAssign +
    Mul<Output = Self> +
    MulAssign +
    Div<Output = Self> +
    DivAssign +
    Rem<Output = Self> +
    RemAssign +
    num_traits::Float +
    num_traits::cast::FromPrimitive
{
    fn cast(f: f64) -> Self {
        Self::from_f64(f).unwrap()
    }
}

/// Trait for vectors: [Vec3](#) or [Vec4](#)
pub trait Vector: Default + Copy + Clone + PartialEq{}

/// Trait for matrices: [Mat3](#) or [Mat4](#)
pub trait Matrix: Default + Copy + Clone + PartialEq{}

impl Scalar for f32{}
impl Scalar for f64{}
impl<T: Scalar> Vector for GVec3<T>{}
impl<T: Scalar> Vector for GVec4<T>{}
impl<T: Scalar> Matrix for GMat3<T>{}
impl<T: Scalar> Matrix for GMat4<T>{}
