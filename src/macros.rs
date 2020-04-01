
#[doc(hidden)]
macro_rules! vector_operations {
    ( $VecN:ident, { $($i:expr),+ } ) => {
        impl std::ops::Index<usize> for $VecN {
            type Output = f32;
        
            fn index(&self, index: usize) -> &f32 {
                &self.0[index]
            }
        }
        
        
        impl std::ops::IndexMut<usize> for $VecN {
            fn index_mut(&mut self, index: usize) -> &mut f32 {
                &mut self.0[index]
            }
        }

        impl std::ops::Neg for $VecN {
            type Output = $VecN;

            fn neg(self) -> $VecN {
                $VecN([
                    $(-self.0[$i],)+
                ])
            }
        }
        
        impl std::ops::Add for $VecN {
            type Output = $VecN;
        
            fn add(self, other: $VecN) -> $VecN {
                $VecN ( [
                    $(self.0[$i] + other.0[$i],)+
                ] )
            }
        }
        
        impl std::ops::AddAssign for $VecN {
            fn add_assign(&mut self, other: $VecN) {
                $(self.0[$i] += other.0[$i];)+
            }
        }
        
        impl std::ops::Sub for $VecN {
            type Output = $VecN;
        
            fn sub(self, other: $VecN) -> $VecN {
                $VecN ( [
                    $(self.0[$i] - other.0[$i],)+
                ] )
            }
        }
        
        impl std::ops::SubAssign for $VecN {
            fn sub_assign(&mut self, other: $VecN) {
                $(self.0[$i] -= other.0[$i];)+
            }
        }
        
        impl std::ops::Mul<f32> for $VecN {
            type Output = $VecN;
        
            fn mul(self, rhs: f32) -> $VecN {
                $VecN ( [
                    $(self.0[$i] * rhs,)+
                ] )
            }
        }

        impl std::ops::MulAssign<f32> for $VecN {
            fn mul_assign(&mut self, rhs: f32) {
                $(self.0[$i] *= rhs;)+
            }
        }
        
        impl std::ops::Div<f32> for $VecN {
            type Output = $VecN;
        
            fn div(self, rhs: f32) -> $VecN {
                if rhs == 0.0 { panic!("Cannot divide by zero. ($VecN / 0.0)"); }
                $VecN ( [
                    $(self.0[$i] / rhs,)+
                ] )
            }
        }

        impl std::ops::DivAssign<f32> for $VecN {
            fn div_assign(&mut self, rhs: f32) {
                if rhs == 0.0 { panic!("Cannot divide by zero. ($VecN / 0.0)"); }
                $(self.0[$i] /= rhs;)+
            }
        }
        
        impl std::ops::Rem<f32> for $VecN {
            type Output = $VecN;
        
            fn rem(self, rhs: f32) -> $VecN {
                if rhs == 0.0 { panic!("Cannot divide by zero. ($VecN % 0.0)"); }
                $VecN ( [
                    $(self.0[$i] % rhs,)+
                ] )
            }
        }
        
        impl std::ops::RemAssign<f32> for $VecN {
        
            fn rem_assign(&mut self, rhs: f32) {
                if rhs == 0.0 { panic!("Cannot divide by zero. ($VecN % 0.0)"); }
                $(self.0[$i] %= rhs;)+
            }
        }
    }
}