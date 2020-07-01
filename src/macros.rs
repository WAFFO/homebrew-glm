
#[doc(hidden)]
#[allow(unused_macros)]
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

#[doc(hidden)]
macro_rules! vector_swizzle_3 {
    () => {
        vector_swizzle_inner_vec!(
            Vec3,
            [
                xxx, xxy, xxz, xyx, xyy, xyz, xzx, xzy, xzz, yxx, yxy, yxz, yyx, yyy, yyz, yzx, yzy,
                yzz, zxx, zxy, zxz, zyx, zyy, zyz, zzx, zzy, zzz
            ],
            [
                [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2], [0,2,0], [0,2,1], [0,2,2],
                [1,0,0], [1,0,1], [1,0,2], [1,1,0], [1,1,1], [1,1,2], [1,2,0], [1,2,1], [1,2,2],
                [2,0,0], [2,0,1], [2,0,2], [2,1,0], [2,1,1], [2,1,2], [2,2,0], [2,2,1], [2,2,2]
            ]
        );
        vector_swizzle_inner_vec!(
            Vec4,
            [
                xxxx, xxxy, xxxz, xxyx, xxyy, xxyz, xxzx, xxzy, xxzz,
                xyxx, xyxy, xyxz, xyyx, xyyy, xyyz, xyzx, xyzy, xyzz,
                xzxx, xzxy, xzxz, xzyx, xzyy, xzyz, xzzx, xzzy, xzzz,

                yxxx, yxxy, yxxz, yxyx, yxyy, yxyz, yxzx, yxzy, yxzz,
                yyxx, yyxy, yyxz, yyyx, yyyy, yyyz, yyzx, yyzy, yyzz,
                yzxx, yzxy, yzxz, yzyx, yzyy, yzyz, yzzx, yzzy, yzzz,

                zxxx, zxxy, zxxz, zxyx, zxyy, zxyz, zxzx, zxzy, zxzz,
                zyxx, zyxy, zyxz, zyyx, zyyy, zyyz, zyzx, zyzy, zyzz,
                zzxx, zzxy, zzxz, zzyx, zzyy, zzyz, zzzx, zzzy, zzzz
            ],
            [
                [0,0,0,0], [0,0,0,1], [0,0,0,2], [0,0,1,0], [0,0,1,1], [0,0,1,2], [0,0,2,0], [0,0,2,1], [0,0,2,2],
                [0,1,0,0], [0,1,0,1], [0,1,0,2], [0,1,1,0], [0,1,1,1], [0,1,1,2], [0,1,2,0], [0,1,2,1], [0,1,2,2],
                [0,2,0,0], [0,2,0,1], [0,2,0,2], [0,2,1,0], [0,2,1,1], [0,2,1,2], [0,2,2,0], [0,2,2,1], [0,2,2,2],

                [1,0,0,0], [1,0,0,1], [1,0,0,2], [1,0,1,0], [1,0,1,1], [1,0,1,2], [1,0,2,0], [1,0,2,1], [1,0,2,2],
                [1,1,0,0], [1,1,0,1], [1,1,0,2], [1,1,1,0], [1,1,1,1], [1,1,1,2], [1,1,2,0], [1,1,2,1], [1,1,2,2],
                [1,2,0,0], [1,2,0,1], [1,2,0,2], [1,2,1,0], [1,2,1,1], [1,2,1,2], [1,2,2,0], [1,2,2,1], [1,2,2,2],

                [2,0,0,0], [2,0,0,1], [2,0,0,2], [2,0,1,0], [2,0,1,1], [2,0,1,2], [2,0,2,0], [2,0,2,1], [2,0,2,2],
                [2,1,0,0], [2,1,0,1], [2,1,0,2], [2,1,1,0], [2,1,1,1], [2,1,1,2], [2,1,2,0], [2,1,2,1], [2,1,2,2],
                [2,2,0,0], [2,2,0,1], [2,2,0,2], [2,2,1,0], [2,2,1,1], [2,2,1,2], [2,2,2,0], [2,2,2,1], [2,2,2,2]
            ]
        );
    }
}

#[doc(hidden)]
macro_rules! vector_swizzle_4 {
    () => {
        vector_swizzle_inner_vec!(
            Vec3,
            [
                xxx, xxy, xxz, xxw, xyx, xyy, xyz, xyw, xzx,
                xzy, xzz, xzw, xwx, xwy, xwz, xww, yxx, yxy,
                yxz, yxw, yyx, yyy, yyz, yyw, yzx, yzy, yzz,
                yzw, ywx, ywy, ywz, yww, zxx, zxy, zxz, zxw,
                zyx, zyy, zyz, zyw, zzx, zzy, zzz, zzw, zwx,
                zwy, zwz, zww, wxx, wxy, wxz, wxw, wyx, wyz,
                wyw, wzx, wzy, wzz, wzw, wwx, wwy, wwz, www
            ],
            [
                [0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,1,0], [0,1,1], [0,1,2], [0,1,3], [0,2,0],
                [0,2,1], [0,2,2], [0,2,3], [0,3,0], [0,3,1], [0,3,2], [0,3,3], [1,0,0], [1,0,1],
                [1,0,2], [1,0,3], [1,1,0], [1,1,1], [1,1,2], [1,1,3], [1,2,0], [1,2,1], [1,2,2],
                [1,2,3], [1,3,0], [1,3,1], [1,3,2], [1,3,3], [2,0,0], [2,0,1], [2,0,2], [2,0,3],
                [2,1,0], [2,1,1], [2,1,2], [2,1,3], [2,2,0], [2,2,1], [2,2,2], [2,2,3], [2,3,0],
                [2,3,1], [2,3,2], [2,3,3], [3,0,0], [3,0,1], [3,0,2], [3,0,3], [3,1,0], [3,1,2],
                [3,1,3], [3,2,0], [3,2,1], [3,2,2], [3,2,3], [3,3,0], [3,3,1], [3,3,2], [3,3,3]
            ]
        );
        vector_swizzle_inner_vec!(
            Vec4,
            [
                xxxx, xxxy, xxxz, xxxw, xxyx, xxyy, xxyz, xxyw, xxzx,
                xxzy, xxzz, xxzw, xxwx, xxwy, xxwz, xxww, xyxx, xyxy,
                xyxz, xyxw, xyyx, xyyy, xyyz, xyyw, xyzx, xyzy, xyzz,
                xyzw, xywx, xywy, xywz, xyww, xzxx, xzxy, xzxz, xzxw,
                xzyx, xzyy, xzyz, xzyw, xzzx, xzzy, xzzz, xzzw, xzwx,
                xzwy, xzwz, xzww, xwxx, xwxy, xwxz, xwxw, xwyx, xwyz,
                xwyw, xwzx, xwzy, xwzz, xwzw, xwwx, xwwy, xwwz, xwww,

                yxxx, yxxy, yxxz, yxxw, yxyx, yxyy, yxyz, yxyw, yxzx,
                yxzy, yxzz, yxzw, yxwx, yxwy, yxwz, yxww, yyxx, yyxy,
                yyxz, yyxw, yyyx, yyyy, yyyz, yyyw, yyzx, yyzy, yyzz,
                yyzw, yywx, yywy, yywz, yyww, yzxx, yzxy, yzxz, yzxw,
                yzyx, yzyy, yzyz, yzyw, yzzx, yzzy, yzzz, yzzw, yzwx,
                yzwy, yzwz, yzww, ywxx, ywxy, ywxz, ywxw, ywyx, ywyz,
                ywyw, ywzx, ywzy, ywzz, ywzw, ywwx, ywwy, ywwz, ywww,

                zxxx, zxxy, zxxz, zxxw, zxyx, zxyy, zxyz, zxyw, zxzx,
                zxzy, zxzz, zxzw, zxwx, zxwy, zxwz, zxww, zyxx, zyxy,
                zyxz, zyxw, zyyx, zyyy, zyyz, zyyw, zyzx, zyzy, zyzz,
                zyzw, zywx, zywy, zywz, zyww, zzxx, zzxy, zzxz, zzxw,
                zzyx, zzyy, zzyz, zzyw, zzzx, zzzy, zzzz, zzzw, zzwx,
                zzwy, zzwz, zzww, zwxx, zwxy, zwxz, zwxw, zwyx, zwyz,
                zwyw, zwzx, zwzy, zwzz, zwzw, zwwx, zwwy, zwwz, zwww,

                wxxx, wxxy, wxxz, wxxw, wxyx, wxyy, wxyz, wxyw, wxzx,
                wxzy, wxzz, wxzw, wxwx, wxwy, wxwz, wxww, wyxx, wyxy,
                wyxz, wyxw, wyyx, wyyy, wyyz, wyyw, wyzx, wyzy, wyzz,
                wyzw, wywx, wywy, wywz, wyww, wzxx, wzxy, wzxz, wzxw,
                wzyx, wzyy, wzyz, wzyw, wzzx, wzzy, wzzz, wzzw, wzwx,
                wzwy, wzwz, wzww, wwxx, wwxy, wwxz, wwxw, wwyx, wwyz,
                wwyw, wwzx, wwzy, wwzz, wwzw, wwwx, wwwy, wwwz, wwww
            ],
            [
                [0,0,0,0], [0,0,0,1], [0,0,0,2], [0,0,0,3], [0,0,1,0], [0,0,1,1], [0,0,1,2], [0,0,1,3], [0,0,2,0],
                [0,0,2,1], [0,0,2,2], [0,0,2,3], [0,0,3,0], [0,0,3,1], [0,0,3,2], [0,0,3,3], [0,1,0,0], [0,1,0,1],
                [0,1,0,2], [0,1,0,3], [0,1,1,0], [0,1,1,1], [0,1,1,2], [0,1,1,3], [0,1,2,0], [0,1,2,1], [0,1,2,2],
                [0,1,2,3], [0,1,3,0], [0,1,3,1], [0,1,3,2], [0,1,3,3], [0,2,0,0], [0,2,0,1], [0,2,0,2], [0,2,0,3],
                [0,2,1,0], [0,2,1,1], [0,2,1,2], [0,2,1,3], [0,2,2,0], [0,2,2,1], [0,2,2,2], [0,2,2,3], [0,2,3,0],
                [0,2,3,1], [0,2,3,2], [0,2,3,3], [0,3,0,0], [0,3,0,1], [0,3,0,2], [0,3,0,3], [0,3,1,0], [0,3,1,2],
                [0,3,1,3], [0,3,2,0], [0,3,2,1], [0,3,2,2], [0,3,2,3], [0,3,3,0], [0,3,3,1], [0,3,3,2], [0,3,3,3],

                [1,0,0,0], [1,0,0,1], [1,0,0,2], [1,0,0,3], [1,0,1,0], [1,0,1,1], [1,0,1,2], [1,0,1,3], [1,0,2,0],
                [1,0,2,1], [1,0,2,2], [1,0,2,3], [1,0,3,0], [1,0,3,1], [1,0,3,2], [1,0,3,3], [1,1,0,0], [1,1,0,1],
                [1,1,0,2], [1,1,0,3], [1,1,1,0], [1,1,1,1], [1,1,1,2], [1,1,1,3], [1,1,2,0], [1,1,2,1], [1,1,2,2],
                [1,1,2,3], [1,1,3,0], [1,1,3,1], [1,1,3,2], [1,1,3,3], [1,2,0,0], [1,2,0,1], [1,2,0,2], [1,2,0,3],
                [1,2,1,0], [1,2,1,1], [1,2,1,2], [1,2,1,3], [1,2,2,0], [1,2,2,1], [1,2,2,2], [1,2,2,3], [1,2,3,0],
                [1,2,3,1], [1,2,3,2], [1,2,3,3], [1,3,0,0], [1,3,0,1], [1,3,0,2], [1,3,0,3], [1,3,1,0], [1,3,1,2],
                [1,3,1,3], [1,3,2,0], [1,3,2,1], [1,3,2,2], [1,3,2,3], [1,3,3,0], [1,3,3,1], [1,3,3,2], [1,3,3,3],

                [2,0,0,0], [2,0,0,1], [2,0,0,2], [2,0,0,3], [2,0,1,0], [2,0,1,1], [2,0,1,2], [2,0,1,3], [2,0,2,0],
                [2,0,2,1], [2,0,2,2], [2,0,2,3], [2,0,3,0], [2,0,3,1], [2,0,3,2], [2,0,3,3], [2,1,0,0], [2,1,0,1],
                [2,1,0,2], [2,1,0,3], [2,1,1,0], [2,1,1,1], [2,1,1,2], [2,1,1,3], [2,1,2,0], [2,1,2,1], [2,1,2,2],
                [2,1,2,3], [2,1,3,0], [2,1,3,1], [2,1,3,2], [2,1,3,3], [2,2,0,0], [2,2,0,1], [2,2,0,2], [2,2,0,3],
                [2,2,1,0], [2,2,1,1], [2,2,1,2], [2,2,1,3], [2,2,2,0], [2,2,2,1], [2,2,2,2], [2,2,2,3], [2,2,3,0],
                [2,2,3,1], [2,2,3,2], [2,2,3,3], [2,3,0,0], [2,3,0,1], [2,3,0,2], [2,3,0,3], [2,3,1,0], [2,3,1,2],
                [2,3,1,3], [2,3,2,0], [2,3,2,1], [2,3,2,2], [2,3,2,3], [2,3,3,0], [2,3,3,1], [2,3,3,2], [2,3,3,3],

                [3,0,0,0], [3,0,0,1], [3,0,0,2], [3,0,0,3], [3,0,1,0], [3,0,1,1], [3,0,1,2], [3,0,1,3], [3,0,2,0],
                [3,0,2,1], [3,0,2,2], [3,0,2,3], [3,0,3,0], [3,0,3,1], [3,0,3,2], [3,0,3,3], [3,1,0,0], [3,1,0,1],
                [3,1,0,2], [3,1,0,3], [3,1,1,0], [3,1,1,1], [3,1,1,2], [3,1,1,3], [3,1,2,0], [3,1,2,1], [3,1,2,2],
                [3,1,2,3], [3,1,3,0], [3,1,3,1], [3,1,3,2], [3,1,3,3], [3,2,0,0], [3,2,0,1], [3,2,0,2], [3,2,0,3],
                [3,2,1,0], [3,2,1,1], [3,2,1,2], [3,2,1,3], [3,2,2,0], [3,2,2,1], [3,2,2,2], [3,2,2,3], [3,2,3,0],
                [3,2,3,1], [3,2,3,2], [3,2,3,3], [3,3,0,0], [3,3,0,1], [3,3,0,2], [3,3,0,3], [3,3,1,0], [3,3,1,2],
                [3,3,1,3], [3,3,2,0], [3,3,2,1], [3,3,2,2], [3,3,2,3], [3,3,3,0], [3,3,3,1], [3,3,3,2], [3,3,3,3]
            ]
        );
    }
}

#[doc(hidden)]
macro_rules! vector_swizzle_inner_vec {
    ( $VecN:ident, [ $($i:ident),+ ], [ $($t:tt),+ ] ) => {
        $(#[doc(hidden)] pub fn $i(&self) -> $VecN { vector_swizzle_expand_vec!($VecN, self, $t) })+
    }
}

#[doc(hidden)]
macro_rules! vector_swizzle_expand_vec {
    ( $VecN:ident, $s:ident, [ $($i:expr),+ ] ) => { $VecN::new($($s[$i],)+) }
}