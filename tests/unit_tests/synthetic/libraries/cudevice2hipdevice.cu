// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hip/hip_fp8.h"
#include "cuda_fp8.h"
// CHECK-NOT: #include "hip/hip_fp8.h"
// CHECK-NOT: #include "cuda_fp8.h"

int main() {
  printf("24. CUDA Device API to HIP Device API synthetic test\n");

  double da = 0.0f;
  double dx = 0.0f;
  float fa = 0.0f;
  float fx = 0.0f;
  short int shi = 0;
  unsigned short int ushi = 0;
  double2 d2 = { 0.0f, 0.0f };
  float2 f2 = { 0.0f, 0.0f };
  __half hx = { 0.0f };
  __half hy = { 0.0f };
  __half2 h2 = { 0.0f, 0.0f };
  __half_raw hrx = { 0 };
  __half2_raw h2rx = { 0, 0 };

  // CHECK: hipRoundMode RoundMode;
  cudaRoundMode RoundMode;

  char c_1 = 0;
  unsigned char uc_1 = 0;
  short s_1 = 0;
  unsigned short us_1 = 0;
  int i_1 = 0;
  unsigned int ui_1 = 0;
  float f_1 = 0.f;
  long l_1 = 0;
  unsigned long ul_1 = 0;
  double d_1 = 0.0f;
  long long ll_1 = 0;
  unsigned long long ull_1 = 0;

  short1 sh_1 = { 0 };
  ushort1 ush_1 = { 0 };

  // CHECK: char1 ch_1_c = make_char1(c_1);
  // CHECK-NEXT: char1 ch_1_uc = make_char1(uc_1);
  // CHECK-NEXT: char1 ch_1_s = make_char1(s_1);
  // CHECK-NEXT: char1 ch_1_us = make_char1(us_1);
  // CHECK-NEXT: char1 ch_1_i = make_char1(i_1);
  // CHECK-NEXT: char1 ch_1_ui = make_char1(ui_1);
  // CHECK-NEXT: char1 ch_1_f = make_char1(f_1);
  // CHECK-NEXT: char1 ch_1_l = make_char1(l_1);
  // CHECK-NEXT: char1 ch_1_ul = make_char1(ul_1);
  // CHECK-NEXT: char1 ch_1_d = make_char1(d_1);
  // CHECK-NEXT: char1 ch_1_ll = make_char1(ll_1);
  // CHECK-NEXT: char1 ch_1_ull = make_char1(ull_1);
  char1 ch_1_c = make_char1(c_1);
  char1 ch_1_uc = make_char1(uc_1);
  char1 ch_1_s = make_char1(s_1);
  char1 ch_1_us = make_char1(us_1);
  char1 ch_1_i = make_char1(i_1);
  char1 ch_1_ui = make_char1(ui_1);
  char1 ch_1_f = make_char1(f_1);
  char1 ch_1_l = make_char1(l_1);
  char1 ch_1_ul = make_char1(ul_1);
  char1 ch_1_d = make_char1(d_1);
  char1 ch_1_ll = make_char1(ll_1);
  char1 ch_1_ull = make_char1(ull_1);

  char c_2 = 0;
  unsigned char uc_2 = 0;
  short s_2 = 0;
  unsigned short us_2 = 0;
  int i_2 = 0;
  unsigned int ui_2 = 0;
  float f_2 = 0.f;
  long l_2 = 0;
  unsigned long ul_2 = 0;
  double d_2 = 0.0f;
  long long ll_2 = 0;
  unsigned long long ull_2 = 0;

  short1 sh_2 = { 0 };
  ushort1 ush_2 = { 0 };

  // CHECK: char2 ch_2_c = make_char2(c_1, c_2);
  // CHECK-NEXT: char2 ch_2_uc = make_char2(uc_1, uc_2);
  // CHECK-NEXT: char2 ch_2_s = make_char2(s_1, s_2);
  // CHECK-NEXT: char2 ch_2_us = make_char2(us_1, us_2);
  // CHECK-NEXT: char2 ch_2_i = make_char2(i_1, i_2);
  // CHECK-NEXT: char2 ch_2_ui = make_char2(ui_1, ui_2);
  // CHECK-NEXT: char2 ch_2_f = make_char2(f_1, f_2);
  // CHECK-NEXT: char2 ch_2_l = make_char2(l_1, l_2);
  // CHECK-NEXT: char2 ch_2_ul = make_char2(ul_1, ul_2);
  // CHECK-NEXT: char2 ch_2_d = make_char2(d_1, d_2);
  // CHECK-NEXT: char2 ch_2_ll = make_char2(ll_1, ll_2);
  // CHECK-NEXT: char2 ch_2_ull = make_char2(ull_1, ull_2);
  char2 ch_2_c = make_char2(c_1, c_2);
  char2 ch_2_uc = make_char2(uc_1, uc_2);
  char2 ch_2_s = make_char2(s_1, s_2);
  char2 ch_2_us = make_char2(us_1, us_2);
  char2 ch_2_i = make_char2(i_1, i_2);
  char2 ch_2_ui = make_char2(ui_1, ui_2);
  char2 ch_2_f = make_char2(f_1, f_2);
  char2 ch_2_l = make_char2(l_1, l_2);
  char2 ch_2_ul = make_char2(ul_1, ul_2);
  char2 ch_2_d = make_char2(d_1, d_2);
  char2 ch_2_ll = make_char2(ll_1, ll_2);
  char2 ch_2_ull = make_char2(ull_1, ull_2);

  char c_3 = 0;
  unsigned char uc_3 = 0;
  short s_3 = 0;
  unsigned short us_3 = 0;
  int i_3 = 0;
  unsigned int ui_3 = 0;
  float f_3 = 0.f;
  long l_3 = 0;
  unsigned long ul_3 = 0;
  double d_3 = 0.0f;
  long long ll_3 = 0;
  unsigned long long ull_3 = 0;

  short1 sh_3 = { 0 };
  ushort1 ush_3 = { 0 };

  // CHECK: char3 ch_3_c = make_char3(c_1, c_2, c_3);
  // CHECK-NEXT: char3 ch_3_uc = make_char3(uc_1, uc_2, uc_3);
  // CHECK-NEXT: char3 ch_3_s = make_char3(s_1, s_2, s_3);
  // CHECK-NEXT: char3 ch_3_us = make_char3(us_1, us_2, us_3);
  // CHECK-NEXT: char3 ch_3_i = make_char3(i_1, i_2, i_3);
  // CHECK-NEXT: char3 ch_3_ui = make_char3(ui_1, ui_2, ui_3);
  // CHECK-NEXT: char3 ch_3_f = make_char3(f_1, f_2, f_3);
  // CHECK-NEXT: char3 ch_3_l = make_char3(l_1, l_2, l_3);
  // CHECK-NEXT: char3 ch_3_ul = make_char3(ul_1, ul_2, ul_3);
  // CHECK-NEXT: char3 ch_3_d = make_char3(d_1, d_2, d_3);
  // CHECK-NEXT: char3 ch_3_ll = make_char3(ll_1, ll_2, ll_3);
  // CHECK-NEXT: char3 ch_3_ull = make_char3(ull_1, ull_2, ull_3);
  char3 ch_3_c = make_char3(c_1, c_2, c_3);
  char3 ch_3_uc = make_char3(uc_1, uc_2, uc_3);
  char3 ch_3_s = make_char3(s_1, s_2, s_3);
  char3 ch_3_us = make_char3(us_1, us_2, us_3);
  char3 ch_3_i = make_char3(i_1, i_2, i_3);
  char3 ch_3_ui = make_char3(ui_1, ui_2, ui_3);
  char3 ch_3_f = make_char3(f_1, f_2, f_3);
  char3 ch_3_l = make_char3(l_1, l_2, l_3);
  char3 ch_3_ul = make_char3(ul_1, ul_2, ul_3);
  char3 ch_3_d = make_char3(d_1, d_2, d_3);
  char3 ch_3_ll = make_char3(ll_1, ll_2, ll_3);
  char3 ch_3_ull = make_char3(ull_1, ull_2, ull_3);

  char c_4 = 0;
  unsigned char uc_4 = 0;
  short s_4 = 0;
  unsigned short us_4 = 0;
  int i_4 = 0;
  unsigned int ui_4 = 0;
  float f_4 = 0.f;
  long l_4 = 0;
  unsigned long ul_4 = 0;
  double d_4 = 0.0f;
  long long ll_4 = 0;
  unsigned long long ull_4 = 0;

  short1 sh_4 = { 0 };
  ushort1 ush_4 = { 0 };

  // CHECK: char4 ch_4_c = make_char4(c_1, c_2, c_3, c_4);
  // CHECK-NEXT: char4 ch_4_uc = make_char4(uc_1, uc_2, uc_3, uc_4);
  // CHECK-NEXT: char4 ch_4_s = make_char4(s_1, s_2, s_3, s_4);
  // CHECK-NEXT: char4 ch_4_us = make_char4(us_1, us_2, us_3, us_4);
  // CHECK-NEXT: char4 ch_4_i = make_char4(i_1, i_2, i_3, i_4);
  // CHECK-NEXT: char4 ch_4_ui = make_char4(ui_1, ui_2, ui_3, ui_4);
  // CHECK-NEXT: char4 ch_4_f = make_char4(f_1, f_2, f_3, f_4);
  // CHECK-NEXT: char4 ch_4_l = make_char4(l_1, l_2, l_3, l_4);
  // CHECK-NEXT: char4 ch_4_ul = make_char4(ul_1, ul_2, ul_3, ul_4);
  // CHECK-NEXT: char4 ch_4_d = make_char4(d_1, d_2, d_3, d_4);
  // CHECK-NEXT: char4 ch_4_ll = make_char4(ll_1, ll_2, ll_3, ll_4);
  // CHECK-NEXT: char4 ch_4_ull = make_char4(ull_1, ull_2, ull_3, ull_4);
  char4 ch_4_c = make_char4(c_1, c_2, c_3, c_4);
  char4 ch_4_uc = make_char4(uc_1, uc_2, uc_3, uc_4);
  char4 ch_4_s = make_char4(s_1, s_2, s_3, s_4);
  char4 ch_4_us = make_char4(us_1, us_2, us_3, us_4);
  char4 ch_4_i = make_char4(i_1, i_2, i_3, i_4);
  char4 ch_4_ui = make_char4(ui_1, ui_2, ui_3, ui_4);
  char4 ch_4_f = make_char4(f_1, f_2, f_3, f_4);
  char4 ch_4_l = make_char4(l_1, l_2, l_3, l_4);
  char4 ch_4_ul = make_char4(ul_1, ul_2, ul_3, ul_4);
  char4 ch_4_d = make_char4(d_1, d_2, d_3, d_4);
  char4 ch_4_ll = make_char4(ll_1, ll_2, ll_3, ll_4);
  char4 ch_4_ull = make_char4(ull_1, ull_2, ull_3, ull_4);

  // CHECK: uchar1 uch_1_c = make_uchar1(c_1);
  // CHECK-NEXT: uchar1 uch_1_uc = make_uchar1(uc_1);
  // CHECK-NEXT: uchar1 uch_1_s = make_uchar1(s_1);
  // CHECK-NEXT: uchar1 uch_1_us = make_uchar1(us_1);
  // CHECK-NEXT: uchar1 uch_1_i = make_uchar1(i_1);
  // CHECK-NEXT: uchar1 uch_1_ui = make_uchar1(ui_1);
  // CHECK-NEXT: uchar1 uch_1_f = make_uchar1(f_1);
  // CHECK-NEXT: uchar1 uch_1_l = make_uchar1(l_1);
  // CHECK-NEXT: uchar1 uch_1_ul = make_uchar1(ul_1);
  // CHECK-NEXT: uchar1 uch_1_d = make_uchar1(d_1);
  // CHECK-NEXT: uchar1 uch_1_ll = make_uchar1(ll_1);
  // CHECK-NEXT: uchar1 uch_1_ull = make_uchar1(ull_1);
  uchar1 uch_1_c = make_uchar1(c_1);
  uchar1 uch_1_uc = make_uchar1(uc_1);
  uchar1 uch_1_s = make_uchar1(s_1);
  uchar1 uch_1_us = make_uchar1(us_1);
  uchar1 uch_1_i = make_uchar1(i_1);
  uchar1 uch_1_ui = make_uchar1(ui_1);
  uchar1 uch_1_f = make_uchar1(f_1);
  uchar1 uch_1_l = make_uchar1(l_1);
  uchar1 uch_1_ul = make_uchar1(ul_1);
  uchar1 uch_1_d = make_uchar1(d_1);
  uchar1 uch_1_ll = make_uchar1(ll_1);
  uchar1 uch_1_ull = make_uchar1(ull_1);

  // CHECK: uchar2 uch_2_c = make_uchar2(c_1, c_2);
  // CHECK-NEXT: uchar2 uch_2_uc = make_uchar2(uc_1, uc_2);
  // CHECK-NEXT: uchar2 uch_2_s = make_uchar2(s_1, s_2);
  // CHECK-NEXT: uchar2 uch_2_us = make_uchar2(us_1, us_2);
  // CHECK-NEXT: uchar2 uch_2_i = make_uchar2(i_1, i_2);
  // CHECK-NEXT: uchar2 uch_2_ui = make_uchar2(ui_1, ui_2);
  // CHECK-NEXT: uchar2 uch_2_f = make_uchar2(f_1, f_2);
  // CHECK-NEXT: uchar2 uch_2_l = make_uchar2(l_1, l_2);
  // CHECK-NEXT: uchar2 uch_2_ul = make_uchar2(ul_1, ul_2);
  // CHECK-NEXT: uchar2 uch_2_d = make_uchar2(d_1, d_2);
  // CHECK-NEXT: uchar2 uch_2_ll = make_uchar2(ll_1, ll_2);
  // CHECK-NEXT: uchar2 uch_2_ull = make_uchar2(ull_1, ull_2);
  uchar2 uch_2_c = make_uchar2(c_1, c_2);
  uchar2 uch_2_uc = make_uchar2(uc_1, uc_2);
  uchar2 uch_2_s = make_uchar2(s_1, s_2);
  uchar2 uch_2_us = make_uchar2(us_1, us_2);
  uchar2 uch_2_i = make_uchar2(i_1, i_2);
  uchar2 uch_2_ui = make_uchar2(ui_1, ui_2);
  uchar2 uch_2_f = make_uchar2(f_1, f_2);
  uchar2 uch_2_l = make_uchar2(l_1, l_2);
  uchar2 uch_2_ul = make_uchar2(ul_1, ul_2);
  uchar2 uch_2_d = make_uchar2(d_1, d_2);
  uchar2 uch_2_ll = make_uchar2(ll_1, ll_2);
  uchar2 uch_2_ull = make_uchar2(ull_1, ull_2);

  // CHECK: uchar3 uch_3_c = make_uchar3(c_1, c_2, c_3);
  // CHECK-NEXT: uchar3 uch_3_uc = make_uchar3(uc_1, uc_2, uc_3);
  // CHECK-NEXT: uchar3 uch_3_s = make_uchar3(s_1, s_2, s_3);
  // CHECK-NEXT: uchar3 uch_3_us = make_uchar3(us_1, us_2, us_3);
  // CHECK-NEXT: uchar3 uch_3_i = make_uchar3(i_1, i_2, i_3);
  // CHECK-NEXT: uchar3 uch_3_ui = make_uchar3(ui_1, ui_2, ui_3);
  // CHECK-NEXT: uchar3 uch_3_f = make_uchar3(f_1, f_2, f_3);
  // CHECK-NEXT: uchar3 uch_3_l = make_uchar3(l_1, l_2, l_3);
  // CHECK-NEXT: uchar3 uch_3_ul = make_uchar3(ul_1, ul_2, ul_3);
  // CHECK-NEXT: uchar3 uch_3_d = make_uchar3(d_1, d_2, d_3);
  // CHECK-NEXT: uchar3 uch_3_ll = make_uchar3(ll_1, ll_2, ll_3);
  // CHECK-NEXT: uchar3 uch_3_ull = make_uchar3(ull_1, ull_2, ull_3);
  uchar3 uch_3_c = make_uchar3(c_1, c_2, c_3);
  uchar3 uch_3_uc = make_uchar3(uc_1, uc_2, uc_3);
  uchar3 uch_3_s = make_uchar3(s_1, s_2, s_3);
  uchar3 uch_3_us = make_uchar3(us_1, us_2, us_3);
  uchar3 uch_3_i = make_uchar3(i_1, i_2, i_3);
  uchar3 uch_3_ui = make_uchar3(ui_1, ui_2, ui_3);
  uchar3 uch_3_f = make_uchar3(f_1, f_2, f_3);
  uchar3 uch_3_l = make_uchar3(l_1, l_2, l_3);
  uchar3 uch_3_ul = make_uchar3(ul_1, ul_2, ul_3);
  uchar3 uch_3_d = make_uchar3(d_1, d_2, d_3);
  uchar3 uch_3_ll = make_uchar3(ll_1, ll_2, ll_3);
  uchar3 uch_3_ull = make_uchar3(ull_1, ull_2, ull_3);

  // CHECK: uchar4 uch_4_c = make_uchar4(c_1, c_2, c_3, c_4);
  // CHECK-NEXT: uchar4 uch_4_uc = make_uchar4(uc_1, uc_2, uc_3, uc_4);
  // CHECK-NEXT: uchar4 uch_4_s = make_uchar4(s_1, s_2, s_3, s_4);
  // CHECK-NEXT: uchar4 uch_4_us = make_uchar4(us_1, us_2, us_3, us_4);
  // CHECK-NEXT: uchar4 uch_4_i = make_uchar4(i_1, i_2, i_3, i_4);
  // CHECK-NEXT: uchar4 uch_4_ui = make_uchar4(ui_1, ui_2, ui_3, ui_4);
  // CHECK-NEXT: uchar4 uch_4_f = make_uchar4(f_1, f_2, f_3, f_4);
  // CHECK-NEXT: uchar4 uch_4_l = make_uchar4(l_1, l_2, l_3, l_4);
  // CHECK-NEXT: uchar4 uch_4_ul = make_uchar4(ul_1, ul_2, ul_3, ul_4);
  // CHECK-NEXT: uchar4 uch_4_d = make_uchar4(d_1, d_2, d_3, d_4);
  // CHECK-NEXT: uchar4 uch_4_ll = make_uchar4(ll_1, ll_2, ll_3, ll_4);
  // CHECK-NEXT: uchar4 uch_4_ull = make_uchar4(ull_1, ull_2, ull_3, ull_4);
  uchar4 uch_4_c = make_uchar4(c_1, c_2, c_3, c_4);
  uchar4 uch_4_uc = make_uchar4(uc_1, uc_2, uc_3, uc_4);
  uchar4 uch_4_s = make_uchar4(s_1, s_2, s_3, s_4);
  uchar4 uch_4_us = make_uchar4(us_1, us_2, us_3, us_4);
  uchar4 uch_4_i = make_uchar4(i_1, i_2, i_3, i_4);
  uchar4 uch_4_ui = make_uchar4(ui_1, ui_2, ui_3, ui_4);
  uchar4 uch_4_f = make_uchar4(f_1, f_2, f_3, f_4);
  uchar4 uch_4_l = make_uchar4(l_1, l_2, l_3, l_4);
  uchar4 uch_4_ul = make_uchar4(ul_1, ul_2, ul_3, ul_4);
  uchar4 uch_4_d = make_uchar4(d_1, d_2, d_3, d_4);
  uchar4 uch_4_ll = make_uchar4(ll_1, ll_2, ll_3, ll_4);
  uchar4 uch_4_ull = make_uchar4(ull_1, ull_2, ull_3, ull_4);

  // CHECK: short1 sh_1_c = make_short1(c_1);
  // CHECK-NEXT: short1 sh_1_uc = make_short1(uc_1);
  // CHECK-NEXT: short1 sh_1_s = make_short1(s_1);
  // CHECK-NEXT: short1 sh_1_us = make_short1(us_1);
  // CHECK-NEXT: short1 sh_1_i = make_short1(i_1);
  // CHECK-NEXT: short1 sh_1_ui = make_short1(ui_1);
  // CHECK-NEXT: short1 sh_1_f = make_short1(f_1);
  // CHECK-NEXT: short1 sh_1_l = make_short1(l_1);
  // CHECK-NEXT: short1 sh_1_ul = make_short1(ul_1);
  // CHECK-NEXT: short1 sh_1_d = make_short1(d_1);
  // CHECK-NEXT: short1 sh_1_ll = make_short1(ll_1);
  // CHECK-NEXT: short1 sh_1_ull = make_short1(ull_1);
  short1 sh_1_c = make_short1(c_1);
  short1 sh_1_uc = make_short1(uc_1);
  short1 sh_1_s = make_short1(s_1);
  short1 sh_1_us = make_short1(us_1);
  short1 sh_1_i = make_short1(i_1);
  short1 sh_1_ui = make_short1(ui_1);
  short1 sh_1_f = make_short1(f_1);
  short1 sh_1_l = make_short1(l_1);
  short1 sh_1_ul = make_short1(ul_1);
  short1 sh_1_d = make_short1(d_1);
  short1 sh_1_ll = make_short1(ll_1);
  short1 sh_1_ull = make_short1(ull_1);

  // CHECK: short2 sh_2_c = make_short2(c_1, c_2);
  // CHECK-NEXT: short2 sh_2_uc = make_short2(uc_1, uc_2);
  // CHECK-NEXT: short2 sh_2_s = make_short2(s_1, s_2);
  // CHECK-NEXT: short2 sh_2_us = make_short2(us_1, us_2);
  // CHECK-NEXT: short2 sh_2_i = make_short2(i_1, i_2);
  // CHECK-NEXT: short2 sh_2_ui = make_short2(ui_1, ui_2);
  // CHECK-NEXT: short2 sh_2_f = make_short2(f_1, f_2);
  // CHECK-NEXT: short2 sh_2_l = make_short2(l_1, l_2);
  // CHECK-NEXT: short2 sh_2_ul = make_short2(ul_1, ul_2);
  // CHECK-NEXT: short2 sh_2_d = make_short2(d_1, d_2);
  // CHECK-NEXT: short2 sh_2_ll = make_short2(ll_1, ll_2);
  // CHECK-NEXT: short2 sh_2_ull = make_short2(ull_1, ull_2);
  short2 sh_2_c = make_short2(c_1, c_2);
  short2 sh_2_uc = make_short2(uc_1, uc_2);
  short2 sh_2_s = make_short2(s_1, s_2);
  short2 sh_2_us = make_short2(us_1, us_2);
  short2 sh_2_i = make_short2(i_1, i_2);
  short2 sh_2_ui = make_short2(ui_1, ui_2);
  short2 sh_2_f = make_short2(f_1, f_2);
  short2 sh_2_l = make_short2(l_1, l_2);
  short2 sh_2_ul = make_short2(ul_1, ul_2);
  short2 sh_2_d = make_short2(d_1, d_2);
  short2 sh_2_ll = make_short2(ll_1, ll_2);
  short2 sh_2_ull = make_short2(ull_1, ull_2);

  // CHECK: short3 sh_3_c = make_short3(c_1, c_2, c_3);
  // CHECK-NEXT: short3 sh_3_uc = make_short3(uc_1, uc_2, uc_3);
  // CHECK-NEXT: short3 sh_3_s = make_short3(s_1, s_2, s_3);
  // CHECK-NEXT: short3 sh_3_us = make_short3(us_1, us_2, us_3);
  // CHECK-NEXT: short3 sh_3_i = make_short3(i_1, i_2, i_3);
  // CHECK-NEXT: short3 sh_3_ui = make_short3(ui_1, ui_2, ui_3);
  // CHECK-NEXT: short3 sh_3_f = make_short3(f_1, f_2, f_3);
  // CHECK-NEXT: short3 sh_3_l = make_short3(l_1, l_2, l_3);
  // CHECK-NEXT: short3 sh_3_ul = make_short3(ul_1, ul_2, ul_3);
  // CHECK-NEXT: short3 sh_3_d = make_short3(d_1, d_2, d_3);
  // CHECK-NEXT: short3 sh_3_ll = make_short3(ll_1, ll_2, ll_3);
  // CHECK-NEXT: short3 sh_3_ull = make_short3(ull_1, ull_2, ull_3);
  short3 sh_3_c = make_short3(c_1, c_2, c_3);
  short3 sh_3_uc = make_short3(uc_1, uc_2, uc_3);
  short3 sh_3_s = make_short3(s_1, s_2, s_3);
  short3 sh_3_us = make_short3(us_1, us_2, us_3);
  short3 sh_3_i = make_short3(i_1, i_2, i_3);
  short3 sh_3_ui = make_short3(ui_1, ui_2, ui_3);
  short3 sh_3_f = make_short3(f_1, f_2, f_3);
  short3 sh_3_l = make_short3(l_1, l_2, l_3);
  short3 sh_3_ul = make_short3(ul_1, ul_2, ul_3);
  short3 sh_3_d = make_short3(d_1, d_2, d_3);
  short3 sh_3_ll = make_short3(ll_1, ll_2, ll_3);
  short3 sh_3_ull = make_short3(ull_1, ull_2, ull_3);

  // CHECK: short4 sh_4_c = make_short4(c_1, c_2, c_3, c_4);
  // CHECK-NEXT: short4 sh_4_uc = make_short4(uc_1, uc_2, uc_3, uc_4);
  // CHECK-NEXT: short4 sh_4_s = make_short4(s_1, s_2, s_3, s_4);
  // CHECK-NEXT: short4 sh_4_us = make_short4(us_1, us_2, us_3, us_4);
  // CHECK-NEXT: short4 sh_4_i = make_short4(i_1, i_2, i_3, i_4);
  // CHECK-NEXT: short4 sh_4_ui = make_short4(ui_1, ui_2, ui_3, ui_4);
  // CHECK-NEXT: short4 sh_4_f = make_short4(f_1, f_2, f_3, f_4);
  // CHECK-NEXT: short4 sh_4_l = make_short4(l_1, l_2, l_3, l_4);
  // CHECK-NEXT: short4 sh_4_ul = make_short4(ul_1, ul_2, ul_3, ul_4);
  // CHECK-NEXT: short4 sh_4_d = make_short4(d_1, d_2, d_3, d_4);
  // CHECK-NEXT: short4 sh_4_ll = make_short4(ll_1, ll_2, ll_3, ll_4);
  // CHECK-NEXT: short4 sh_4_ull = make_short4(ull_1, ull_2, ull_3, ull_4);
  short4 sh_4_c = make_short4(c_1, c_2, c_3, c_4);
  short4 sh_4_uc = make_short4(uc_1, uc_2, uc_3, uc_4);
  short4 sh_4_s = make_short4(s_1, s_2, s_3, s_4);
  short4 sh_4_us = make_short4(us_1, us_2, us_3, us_4);
  short4 sh_4_i = make_short4(i_1, i_2, i_3, i_4);
  short4 sh_4_ui = make_short4(ui_1, ui_2, ui_3, ui_4);
  short4 sh_4_f = make_short4(f_1, f_2, f_3, f_4);
  short4 sh_4_l = make_short4(l_1, l_2, l_3, l_4);
  short4 sh_4_ul = make_short4(ul_1, ul_2, ul_3, ul_4);
  short4 sh_4_d = make_short4(d_1, d_2, d_3, d_4);
  short4 sh_4_ll = make_short4(ll_1, ll_2, ll_3, ll_4);
  short4 sh_4_ull = make_short4(ull_1, ull_2, ull_3, ull_4);

  // CHECK: ushort1 ush_1_c = make_ushort1(c_1);
  // CHECK-NEXT: ushort1 ush_1_uc = make_ushort1(uc_1);
  // CHECK-NEXT: ushort1 ush_1_s = make_ushort1(s_1);
  // CHECK-NEXT: ushort1 ush_1_us = make_ushort1(us_1);
  // CHECK-NEXT: ushort1 ush_1_i = make_ushort1(i_1);
  // CHECK-NEXT: ushort1 ush_1_ui = make_ushort1(ui_1);
  // CHECK-NEXT: ushort1 ush_1_f = make_ushort1(f_1);
  // CHECK-NEXT: ushort1 ush_1_l = make_ushort1(l_1);
  // CHECK-NEXT: ushort1 ush_1_ul = make_ushort1(ul_1);
  // CHECK-NEXT: ushort1 ush_1_d = make_ushort1(d_1);
  // CHECK-NEXT: ushort1 ush_1_ll = make_ushort1(ll_1);
  // CHECK-NEXT: ushort1 ush_1_ull = make_ushort1(ull_1);
  ushort1 ush_1_c = make_ushort1(c_1);
  ushort1 ush_1_uc = make_ushort1(uc_1);
  ushort1 ush_1_s = make_ushort1(s_1);
  ushort1 ush_1_us = make_ushort1(us_1);
  ushort1 ush_1_i = make_ushort1(i_1);
  ushort1 ush_1_ui = make_ushort1(ui_1);
  ushort1 ush_1_f = make_ushort1(f_1);
  ushort1 ush_1_l = make_ushort1(l_1);
  ushort1 ush_1_ul = make_ushort1(ul_1);
  ushort1 ush_1_d = make_ushort1(d_1);
  ushort1 ush_1_ll = make_ushort1(ll_1);
  ushort1 ush_1_ull = make_ushort1(ull_1);

  // CHECK: ushort2 ush_2_c = make_ushort2(c_1, c_2);
  // CHECK-NEXT: ushort2 ush_2_uc = make_ushort2(uc_1, uc_2);
  // CHECK-NEXT: ushort2 ush_2_s = make_ushort2(s_1, s_2);
  // CHECK-NEXT: ushort2 ush_2_us = make_ushort2(us_1, us_2);
  // CHECK-NEXT: ushort2 ush_2_i = make_ushort2(i_1, i_2);
  // CHECK-NEXT: ushort2 ush_2_ui = make_ushort2(ui_1, ui_2);
  // CHECK-NEXT: ushort2 ush_2_f = make_ushort2(f_1, f_2);
  // CHECK-NEXT: ushort2 ush_2_l = make_ushort2(l_1, l_2);
  // CHECK-NEXT: ushort2 ush_2_ul = make_ushort2(ul_1, ul_2);
  // CHECK-NEXT: ushort2 ush_2_d = make_ushort2(d_1, d_2);
  // CHECK-NEXT: ushort2 ush_2_ll = make_ushort2(ll_1, ll_2);
  // CHECK-NEXT: ushort2 ush_2_ull = make_ushort2(ull_1, ull_2);
  ushort2 ush_2_c = make_ushort2(c_1, c_2);
  ushort2 ush_2_uc = make_ushort2(uc_1, uc_2);
  ushort2 ush_2_s = make_ushort2(s_1, s_2);
  ushort2 ush_2_us = make_ushort2(us_1, us_2);
  ushort2 ush_2_i = make_ushort2(i_1, i_2);
  ushort2 ush_2_ui = make_ushort2(ui_1, ui_2);
  ushort2 ush_2_f = make_ushort2(f_1, f_2);
  ushort2 ush_2_l = make_ushort2(l_1, l_2);
  ushort2 ush_2_ul = make_ushort2(ul_1, ul_2);
  ushort2 ush_2_d = make_ushort2(d_1, d_2);
  ushort2 ush_2_ll = make_ushort2(ll_1, ll_2);
  ushort2 ush_2_ull = make_ushort2(ull_1, ull_2);

  // CHECK: ushort3 ush_3_c = make_ushort3(c_1, c_2, c_3);
  // CHECK-NEXT: ushort3 ush_3_uc = make_ushort3(uc_1, uc_2, uc_3);
  // CHECK-NEXT: ushort3 ush_3_s = make_ushort3(s_1, s_2, s_3);
  // CHECK-NEXT: ushort3 ush_3_us = make_ushort3(us_1, us_2, us_3);
  // CHECK-NEXT: ushort3 ush_3_i = make_ushort3(i_1, i_2, i_3);
  // CHECK-NEXT: ushort3 ush_3_ui = make_ushort3(ui_1, ui_2, ui_3);
  // CHECK-NEXT: ushort3 ush_3_f = make_ushort3(f_1, f_2, f_3);
  // CHECK-NEXT: ushort3 ush_3_l = make_ushort3(l_1, l_2, l_3);
  // CHECK-NEXT: ushort3 ush_3_ul = make_ushort3(ul_1, ul_2, ul_3);
  // CHECK-NEXT: ushort3 ush_3_d = make_ushort3(d_1, d_2, d_3);
  // CHECK-NEXT: ushort3 ush_3_ll = make_ushort3(ll_1, ll_2, ll_3);
  // CHECK-NEXT: ushort3 ush_3_ull = make_ushort3(ull_1, ull_2, ull_3);
  ushort3 ush_3_c = make_ushort3(c_1, c_2, c_3);
  ushort3 ush_3_uc = make_ushort3(uc_1, uc_2, uc_3);
  ushort3 ush_3_s = make_ushort3(s_1, s_2, s_3);
  ushort3 ush_3_us = make_ushort3(us_1, us_2, us_3);
  ushort3 ush_3_i = make_ushort3(i_1, i_2, i_3);
  ushort3 ush_3_ui = make_ushort3(ui_1, ui_2, ui_3);
  ushort3 ush_3_f = make_ushort3(f_1, f_2, f_3);
  ushort3 ush_3_l = make_ushort3(l_1, l_2, l_3);
  ushort3 ush_3_ul = make_ushort3(ul_1, ul_2, ul_3);
  ushort3 ush_3_d = make_ushort3(d_1, d_2, d_3);
  ushort3 ush_3_ll = make_ushort3(ll_1, ll_2, ll_3);
  ushort3 ush_3_ull = make_ushort3(ull_1, ull_2, ull_3);

  // CHECK: ushort4 ush_4_c = make_ushort4(c_1, c_2, c_3, c_4);
  // CHECK-NEXT: ushort4 ush_4_uc = make_ushort4(uc_1, uc_2, uc_3, uc_4);
  // CHECK-NEXT: ushort4 ush_4_s = make_ushort4(s_1, s_2, s_3, s_4);
  // CHECK-NEXT: ushort4 ush_4_us = make_ushort4(us_1, us_2, us_3, us_4);
  // CHECK-NEXT: ushort4 ush_4_i = make_ushort4(i_1, i_2, i_3, i_4);
  // CHECK-NEXT: ushort4 ush_4_ui = make_ushort4(ui_1, ui_2, ui_3, ui_4);
  // CHECK-NEXT: ushort4 ush_4_f = make_ushort4(f_1, f_2, f_3, f_4);
  // CHECK-NEXT: ushort4 ush_4_l = make_ushort4(l_1, l_2, l_3, l_4);
  // CHECK-NEXT: ushort4 ush_4_ul = make_ushort4(ul_1, ul_2, ul_3, ul_4);
  // CHECK-NEXT: ushort4 ush_4_d = make_ushort4(d_1, d_2, d_3, d_4);
  // CHECK-NEXT: ushort4 ush_4_ll = make_ushort4(ll_1, ll_2, ll_3, ll_4);
  // CHECK-NEXT: ushort4 ush_4_ull = make_ushort4(ull_1, ull_2, ull_3, ull_4);
  ushort4 ush_4_c = make_ushort4(c_1, c_2, c_3, c_4);
  ushort4 ush_4_uc = make_ushort4(uc_1, uc_2, uc_3, uc_4);
  ushort4 ush_4_s = make_ushort4(s_1, s_2, s_3, s_4);
  ushort4 ush_4_us = make_ushort4(us_1, us_2, us_3, us_4);
  ushort4 ush_4_i = make_ushort4(i_1, i_2, i_3, i_4);
  ushort4 ush_4_ui = make_ushort4(ui_1, ui_2, ui_3, ui_4);
  ushort4 ush_4_f = make_ushort4(f_1, f_2, f_3, f_4);
  ushort4 ush_4_l = make_ushort4(l_1, l_2, l_3, l_4);
  ushort4 ush_4_ul = make_ushort4(ul_1, ul_2, ul_3, ul_4);
  ushort4 ush_4_d = make_ushort4(d_1, d_2, d_3, d_4);
  ushort4 ush_4_ll = make_ushort4(ll_1, ll_2, ll_3, ll_4);
  ushort4 ush_4_ull = make_ushort4(ull_1, ull_2, ull_3, ull_4);

#if CUDA_VERSION >= 11000
  // CHECK: __hip_bfloat16 _bf16 = { 0.0f };
  // CHECK-NEXT: __hip_bfloat16 bf16a = { 0.0f };
  // CHECK-NEXT: __hip_bfloat16 bf16b = { 0.0f };
  __nv_bfloat16 _bf16 = { 0.0f };
  __nv_bfloat16 bf16a = { 0.0f };
  __nv_bfloat16 bf16b = { 0.0f };

  // CHECK: hip_bfloat16 bf16 = { 0 };
  nv_bfloat16 bf16 = { 0 };

  // CHECK: __hip_bfloat16_raw bf16r = { 0 };
  __nv_bfloat16_raw bf16r = { 0 };

  // CHECK: __hip_bfloat162 bf162 = { 0, 0 };
  // CHECK-NEXT: __hip_bfloat162 bf162a = { 0, 0 };
  // CHECK-NEXT: __hip_bfloat162 bf162b = { 0, 0 };
  __nv_bfloat162 bf162 = { 0, 0 };
  __nv_bfloat162 bf162a = { 0, 0 };
  __nv_bfloat162 bf162b = { 0, 0 };

  // CHECK: __hip_bfloat162_raw bf162r = { 0, 0 };
  __nv_bfloat162_raw bf162r = { 0, 0 };

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __double2bfloat16(const double a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __double2bfloat16(const double a)
  // CHECK: _bf16 = __double2bfloat16(da);
  _bf16 = __double2bfloat16(da);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16(const float a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __float2bfloat16(float f);
  // CHECK: _bf16 = __float2bfloat16(fa);
  _bf16 = __float2bfloat16(fa);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ float __bfloat162float(const __nv_bfloat16 a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ float __bfloat162float(__hip_bfloat16 a);
  // CHECK: _bf16 = __bfloat162float(fa);
  _bf16 = __bfloat162float(fa);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ float2 __bfloat1622float2(const __nv_bfloat162 a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ float2 __bfloat1622float2(const __hip_bfloat162 a);
  // CHECK: f2 = __bfloat1622float2(bf162);
  f2 = __bfloat1622float2(bf162);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __float22bfloat162_rn(const float2 a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __float22bfloat162_rn(const float2 a);
  // CHECK: bf162 = __float22bfloat162_rn(f2);
  bf162 = __float22bfloat162_rn(f2);
#endif

#if CUDA_VERSION >= 11080
  // CHECK: __hip_fp8_storage_t fp8_storage_t;
  __nv_fp8_storage_t fp8_storage_t;

  // CHECK: __hip_fp8x2_storage_t fp8x2_storage_t;
  __nv_fp8x2_storage_t fp8x2_storage_t;

  // CHECK: __hip_fp8x4_storage_t fp8x4_storage_t;
  __nv_fp8x4_storage_t fp8x4_storage_t;

  // CHECK: __hip_fp8_e5m2_fnuz fp8_e5m2;
  __nv_fp8_e5m2 fp8_e5m2;

  // CHECK: __hip_fp8x2_e5m2_fnuz fp8x2_e5m2;
  __nv_fp8x2_e5m2 fp8x2_e5m2;

  // CHECK: __hip_fp8_e4m3_fnuz fp8_e4m3;
  __nv_fp8_e4m3 fp8_e4m3;

  // CHECK: __hip_fp8x2_e4m3_fnuz fp8x2_e4m3;
  __nv_fp8x2_e4m3 fp8x2_e4m3;

  // CHECK: __hip_fp8x4_e4m3_fnuz fp8x4_e4m3;
  __nv_fp8x4_e4m3 fp8x4_e4m3;

  // CHECK: __hip_saturation_t saturation_t;
  // CHECK-NEXT: __hip_saturation_t NOSAT = __HIP_NOSAT;
  // CHECK-NEXT: __hip_saturation_t SATFINITE = __HIP_SATFINITE;
  __nv_saturation_t saturation_t;
  __nv_saturation_t NOSAT = __NV_NOSAT;
  __nv_saturation_t SATFINITE = __NV_SATFINITE;

  // CHECK: __hip_fp8_interpretation_t fp8_interpretation_t;
  // CHECK-NEXT: __hip_fp8_interpretation_t E4M3 = __HIP_E4M3_FNUZ;
  // CHECK-NEXT: __hip_fp8_interpretation_t E5M2 = __HIP_E5M2_FNUZ;
  __nv_fp8_interpretation_t fp8_interpretation_t;
  __nv_fp8_interpretation_t E4M3 = __NV_E4M3;
  __nv_fp8_interpretation_t E5M2 = __NV_E5M2;

  // CHECK: __hip_fp8x4_e5m2_fnuz fp8x4_e5m2;
  __nv_fp8x4_e5m2 fp8x4_e5m2;

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t __nv_cvt_double_to_fp8(const double x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_double_to_fp8(const double d, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8_storage_t = __hip_cvt_double_to_fp8(dx, saturation_t, fp8_interpretation_t);
  fp8_storage_t = __nv_cvt_double_to_fp8(dx, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t __nv_cvt_double2_to_fp8x2(const double2 x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_double2_to_fp8x2(const double2 d2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8x2_storage_t = __hip_cvt_double2_to_fp8x2(d2, saturation_t, fp8_interpretation_t);
  fp8x2_storage_t = __nv_cvt_double2_to_fp8x2(d2, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t __nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_float_to_fp8(const float f, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8_storage_t = __hip_cvt_float_to_fp8(fx, saturation_t, fp8_interpretation_t);
  fp8_storage_t = __nv_cvt_float_to_fp8(fx, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(const float2 x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_float2_to_fp8x2(const float2 f2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8x2_storage_t = __hip_cvt_float2_to_fp8x2(f2, saturation_t, fp8_interpretation_t);
  fp8x2_storage_t = __nv_cvt_float2_to_fp8x2(f2, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(const __half_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_halfraw_to_fp8(const __half_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8_storage_t = __hip_cvt_halfraw_to_fp8(hrx, saturation_t, fp8_interpretation_t);
  fp8_storage_t = __nv_cvt_halfraw_to_fp8(hrx, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(const __half2_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_halfraw2_to_fp8x2(const __half2_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8x2_storage_t = __hip_cvt_halfraw2_to_fp8x2(h2rx, saturation_t, fp8_interpretation_t);
  fp8x2_storage_t = __nv_cvt_halfraw2_to_fp8x2(h2rx, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(const __nv_bfloat16_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_bfloat16raw_to_fp8(const __hip_bfloat16_raw hr, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8_storage_t = __hip_cvt_bfloat16raw_to_fp8(bf16r, saturation_t, fp8_interpretation_t);
  fp8_storage_t = __nv_cvt_bfloat16raw_to_fp8(bf16r, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(const __nv_bfloat162_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_bfloat16raw2_to_fp8x2(const __hip_bfloat162_raw hr, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8x2_storage_t = __hip_cvt_bfloat16raw2_to_fp8x2(bf162r, saturation_t, fp8_interpretation_t);
  fp8x2_storage_t = __nv_cvt_bfloat16raw2_to_fp8x2(bf162r, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __half_raw __nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __half_raw __hip_cvt_fp8_to_halfraw(const __hip_fp8_storage_t x, const __hip_fp8_interpretation_t type);
  // CHECK: hrx = __hip_cvt_fp8_to_halfraw(fp8_storage_t, fp8_interpretation_t);
  hrx = __nv_cvt_fp8_to_halfraw(fp8_storage_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __half2_raw __nv_cvt_fp8x2_to_halfraw2(const __nv_fp8x2_storage_t x, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp8x2_to_halfraw2(const __hip_fp8x2_storage_t x, const __hip_fp8_interpretation_t type);
  // CHECK: h2rx = __hip_cvt_fp8x2_to_halfraw2(fp8x2_storage_t, fp8_interpretation_t);
  h2rx = __nv_cvt_fp8x2_to_halfraw2(fp8x2_storage_t, fp8_interpretation_t);
#endif

#if CUDA_VERSION >= 12020
  // CUDA: __CUDA_HOSTDEVICE_FP16_DECL__ __half2 make_half2(const __half x, const __half y);
  // HIP: __HOST_DEVICE__ __half2 make_half2(__half x, __half y);
  // CHECK: h2 = make_half2(hx, hy);
  h2 = make_half2(hx, hy);

  // CHECK: __half _INF_FP16 = HIPRT_INF_FP16;
  // CHECK-NEXT: __half _MAX_NORMAL_FP16 = HIPRT_MAX_NORMAL_FP16;
  // CHECK-NEXT: __half _MIN_DENORM_FP16 = HIPRT_MIN_DENORM_FP16;
  // CHECK-NEXT: __half _NAN_FP16 = HIPRT_NAN_FP16;
  // CHECK-NEXT: __half _NEG_ZERO_FP16 = HIPRT_NEG_ZERO_FP16;
  // CHECK-NEXT: __half _ONE_FP16 = HIPRT_ONE_FP16;
  // CHECK-NEXT: __half _ZERO_FP16 = HIPRT_ZERO_FP16;
  __half _INF_FP16 = CUDART_INF_FP16;
  __half _MAX_NORMAL_FP16 = CUDART_MAX_NORMAL_FP16;
  __half _MIN_DENORM_FP16 = CUDART_MIN_DENORM_FP16;
  __half _NAN_FP16 = CUDART_NAN_FP16;
  __half _NEG_ZERO_FP16 = CUDART_NEG_ZERO_FP16;
  __half _ONE_FP16 = CUDART_ONE_FP16;
  __half _ZERO_FP16 = CUDART_ZERO_FP16;
#endif

  return 0;
}
