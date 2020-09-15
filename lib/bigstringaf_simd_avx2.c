#include <caml/mlvalues.h>
#include <caml/bigarray.h>
#include <immintrin.h>
#include <stdint.h>

#define get_ops1(a) \
  uint8_t* op_a = Caml_ba_data_val(a); \

#define get_ops2(a,b) \
  uint8_t* op_a = Caml_ba_data_val(a); \
  uint8_t* op_b = Caml_ba_data_val(b); \

#define get_ops3(a,b,c) \
  uint8_t* op_a = Caml_ba_data_val(a); \
  uint8_t* op_b = Caml_ba_data_val(b); \
  uint8_t* op_c = Caml_ba_data_val(c); \

#define get_ops4(a,b,c,d) \
  uint8_t* op_a = Caml_ba_data_val(a); \
  uint8_t* op_b = Caml_ba_data_val(b); \
  uint8_t* op_c = Caml_ba_data_val(c); \
  uint8_t* op_d = Caml_ba_data_val(d); \

value bigstringaf_simd_abs_i8(value a, intnat o1, value b, intnat o2) {
  get_ops2(a,b);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i r  = _mm256_abs_epi8(ra);
  _mm256_store_si256((__m256i*)(op_b + o2), r);

  return Val_unit;
}

value bigstringaf_simd_add_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_add_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_adds_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_adds_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_adds_u8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_adds_epu8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_and_si256(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_and_si256(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_andnot_si256(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_andnot_si256(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_avg_u8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_avg_epu8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_blend_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3, value d, intnat o4) {
  get_ops4(a,b,c,d);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i rc = _mm256_load_si256((__m256i*)(op_c + o3));
  __m256i r  = _mm256_blendv_epi8(ra, rb, rc);
  _mm256_store_si256((__m256i*)(op_d + o4), r);

  return Val_unit;
}

value bigstringaf_simd_broadcast_i8(value a, intnat o1, value b) {
  get_ops1(a);
  uint8_t constant = (uint8_t)(Int_val(b));

  __m256i r = _mm256_set1_epi8(constant);
  _mm256_store_si256((__m256i*)(op_a + o1), r);

  return Val_unit;
}

value bigstringaf_simd_cmpeq_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r = _mm256_cmpeq_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_cmpgt_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r = _mm256_cmpgt_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_max_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r = _mm256_max_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_max_u8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r = _mm256_max_epu8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_min_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r = _mm256_min_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_min_u8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r = _mm256_min_epu8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_movemask_i8(value a, intnat o1) {
  get_ops1(a);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  uint32_t r  = _mm256_movemask_epi8(ra);

  return Val_long(r);
}

value bigstringaf_simd_or_si256(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_or_si256(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_set_i8(value a, intnat o1, value b) {
  get_ops1(a);

  __m256i r = _mm256_setr_epi8(
    (uint8_t)Int_val(Field(b,0)),
    (uint8_t)Int_val(Field(b,1)),
    (uint8_t)Int_val(Field(b,2)),
    (uint8_t)Int_val(Field(b,3)),
    (uint8_t)Int_val(Field(b,4)),
    (uint8_t)Int_val(Field(b,5)),
    (uint8_t)Int_val(Field(b,6)),
    (uint8_t)Int_val(Field(b,7)),
    (uint8_t)Int_val(Field(b,8)),
    (uint8_t)Int_val(Field(b,9)),
    (uint8_t)Int_val(Field(b,10)),
    (uint8_t)Int_val(Field(b,11)),
    (uint8_t)Int_val(Field(b,12)),
    (uint8_t)Int_val(Field(b,13)),
    (uint8_t)Int_val(Field(b,14)),
    (uint8_t)Int_val(Field(b,15)),
    (uint8_t)Int_val(Field(b,16)),
    (uint8_t)Int_val(Field(b,17)),
    (uint8_t)Int_val(Field(b,18)),
    (uint8_t)Int_val(Field(b,19)),
    (uint8_t)Int_val(Field(b,20)),
    (uint8_t)Int_val(Field(b,21)),
    (uint8_t)Int_val(Field(b,22)),
    (uint8_t)Int_val(Field(b,23)),
    (uint8_t)Int_val(Field(b,24)),
    (uint8_t)Int_val(Field(b,25)),
    (uint8_t)Int_val(Field(b,26)),
    (uint8_t)Int_val(Field(b,27)),
    (uint8_t)Int_val(Field(b,28)),
    (uint8_t)Int_val(Field(b,29)),
    (uint8_t)Int_val(Field(b,30)),
    (uint8_t)Int_val(Field(b,31)));

  _mm256_store_si256((__m256i*)(op_a + o1), r);

  return Val_unit;
}

value bigstringaf_simd_shuffle_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r = _mm256_shuffle_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_sign_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r = _mm256_sign_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_sub_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_sub_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_subs_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_subs_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_subs_u8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_subs_epu8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_testc_si256(value a, intnat o1, value b, intnat o2) {
  get_ops2(a,b);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  uint32_t r  = _mm256_testc_si256(ra, rb);

  return Val_long(r);
}

value bigstringaf_simd_testnzc_si256(value a, intnat o1, value b, intnat o2) {
  get_ops2(a,b);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  uint32_t r  = _mm256_testnzc_si256(ra, rb);

  return Val_long(r);
}

value bigstringaf_simd_testz_si256(value a, intnat o1, value b, intnat o2) {
  get_ops2(a,b);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  uint32_t r  = _mm256_testz_si256(ra, rb);

  return Val_long(r);
}

value bigstringaf_simd_unpackhi_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_unpackhi_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_unpacklo_i8(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_unpacklo_epi8(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}

value bigstringaf_simd_xor_si256(value a, intnat o1, value b, intnat o2, value c, intnat o3) {
  get_ops3(a,b,c);

  __m256i ra = _mm256_load_si256((__m256i*)(op_a + o1));
  __m256i rb = _mm256_load_si256((__m256i*)(op_b + o2));
  __m256i r  = _mm256_xor_si256(ra, rb);
  _mm256_store_si256((__m256i*)(op_c + o3), r);

  return Val_unit;
}
