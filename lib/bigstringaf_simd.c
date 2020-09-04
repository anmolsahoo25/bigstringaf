#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/fail.h>
#include <caml/custom.h>
#include <caml/bigarray.h>

#include <stdlib.h>
#include <stdint.h>

void bigstringaf_simd_finalize(value v);

static struct custom_operations bigstringaf_simd_ops = {
  "bigstringaf_simd_ops",
  bigstringaf_simd_finalize,
  custom_compare_default,
  custom_hash_default,
  custom_serialize_default,
  custom_deserialize_default,
  custom_compare_ext_default,
  custom_fixed_length_default
};

int bigstringaf_simd_element_size[] =
{ 4 /*FLOAT32*/, 8 /*FLOAT64*/,
  1 /*SINT8*/, 1 /*UINT8*/,
  2 /*SINT16*/, 2 /*UINT16*/,
  4 /*INT32*/, 8 /*INT64*/,
  sizeof(value) /*CAML_INT*/, sizeof(value) /*NATIVE_INT*/,
  8 /*COMPLEX32*/, 16 /*COMPLEX64*/,
  1 /*CHAR*/
};

value bigstringaf_simd_alloc(int flags, int num_dims, void* data, intnat* dim) {
  uintnat num_elts, asize, size;
  int i;
  value res;
  struct caml_ba_array * b;
  intnat dimcopy[CAML_BA_MAX_NUM_DIMS];

  CAMLassert(num_dims >= 0 && num_dims <= CAML_BA_MAX_NUM_DIMS);
  CAMLassert((flags & CAML_BA_KIND_MASK) <= CAML_BA_CHAR);
  for (i = 0; i < num_dims; i++) dimcopy[i] = dim[i];
  size = 0;
  if (data == NULL) {
    num_elts = 1;
    for (i = 0; i < num_dims; i++) {
      if (caml_umul_overflow(num_elts, dimcopy[i], &num_elts))
        caml_raise_out_of_memory();
    }
    if (caml_umul_overflow(num_elts,
                           bigstringaf_simd_element_size[flags & CAML_BA_KIND_MASK],
                           &size))
      caml_raise_out_of_memory();
    int ret = posix_memalign(&data, 32, size);
    if (((data == NULL) && (size != 0)) || ret == 12) caml_raise_out_of_memory();
    flags |= CAML_BA_MANAGED;
  }
  asize = SIZEOF_BA_ARRAY + num_dims * sizeof(intnat);
  res = caml_alloc_custom_mem(&bigstringaf_simd_ops, asize, size);
  b = Caml_ba_array_val(res);
  b->data = data;
  b->num_dims = num_dims;
  b->flags = flags;
  b->proxy = NULL;
  for (i = 0; i < num_dims; i++) b->dim[i] = dimcopy[i];
  return res;
}

value bigstringaf_simd_create(value vkind, value vlayout, value vdim)
{
  intnat dim[CAML_BA_MAX_NUM_DIMS];
  mlsize_t num_dims;
  unsigned int i;
  int flags;

  num_dims = Wosize_val(vdim);
  /* here num_dims is unsigned (mlsize_t) so no need to check (num_dims >= 0) */
  if (num_dims > CAML_BA_MAX_NUM_DIMS)
    caml_invalid_argument("Bigarray.create: bad number of dimensions");
  for (i = 0; i < num_dims; i++) {
    dim[i] = Long_val(Field(vdim, i));
    if (dim[i] < 0)
      caml_invalid_argument("Bigarray.create: negative dimension");
  }
  flags = Caml_ba_kind_val(vkind) | Caml_ba_layout_val(vlayout);
  return bigstringaf_simd_alloc(flags, num_dims, NULL, dim);
}

void bigstringaf_simd_finalize(value v)
{
  struct caml_ba_array * b = Caml_ba_array_val(v);

  switch (b->flags & CAML_BA_MANAGED_MASK) {
  case CAML_BA_EXTERNAL:
    break;
  case CAML_BA_MANAGED:
    if (b->proxy == NULL) {
      free(b->data);
    } else {
      if (-- b->proxy->refcount == 0) {
        free(b->proxy->data);
        free(b->proxy);
      }
    }
    break;
  case CAML_BA_MAPPED_FILE:
    /* Bigarrays for mapped files use a different finalization method */
    /* fallthrough */
  default:
    CAMLassert(0);
  }
}
