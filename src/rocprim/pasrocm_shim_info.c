/* Host-only probe for the PasROCm shim.
 *
 * Prints the identity a shim carries (target, wavefront, widths mask, SDK) and
 * self-tests the pure wavefront-compatibility predicate. It makes no HIP calls,
 * so it runs on a machine without a GPU -- which is what lets CI assert that a
 * build really carries the parameters the matrix asked for, rather than only
 * that it compiled.
 *
 * Exit status is non-zero if any compatibility case disagrees with the expected
 * value, so the binary doubles as a unit test.
 */
#include <stddef.h>
#include <stdio.h>

extern const char* pasrocm_shim_target(void);
extern unsigned    pasrocm_shim_wavefront_size(void);
extern const char* pasrocm_shim_rocm_version(void);
extern unsigned    pasrocm_shim_wavefront_widths_mask(void);
extern int pasrocm_wavefront_compatible(unsigned compiled, unsigned reported,
                                         unsigned allowed_mask);
extern unsigned    pasrocm_shim_abi_version(void);
extern const char* pasrocm_shim_build_info(void);

struct compat_case {
    unsigned compiled;
    unsigned reported;
    unsigned mask;
    int expected;
};

int main(void)
{
    size_t i;
    int failures = 0;

    /* mask: 1 = wave32 only, 2 = wave64 only, 3 = both (RDNA4). */
    static const struct compat_case cases[] = {
        {32, 32, 1, 1},  /* wave32 build, wave32-only hardware              */
        {64, 64, 2, 1},  /* wave64 build, CDNA                              */
        {64, 32, 1, 0},  /* wave64 build on RDNA: wrong library             */
        {32, 64, 2, 0},  /* wave32 build on CDNA: wrong library             */
        {32, 32, 3, 1},  /* dual-width hardware, wave32 build               */
        {64, 64, 3, 1},  /* dual-width hardware, wave64 build               */
        {64, 32, 3, 1},  /* dual-width hardware reports 32, build is 64: OK */
        {64, 64, 1, 0},  /* wave64 build on wave32-only hardware: impossible  */
        {32, 32, 2, 0},  /* wave32 build on wave64-only hardware: impossible  */
        {0,  32, 3, 0},  /* nonsense compiled width                         */
        {32, 0,  3, 0},  /* nonsense reported width                         */
        {32, 32, 0, 0}   /* no supported widths                             */
    };

    printf("target=%s\n", pasrocm_shim_target());
    printf("wavefront=%u\n", pasrocm_shim_wavefront_size());
    printf("widths_mask=%u\n", pasrocm_shim_wavefront_widths_mask());
    printf("rocm=%s\n", pasrocm_shim_rocm_version());
    printf("abi=%u\n", pasrocm_shim_abi_version());
    printf("build=%s\n", pasrocm_shim_build_info());

    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const int got = pasrocm_wavefront_compatible(cases[i].compiled,
                                                     cases[i].reported,
                                                     cases[i].mask);
        printf("compat(%u,%u,%u)=%d\n", cases[i].compiled, cases[i].reported,
               cases[i].mask, got);
        if (got != cases[i].expected) {
            fprintf(stderr, "compat case %u: expected %d, got %d\n",
                    (unsigned)i, cases[i].expected, got);
            failures++;
        }
    }
    return failures == 0 ? 0 : 1;
}
