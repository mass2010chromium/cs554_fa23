from distutils.core import setup, Extension

#all_macros = [('MOTION_DEBUG', None), ('PYTHON', None)]
all_macros = [('PYTHON', None)]
vo_macros = [] #[('VO_RESTRICT', None)]
so3_macros = [
        ('SO3_STRICT', None),
        ('SO3_RESTRICT', None),
    ]

socp = Extension('socplib.socp',
                    sources = ['c/interface.c', 'c/matrix_math.c', 'c/structures/Vector.c', 'c/problem.c', 'c/cone.c'],
                    extra_compile_args = ["-O3"],
                    define_macros = all_macros + so3_macros)

setup (name = 'Cone Programming Thing',
       version = '1.0',
       description = 'Shenanigans for final project',
       ext_modules = [socp])
