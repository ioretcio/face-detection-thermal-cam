from distutils.core import setup
from distutils.extension import Extension
import numpy
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension('ir_cam', ['cython/ir_cam.pyx'],
            libraries = ['irdirectsdk'],
            include_dirs=[numpy.get_include()],
            language='c++'
        )
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension('ir_cam', ['cython/ir_cam.cpp'],
            libraries = ['irdirectsdk'],
            include_dirs=[numpy.get_include()],
            language='c++'
        )
    ]

setup(
    name = "py_libirimager",
    version = "0.1",
    description = "Python wrapper for libirimager",
    url = "http://github.com/cwbollinger/py_ir_imager",
    author = "Chris Bollinger",
    author_email = "cwbollinger@gmail.com",
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    packages = ["py_libirimager"],
    license = "MIT"
)
