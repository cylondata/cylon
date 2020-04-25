# References
'''
https://github.com/FedericoStra/cython-package-example/blob/master/setup.py
https://github.com/thewtex/cython-cmake-example/blob/master/setup.py
'''

import os

from Cython.Build import cythonize
from setuptools import setup, Extension

# os.environ["CXX"] = "mpic++"
extra_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
extra_link_args = os.popen("mpic++ --showme:link").read().strip().split(' ')
additional_compile_args = ['-std=c++14', '-DARROW_METADATA_V4', '-DGOOGLE_GLOG_DLL_DECL="" -DNEED_EXCLUSIVE_SCAN']
# extra_compile_args#.append(additional_compile_args)
extra_compile_args = ['-std=c++14', '-DARROW_METADATA_V4 -DGOOGLE_GLOG_DLL_DECL="" -DNEED_EXCLUSIVE_SCAN']

_include_dirs = ["../cpp/src/twisterx/python",
                 "../cpp/src/twisterx/lib",
                 "../cpp/src/twisterx/",
                 "../cpp/src/twisterx/net",
                 "../cpp/src/twisterx/data",
                 "../cpp/src/twisterx/io",
                 "../cpp/src/twisterx/join",
                 "../cpp/src/twisterx/util",
                 "../cpp/build/arrow/install/lib/",
                 "../cpp/build/arrow/install/include",
                 "../cpp/build/thirdparty/glog/",
                 "../cpp/build/external/Catch/include",
                 "../cpp/thirdparty/glog/src",
                 ]

ext_modules = [
    Extension("pytwisterx.common.code",
              sources=["twisterx/common/code.pyx"],
              include_dirs=_include_dirs,
              language='c++',
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              libraries=["arrow", "twisterx", "glog"],
              library_dirs=["../cpp/build/arrow/install/lib", "../cpp/build/lib"],
              ),
    Extension("pytwisterx.common.status",
              sources=["twisterx/common/status.pyx"],
              include_dirs=_include_dirs,
              language='c++',
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              libraries=["arrow", "twisterx", "glog"],
              library_dirs=["../cpp/build/arrow/install/lib", "../cpp/build/lib"],
              ),
    Extension("pytwisterx.data",
              sources=["twisterx/data/table.pyx"],
              include_dirs=_include_dirs,
              language='c++',
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              libraries=["arrow", "twisterx", "glog"],
              library_dirs=["../cpp/build/arrow/install/lib", "../cpp/build/lib"],
              ),
    Extension("pytwisterx.net.comms.request",
              sources=["twisterx/net/txrequest.pyx",
                       "../cpp/src/twisterx/net/TxRequest.cpp",
                       "../cpp/src/twisterx/util/builtins.cpp"],
              include_dirs=_include_dirs,
              language='c++',
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              ),
    Extension("pytwisterx.net.comms.channel",
              sources=["twisterx/net/channel.pyx",
                       ],
              include_dirs=_include_dirs,
              language='c++',
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              libraries=["arrow", "twisterx", "glog"],
              library_dirs=["../cpp/build/arrow/install/lib", "../cpp/build/lib"],
              ),
    Extension("pytwisterx.net.comms.all_to_all",
              sources=["twisterx/net/all_to_all.pyx",
                       ],
              include_dirs=_include_dirs,
              language='c++',
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              libraries=["arrow", "twisterx", "glog"],
              library_dirs=["../cpp/build/arrow/install/lib", "../cpp/build/lib"],
              ),
    Extension("pytwisterx.common.join.config",
              sources=["twisterx/common/join_config.pyx",
                       ],
              include_dirs=_include_dirs,
              language='c++',
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              libraries=["arrow", "twisterx", "glog"],
              library_dirs=["../cpp/build/arrow/install/lib", "../cpp/build/lib"],
              ),
    Extension("pytwisterx.utils.join",
              sources=["twisterx/util/joinutils.py",
                       ],
              include_dirs=_include_dirs,
              language='c++',
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              libraries=["arrow", "twisterx", "glog"],
              library_dirs=["../cpp/build/arrow/install/lib", "../cpp/build/lib"],
              ),

    # Extension("pytwisterx.io.csv",
    #           sources=["twisterx/io/csv_read_config.pyx", "../cpp/src/twisterx/io/csv_read_config.cpp"],
    #           include_dirs=_include_dirs,
    #           language='c++',
    #           extra_compile_args=extra_compile_args,
    #           extra_link_args=extra_link_args,
    #           libraries=["arrow", "twisterx", "glog"],
    #           library_dirs=["../cpp/build/arrow/install/lib", "../cpp/build/lib"],
    #           ),
]

compiler_directives = {"language_level": 3, "embedsignature": True}

ext_modules = cythonize(ext_modules, compiler_directives=compiler_directives, gdb_debug=False)

# for ext in ext_modules:
#     # The Numpy C headers are currently required
#     ext.include_dirs.append(np.get_include())
#     ext.include_dirs.append(pa.get_include())
#     ext.libraries.extend(pa.get_libraries())
#     ext.library_dirs.extend(pa.get_library_dirs())
#     ext.define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))

setup(
    name="pytwisterx",
    packages=['twisterx',
              'twisterx.common',
              'twisterx.net',
              'twisterx.io',
              'twisterx.data'],
    version='0.0.1',
    setup_requires=["cython",
                    "setuptools",
                    "numpy"],
    ext_modules=ext_modules,
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'cython',
        'pyarrow'
    ],
    zip_safe=False,
)
