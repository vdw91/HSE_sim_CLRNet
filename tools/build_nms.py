from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name='nms_impl',
    ext_modules=[
        CUDAExtension(
            name='clrnet.ops.nms_impl',
            sources=['clrnet/ops/csrc/nms.cpp'],  # Add your .cu if any
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    # Add this line for verbose output:
    options={'build_ext': {'verbose': True}},
)