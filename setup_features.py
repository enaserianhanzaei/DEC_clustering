'''from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="features_",
    ext_modules=cythonize('/home/elahe/NortfaceProject/codes/jhbjf-e551/jhbjf/temp.pyx'),
    include_dirs=[np.get_include()]
)'''

# Use this setup.py if you want to specify each extension module by hand
# To run this setup do exefile('pathToThisSetup.py')


import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = ['.'] #your include_dirs must contains the '.' for setup to search all the subfolder of the codeRootFolder
        )


extNames = scandir('utils')

extensions = [makeExtension(name) for name in extNames]

setup(
  name="DEC-Keras",
  ext_modules=extensions,
  cmdclass = {'build_ext': build_ext},
  script_args = ['build_ext'],
  options = {'build_ext':{'inplace':True, 'force':True}}
)

print('********CYTHON COMPLETE******')
