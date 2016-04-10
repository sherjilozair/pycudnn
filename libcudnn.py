import sys
import ctypes
import ctypes.util

if sys.platform in ('linux2', 'linux'):
    _libcudnn_libname_list = ['libcudnn.so', 'libcudnn.so.5', 'libcudnn.so.5.0.4']
elif sys.platform == 'darwin':
    _libcudnn_libname_list = ['libcudnn.dylib', 'libcudnn.5.dylib']
elif sys.platform == 'win32':
    _libcudnn_libname_list = ['cudnn64_5.dll']
else:
    raise RuntimeError('unsupported platform')

_libcudnn = None
for _libcudnn_libname in _libcudnn_libname_list:
    try:
        _libcudnn = ctypes.cdll.LoadLibrary(_libcudnn_libname)
    except OSError:
        pass
    else:
        break
if _libcudnn is None:
    raise OSError('cuDNN library not found')

# cuDNN error
_libcudnn.cudnnGetErrorString.restype = ctypes.c_char_p
_libcudnn.cudnnGetErrorString.argtypes = [ctypes.c_int]
class cudnnError(Exception):
    def __init__(self, status):
        self.status = status
    def __str__(self):
        error = _libcudnn.cudnnGetErrorString(self.status)
        return '%s' % (error)


# Helper functions

_libcudnn.cudnnGetVersion.restype = ctypes.c_size_t
_libcudnn.cudnnGetVersion.argtypes = []
def cudnnGetVersion():
    """
    Get cuDNN Version.
    """
    return _libcudnn.cudnnGetVersion()
