from .._ffi.ndarray import TVMContext, TVMType, NDArrayBase
from .._ffi.ndarray import context, empty, from_dlpack
from .._ffi.ndarray import _set_class_ndarray
from .._ffi.ndarray import register_extension, free_extension_handle

def micro_dev(dev_id=0):
    """Construct a micro device

    Parameters
    ----------
    dev_id : int, optional
        The integer device id

    Returns
    -------
    ctx : TVMContext
        The created context

    Note
    ----
    This API is reserved for quick testing of new
    device by plugin device API as ext_dev.
    """
    return TVMContext(13, dev_id)
