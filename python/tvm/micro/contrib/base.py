import glob
import os
import shutil


GLOB_PATTERNS = ['__tvm_*', 'libtvm__*']

def populate_tvm_objs(dest_dir, objs):
    """Replace tvm-prefixed files in a build worktree.

    This function is intended to be used to place TVM source files and libraries into a
    template on-device runtime project.

    Parameters
    ----------
    dest_dir : str
        Path to the destination directory.

    objs : List[MicroLibrary]
        List of MicroLibrary to place in the project directory.

    Returns
    -------
    List[str] :
        List of paths, each relative to  `dest_dir` to the newly-copied MicroLibrary files.
    """
    copied = []
    for p in GLOB_PATTERNS:
        for f in glob.glob(os.path.join(dest_dir, p)):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.unlink(f)

    for obj in objs:
        for lib_file in obj.library_files:
            obj_base = os.path.basename(lib_file)
            if obj_base.endswith('.a'):
                dest_basename = f'libtvm__{obj_base}'
            else:
                dest_basename = f'__tvm_{obj_base}'

            copied.append(dest_basename)
            dest = os.path.join(dest_dir, dest_basename)
            shutil.copy(obj.abspath(lib_file), dest)

    return copied
