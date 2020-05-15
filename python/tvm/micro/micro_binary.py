from tvm.contrib import util
from . import artifact


class MicroBinary(artifact.Artifact):

  ARTIFACT_TYPE = 'micro_binary'

  @classmethod
  def from_unarchived(cls, base_dir, labelled_files, metadata):
    binary_file = labelled_files['binary_file'][0]
    del labelled_files['binary_file']

    debug_files = None
    if 'debug_files' in labelled_files:
      debug_files = labelled_files['debug_files']
      del labelled_files['debug_files']

    return cls(base_dir, binary_file, debug_files=debug_files, labelled_files=labelled_files,
               metadata=metadata)

  def __init__(self, base_dir, binary_file, debug_files=None, labelled_files=None, metadata=None):
    labelled_files = {} if labelled_files is None else dict(labelled_files)
    metadata = {} if metadata is None else dict(metadata)
    labelled_files['binary_file'] = [binary_file]
    if debug_files is not None:
      labelled_files['debug_files'] = debug_files

    super(MicroBinary, self).__init__(base_dir, labelled_files, metadata)

    self.binary_file = binary_file
    self.debug_files = debug_files


def build_micro_binary(compiler, name, libraries, options=None):
  temp = util.tempdir()
  libs = []
  for lib in libraries:
    dest = util.find_available_filename(temp_dir.relpath(os.path.basename(lib.name)))
    with open(dest, 'wb') as dest_f:
      dest_f.write(lib.elf_data)

  bin_path = temp.relpath(name)
  compiler.Binary(bin_path, libs, options)
  with open(bin_path, 'rb') as bin_f:
    elf_data = bin_f.read()

  return MicroBinary(name, elf_data, {})
