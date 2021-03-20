import os
import subprocess
import operator
import re
import argparse
from config import Config

class Cleaner:

  def __init__(self, whatif_mode=True):
    self.whatif_mode = whatif_mode
    self.config = Config(is_tpu=True) # Always set is_tpu=True to get gs:// paths, but do all I/O via gsutil.
    self.network_root = self.config.misc["paths"]["networks"]

  # A non-zero return code from gsutil is often benign, e.g. no matches found,
  # and the same code as actual errors, so just ignore.
  def run(self, command):
    subprocess.run(command, shell=True)

  # A non-zero return code from gsutil is often benign, e.g. no matches found,
  # and the same code as actual errors, so just ignore.
  def run_read_lines(self, command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
    while True:
      line = process.stdout.readline()
      if not line:
        break
      yield line.rstrip("\n")
    process.stdout.close()

  def network_step(self, network_path):
    return int(re.search("_([0-9]+)/$", network_path).group(1))

  def command_delete(self, network, operator, operand):
    total_count = 0
    to_remove = []
    for checkpoint in self.run_read_lines(f"gsutil ls -d \"{self.network_root}/{network}_*\""):
      total_count += 1
      if operator(self.network_step(checkpoint), operand):
        to_remove.append(checkpoint)
    if not to_remove:
      print("No network checkpoints matched")
      return
    elif len(to_remove) == 1:
      print(f"Deleting 1 of {total_count} network checkpoint(s), \"{to_remove[0]}\"")
    else:
      print(f"Deleting {len(to_remove)} of {total_count} network checkpoints, \"{to_remove[0]}\" through \"{to_remove[-1]}\"")
    if self.whatif_mode:
      print("Pass --confirm to actually run \"delete\" command")
      return
    # Work around https://github.com/GoogleCloudPlatform/gsutil/issues/1215 and stay below max command line length by deleting batches.
    batch_size = 32
    for i in range(0, len(to_remove), batch_size):
      to_remove_argument = " ".join(f"\"{c}\"" for c in to_remove[i:i + batch_size])
      self.run(f"gsutil -m rm -r {to_remove_argument}")
    print("Done")

def parse_operator(arg):
  if arg == "<" or arg == "lt":
    return operator.lt
  elif arg == "<=" or arg == "le":
    return operator.le
  elif arg == "==" or arg == "=" or arg == "eq":
    return operator.eq
  elif arg == "!=" or arg == "<>" or arg == "ne":
    return operator.ne
  elif arg == ">=" or arg == "ge":
    return operator.ge
  elif arg == ">" or arg == "gt":
    return operator.gt
  else:
    raise ValueError("Unrecognized operator")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Clean up network checkpoints in Google Cloud Storage")
  parser.add_argument("--confirm", help="Actually run (inverse of what-if mode)", action="store_true")
  parser.add_argument("command", help="Top-level command to run", choices=["delete"])
  parser.add_argument("network", help="Network name")
  parser.add_argument("operator", help="Operator to use when choosing which networks to delete, by step number", type=parse_operator)
  parser.add_argument("operand", help="Operand to use when choosing which networks to delete, by step number", type=int)
  args = parser.parse_args()

  whatif_mode = not args.confirm

  cleaner = Cleaner(whatif_mode)

  if args.command == "delete":
    cleaner.command_delete(args.network, args.operator, args.operand)