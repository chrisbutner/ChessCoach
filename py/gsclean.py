# ChessCoach, a neural network-based chess engine capable of natural-language commentary
# Copyright 2021 Chris Butner
#
# ChessCoach is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ChessCoach is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

import os
import subprocess
import operator
import re
import argparse
from config import Config

class Cleaner:

  def __init__(self, whatif_mode=True):
    self.whatif_mode = whatif_mode
    self.config = Config(is_cloud=True) # Always set is_cloud=True to get gs:// paths, but do all I/O via gsutil.
    self.network_root = self.config.misc["paths"]["networks"]

  # A non-zero return code from gsutil is often benign, e.g. no matches found,
  # and the same code as actual errors, so just ignore.
  def run(self, command):
    subprocess.run(command, stdin=subprocess.DEVNULL, shell=True)

  # A non-zero return code from gsutil is often benign, e.g. no matches found,
  # and the same code as actual errors, so just ignore.
  def run_read_lines(self, command):
    process = subprocess.Popen(command, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
    while True:
      line = process.stdout.readline()
      if not line:
        break
      yield line.rstrip("\n")
    process.stdout.close()

  def network_step(self, network_path):
    return int(re.search("_([0-9]+)/$", network_path).group(1))

  def command_delete_networks(self, network_name, operator, operand):
    total_count = 0
    to_remove = []
    for checkpoint in self.run_read_lines(f"gsutil ls -d \"{self.network_root}/{network_name}_*\""):
      total_count += 1
      if operator(self.network_step(checkpoint), operand):
        to_remove.append(checkpoint)
    if not to_remove:
      print("No named network checkpoints found")
      return
    elif len(to_remove) == 1:
      print(f"Deleting 1 of {total_count} named network checkpoint(s), \"{to_remove[0]}\"")
    else:
      print(f"Deleting {len(to_remove)} of {total_count} named network checkpoints, \"{to_remove[0]}\" through \"{to_remove[-1]}\"")
    if self.whatif_mode:
      print("Pass --confirm to actually run \"delete-networks\" command")
      return
    # Work around https://github.com/GoogleCloudPlatform/gsutil/issues/1215 and stay below max command line length by deleting batches.
    batch_size = 32
    for i in range(0, len(to_remove), batch_size):
      to_remove_argument = " ".join(f"\"{c}\"" for c in to_remove[i:i + batch_size])
      self.run(f"gsutil -m rm -r {to_remove_argument}")
    print("Done")

  def parse_image_tags_version(self, info):
    if len(info) == 1:
      no_tag_version = 0
      return None, no_tag_version
    assert len(info) == 2
    raw_tags = info[1]
    tags = raw_tags.split(",")
    versions = [int(re.search("([0-9]+)$", tag).group(1)) for tag in tags]
    # Return the highest tagged version, giving the image the most protection.
    return raw_tags, max(versions)

  def command_delete_images(self, repository, image_name, operator, operand):
    total_count = 0
    to_delete = []
    to_delete_display = []
    for image in self.run_read_lines(f"gcloud container images list --repository=\"{repository}\" --format=\"get(name)\""):
      if image_name != image and image_name != "*":
        continue
      for line in self.run_read_lines(f"gcloud container images list-tags {image} --format=\"get(digest, tags)\""):
        info = line.split()
        total_count += 1
        tags, version = self.parse_image_tags_version(info)
        if operator(version, operand):
          digest = info[0]
          to_delete.append(f"{image}@{digest}")
          tags_display = tags if tags is not None else "NO_TAGS"
          to_delete_display.append(f"{image}:{tags_display}")
    if total_count == 0:
      print("No named images found")
      return
    joined_display = "\n".join(to_delete_display)
    print(f"Deleting {len(to_delete_display)} of {total_count} named images:\n{joined_display}")
    if self.whatif_mode:
      print("Pass --confirm to actually run \"delete-images\" command")
      return
    batch_size = 16
    for i in range(0, len(to_delete), batch_size):
      to_delete_argument = " ".join(to_delete[i:i + batch_size])
      self.run(f"gcloud container images delete {to_delete_argument} --force-delete-tags --quiet")
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

DEFAULT_REPOSITORY = "eu.gcr.io/chesscoach"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Clean up network checkpoints in Google Cloud Storage")
  parser.add_argument("--repository", help="Google Container Registry repository", default=DEFAULT_REPOSITORY)
  parser.add_argument("--confirm", help="Actually run (inverse of what-if mode)", action="store_true")
  parser.add_argument("command", help="Top-level command to run", choices=["delete-networks", "delete-images"])
  parser.add_argument("name", help="Network or container image name; * for all image names")
  parser.add_argument("operator", help="Operator to use when choosing which networks to delete, by step number", type=parse_operator)
  parser.add_argument("operand", help="Operand to use when choosing which networks to delete, by step number", type=int)
  args = parser.parse_args()

  whatif_mode = not args.confirm

  cleaner = Cleaner(whatif_mode)

  if args.command == "delete-networks":
    cleaner.command_delete_networks(args.name, args.operator, args.operand)
  elif args.command == "delete-images":
    cleaner.command_delete_images(args.repository, args.name, args.operator, args.operand)