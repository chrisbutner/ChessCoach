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

import socket
import threading
import argparse
import sys

PORT = 24377
BUFFER_SIZE = 4096

def connect(host):
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
    connection.connect((host, PORT))
    manage(connection)

def manage(connection):
  def clean_up():
    # Romeo and Juliet styles
    try:
      connection.shutdown(socket.SHUT_RDWR)
    except:
      pass
    try:
      connection.close()
    except:
      pass

  def forward_input():
    try:
      while True:
        data = sys.stdin.buffer.read1(BUFFER_SIZE)
        if not data:
          break
        connection.sendall(data)
    except:
      pass
    clean_up()
  manage_input = threading.Thread(target=forward_input, daemon=True)
  manage_input.start()

  def forward_output():
    try:
      while True:
        data = connection.recv(BUFFER_SIZE)
        if not data:
          break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except:
      pass
    clean_up()
  manage_output = threading.Thread(target=forward_output)
  manage_output.start()

  # We can't interrupt "sys.stdin.buffer.read1()" portably, so just don't wait on input forwarding.
  manage_output.join()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Act as a UCI proxy client")
  parser.add_argument("host", help="UCI proxy server hostname or address")
  args = parser.parse_args()

  connect(args.host)
