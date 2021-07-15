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
import subprocess
import threading
import argparse
import platform
import shlex
import time

HOST = ""
DEFAULT_PORT = 24377
BUFFER_SIZE = 4096

def serve(uci_command, port):
  # Windows doesn't need a split for shell=False.
  if platform.system() != "Windows":
    uci_command = shlex.split(uci_command)
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, port))
    server_socket.listen(0)
    print(f"Listening on port {port}...", flush=True)
    process = None
    thread = None
    while True:
      connection, _ = server_socket.accept()
      print("Proxying...", flush=True)
      if thread:
        clean_up_process(process)
        thread.join()
      process = subprocess.Popen(uci_command, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      thread = threading.Thread(target=manage, args=(process, connection))
      thread.start()

def clean_up_process(process):
  try:
    process.stdin.close()
  except:
    pass
  try:
    process.stdout.close()
  except:
    pass
  try:
    # Try to avoid lingering socket bindings (GRPC). This is just a kill() on Windows,
    # but not doing distributed optimization there right now.
    process.terminate()
    time.sleep(10.0)
  except:
    pass
  try:
    process.kill()
  except:
    pass
  # Try to avoid lingering socket bindings (GRPC), but stay within cutechess-cli timeouts.
  time.sleep(10.0)

def clean_up_connection(connection):
  try:
    connection.shutdown(socket.SHUT_RDWR)
  except:
    pass
  try:
    connection.close()
  except:
    pass

def manage(process, connection):
  def clean_up():
    # Romeo and Juliet styles
    clean_up_process(process)
    clean_up_connection(connection)

  def forward_input():
    try:
      while True:
        data = connection.recv(BUFFER_SIZE)
        if not data:
          break
        process.stdin.write(data)
        process.stdin.flush()
    except:
      pass
    clean_up()
  manage_input = threading.Thread(target=forward_input)
  manage_input.start()

  def forward_output():
    try:
      while True:
        data = process.stdout.read1(BUFFER_SIZE)
        if not data:
          break
        connection.sendall(data)
    except:
      pass
    clean_up()
  manage_output = threading.Thread(target=forward_output)
  manage_output.start()

  manage_input.join()
  manage_output.join()
  print("Done", flush=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Act as a UCI proxy server, listening for UCI proxy clients over TCP")
  parser.add_argument("uci_command", help="UCI engine command")
  parser.add_argument("port", default=DEFAULT_PORT, type=int, nargs="?", help="port to listen on")
  args = parser.parse_args()

  serve(args.uci_command, args.port)
