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
    # In contrast to the server, we want the lifetime of the client to match the lifetime of its connection.
    # However, close() on stdin hangs just like read1() and undoes the daemon workaround, so skip it and just close the connection.
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
