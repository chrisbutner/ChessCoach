import socket
import subprocess
import threading
import argparse

HOST = ""
PORT = 24377
BUFFER_SIZE = 4096

class State:
  def __init__(self):
    self.process = None
    self.connection = None

state = State()

def serve(uci_command):
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen(0)
    print(f"Listening on port {PORT}...", flush=True)
    while True:
      connection, _ = server_socket.accept()
      print("Proxying... ", end="", flush=True)
      manage(uci_command, connection)
      print("Done", flush=True)

def manage(uci_command, connection):
  state.connection = connection
  process = ensure_process(uci_command)
  while True:
    try:
      data = connection.recv(BUFFER_SIZE)
      if not data:
        raise Exception()
    except:
      clean_up(close_process=False)
      break
    try:
      process.stdin.write(data)
      process.stdin.flush()
    except:
      clean_up(close_process=True)
      break
  state.connection = None

def ensure_process(uci_command):
  if not state.process:
    process = subprocess.Popen(uci_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    def forward_output():
      while True:
        try:
          data = process.stdout.read1(BUFFER_SIZE)
          if not data:
            raise Exception()
        except:
          clean_up(close_process=True)
          return
        try:
          state.connection.sendall(data)
        except:
          # Keep forwarding input, ready for the next connection.
          clean_up(close_process=False)
    manage_output = threading.Thread(target=forward_output)
    manage_output.start()
    state.process = process
  return state.process

def clean_up(close_process):
  # When the connection has problems, close it but leave the process around so that the next connection can skip TensorFlow startup time.
  if close_process:
    clean_up_process()
  clean_up_connection()

def clean_up_process():
  if state.process is None:
    return
  try:
    state.process.stdin.close()
  except:
    pass
  try:
    state.process.stdout.close()
  except:
    pass
  try:
    state.process.kill()
  except:
    pass
  state.process = None

def clean_up_connection():
  if state.connection is None:
    return
  try:
    state.connection.shutdown(socket.SHUT_RDWR)
  except:
    pass
  try:
    state.connection.close()
  except:
    pass
  state.connection = None

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Act as a UCI proxy server, listening for UCI proxy clients over TCP")
  parser.add_argument("uci_command", help="UCI engine command")
  args = parser.parse_args()

  serve(args.uci_command)
