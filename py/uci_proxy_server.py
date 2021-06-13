import socket
import subprocess
import threading
import argparse

HOST = ""
PORT = 24377
BUFFER_SIZE = 4096

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
  process = subprocess.Popen(uci_command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  def clean_up():
    # Romeo and Juliet styles
    try:
      process.stdin.close()
    except:
      pass
    try:
      process.stdout.close()
    except:
      pass
    try:
      process.kill()
    except:
      pass
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

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Act as a UCI proxy server, listening for UCI proxy clients over TCP")
  parser.add_argument("uci_command", help="UCI engine command")
  args = parser.parse_args()

  serve(args.uci_command)
