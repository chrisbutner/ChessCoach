import http.server
import socketserver
import functools
import threading
import webbrowser

port = 8000
serving = False

def _serve():
  handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory="../js")
  with socketserver.TCPServer(("", port), handler) as httpd:
    httpd.serve_forever()

def ensure_serving():
  global serving
  if not serving:
    serving = True
    thread = threading.Thread(target=_serve)
    thread.start()

def launch():
  ensure_serving()
  webbrowser.open(f"http://localhost:{port}/gui.html")