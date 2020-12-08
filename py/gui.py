import http.server
import socketserver
import functools
import threading
import webbrowser
import websockets
import asyncio
import json
import logging

serving = False
port = 8000
websocket_port = 8001
websocket_connected = set()
websocket_loop = None

class HttpHandler(http.server.SimpleHTTPRequestHandler):
  def log_message(self, format, *args):
    pass

def _serve():
  handler = functools.partial(HttpHandler, directory="../js")
  with socketserver.TCPServer(("localhost", port), handler) as httpd:
    httpd.serve_forever()

def _websocket_serve():
  global websocket_loop
  websocket_loop = asyncio.new_event_loop()
  asyncio.set_event_loop(websocket_loop)
  start_server = websockets.serve(_websocket_handler, "localhost", websocket_port)
  websocket_loop.run_until_complete(start_server)
  websocket_loop.run_forever()

def _ensure_serving():
  global serving
  if not serving:
    serving = True
    threading.Thread(target=_serve).start()
    threading.Thread(target=_websocket_serve).start()
    websockets_logger = logging.getLogger("websockets.server")
    websockets_logger.setLevel(logging.CRITICAL)

async def _websocket_handler(websocket, path):
  try:
    websocket_connected.add(websocket)
    while True:
      await asyncio.sleep(30)
      await websocket.send("ping")
  finally:
    websocket_connected.remove(websocket)

async def _send(websocket, message):
  try:
    return await websocket.send(message)
  except:
    pass

def _send_all(websockets, message):
  if websockets:
    asyncio.create_task(asyncio.wait([_send(websocket, message) for websocket in websockets]))

def launch():
  _ensure_serving()
  webbrowser.open(f"http://localhost:{port}/gui.html")

def show_data(data):
  message = json.dumps(data)
  websocket_loop.call_soon_threadsafe(_send_all, websocket_connected, message)