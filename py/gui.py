import http.server
import socketserver
import functools
import threading
import webbrowser
import websockets
import asyncio
import json
import logging
import os
import network
import tensorflow as tf
try:
  import chesscoach
except:
  pass

# ----- WebSockets -----

serving = False
port = 8000
websocket_port = 8001
websocket_connected = set()
websocket_loop = None

class HttpHandler(http.server.SimpleHTTPRequestHandler):
  def log_message(self, format, *args):
    pass

def _serve():
  # This is a little hacky, but canonical InstallationScriptPath lives in C++, should work, be lazy for now.
  # Even hackier, prefer the dev location of files on Windows (e.g. from cpp\x64\Release to js).
  root = "../../../js" if os.path.exists("../../../js") else "js" if os.path.exists("js") else "../js"
  handler = functools.partial(HttpHandler, directory=root)
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
      message = await websocket.recv()
      await handle(json.loads(message))
  finally:
    websocket_connected.remove(websocket)

def dispatch(message):
  async def _send_one(websocket, message):
    try:
      return await websocket.send(message)
    except:
      pass

  def _send_all(websockets, message):
    if websockets:
      asyncio.create_task(asyncio.wait([_send_one(websocket, message) for websocket in websockets]))

  websocket_loop.call_soon_threadsafe(_send_all, websocket_connected, message)

async def send(message):
  await asyncio.wait([websocket.send(message) for websocket in websocket_connected])

# ----- Protocol -----

async def handle(message):
  if message["type"] == "hello":
    await send(json.dumps({
      "type": "initialize",
      "game_count": game_count,
    }))
  elif message["type"] == "request":
    requested_game = message.get("game")
    requested_position = message.get("position")
    await show_position(requested_game, requested_position)

# ----- Game and position data -----

config = network.config
games_per_chunk = config.misc["storage"]["games_per_chunk"]
chunks = tf.io.gfile.glob(network.trainer.data_globs[1])
game_count = len(chunks) * games_per_chunk

class Position:
  game = None
  chunk = None
  game_in_chunk = None
  position = None
  position_count = None
  pgn = None
  data = None

def clamp(value, min_value, max_value):
  return max(min_value, min(max_value, value))

async def show_position(requested_game, requested_position):
  # Handle game request.
  game = requested_game if requested_game is not None else Position.game if Position.game is not None else 0
  game = clamp(game, 0, game_count - 1)
  if game != Position.game and requested_position is None:
    requested_position = 0
  Position.game = game

  # Handle position request.
  position = requested_position if requested_position is not None else Position.position if Position.position is not None else 0 # Gets clamped in C++

  chunk = Position.game // games_per_chunk
  game_in_chunk = Position.game % games_per_chunk
  
  if chunk != Position.chunk:
    # Send chunk contents to C++.
    chunk_contents = tf.io.gfile.GFile(chunks[chunk], "rb").read()
    chesscoach.load_chunk(chunk_contents)
    Position.chunk = chunk
    Position.game_in_chunk = None
    Position.position = None

  if game_in_chunk != Position.game_in_chunk:
    # Parse game in C++.
    Position.position_count, Position.pgn = chesscoach.load_game(game_in_chunk)
    Position.game_in_chunk = game_in_chunk
    Position.position = None

  # C++ may update and clamp "position", e.g. if passing "-1" to represent the final position.
  if position != Position.position:
    # Get position data from C++.
    Position.position, *Position.data = chesscoach.load_position(position)
    
  # Send to JS.
  fen, mcts_value, sans, froms, tos, policy_values = Position.data
  await send(json.dumps({
    "type": "data",
    "game": Position.game,
    "position_count": Position.position_count,
    "position": Position.position,
    "pgn": Position.pgn,
    "fen": fen,
    "mcts_value": round(mcts_value, 6),
    "policy": [{ "san": san.decode("utf-8"), "from": move_from.decode("utf-8"), "to": move_to.decode("utf-8"), "value": round(float(value), 6)}
      for (san, move_from, move_to, value) in zip(sans, froms, tos, policy_values)],
  }))

# Example data format:
#
# {
#   "type": "data",
#   "game": 2,
#   "position_count": 8,
#   "position": 4,
#   "pgn": "...",
#   "fen": "3rkb1r/p2nqppp/5n2/1B2p1B1/4P3/1Q6/PPP2PPP/2KR3R w k - 3 13",
#   "mcts_value": 0.6,
#   "policy": [
#     { "san": "e4", "from": "e2", "to": "e4", "value": 0.25 },
#     { "san": "d4", "from": "d2", "to": "d4", "value": 0.75 },
#   ]
# }

# ----- API -----

def launch():
  _ensure_serving()
  webbrowser.open(f"http://localhost:{port}/gui.html")