# Based on https://github.com/ShailChoksi/lichess-bot
#
# AGPL3 license (see lichess-bot/LICENSE)
#
# Modifications by Chris Butner, 2021.

import os
import json
import threading
import queue
import time
import backoff
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, HTTPError, ReadTimeout
from urllib3.exceptions import ProtocolError
from http.client import RemoteDisconnected
from urllib.parse import urljoin
import textwrap

try:
  import chesscoach # See PythonModule.cpp
except:
  pass

# --- Data structures ---

class Challenge:

  def __init__(self, event):
    self.id = event["id"]
    self.rated = event["rated"]
    self.variant = event["variant"]["key"]
    self.perf_name = event["perf"]["name"]
    self.speed = event["speed"]
    self.increment = event.get("timeControl", {}).get("increment", -1)
    self.base = event.get("timeControl", {}).get("limit", -1)
    self.challenger = event.get("challenger")
    self.challenger_title = self.challenger.get("title") if self.challenger else None
    self.challenger_is_bot = self.challenger_title == "BOT"
    self.challenger_master_title = self.challenger_title if not self.challenger_is_bot else None
    self.challenger_name = self.challenger["name"] if self.challenger else "Anonymous"
    self.challenger_rating_int = self.challenger["rating"] if self.challenger else 0
    self.challenger_rating = self.challenger_rating_int or "?"
    self.color = event["color"]

  def is_supported_variant(self):
    # TODO: Handle piece odds - simple combos - only play down - base on rating
    #{{'key': 'fromPosition', 'name': 'From Position', 'short': 'FEN'} ... 'initialFen': '...' }
    return self.variant == "standard" and self.color == "random"

  def is_fast_enough(self):
    return self.speed in ["bullet", "blitz", "rapid"]

  def is_slow_enough(self):
    return self.base >= 180 or self.increment >= 3

  # Returns (queue index, decline reason) tuple.
  # Prefer rated games over casual games.
  def is_supported(self):
    if not self.is_supported_variant():
      return None, "variant"
    if not self.is_fast_enough():
      return None, "tooSlow"
    if not self.is_slow_enough():
      return None, "tooFast"
    return (0 if self.rated else 1), None

  def mode(self):
    return "rated" if self.rated else "casual"

  def challenger_full_name(self):
    return "{}{}".format(self.challenger_title + " " if self.challenger_title else "", self.challenger_name)

  def __str__(self):
    return "{} {} challenge from {}({})".format(self.perf_name, self.mode(), self.challenger_full_name(), self.challenger_rating)

  def __repr__(self):
    return self.__str__()

class Game:

  def __init__(self, json, username, base_url):
    self.username = username
    self.id = json.get("id")
    self.speed = json.get("speed")
    clock = json.get("clock", {}) or {}
    self.clock_initial = clock.get("initial", 1000 * 3600 * 24 * 365 * 10)  # unlimited = 10 years
    self.clock_increment = clock.get("increment", 0)
    self.perf_name = json.get("perf").get("name") if json.get("perf") else "{perf?}"
    self.variant_name = json.get("variant")["name"]
    self.white = Player(json.get("white"))
    self.black = Player(json.get("black"))
    self.initial_fen = json.get("initialFen")
    self.state = json.get("state")
    self.is_white = bool(self.white.name and self.white.name.lower() == username.lower())
    self.my_color = "white" if self.is_white else "black"
    self.opponent_color = "black" if self.is_white else "white"
    self.me = self.white if self.is_white else self.black
    self.opponent = self.black if self.is_white else self.white
    self.base_url = base_url
    self.white_starts = self.initial_fen == "startpos" or self.initial_fen.split()[1] == "w"
    self.abort_at = time.time() + 20

    self.searching_moves = "(not searching yet)"
    self.pending_comment = None
    self.last_comment_time = time.time() - 1000.0

  def url(self):
    return urljoin(self.base_url, "{}/{}".format(self.id, self.my_color))

  def is_abortable(self):
    return len(self.state["moves"]) < 6

  def on_activity(self):
    self.abort_at = time.time() + 20

  def should_abort_now(self):
    return self.is_abortable() and time.time() > self.abort_at

  # Only allow commenting every 5 seconds.
  def try_pop_comment(self):
    comment = self.pending_comment
    if not comment:
      return None
    now = time.time()
    if (now - self.last_comment_time) < 5.0:
      return None
    self.pending_comment = None
    self.last_comment_time = now
    return comment

  def __str__(self):
    return "{} {} vs {}".format(self.url(), self.perf_name, self.opponent.__str__())

  def __repr__(self):
    return self.__str__()

class Player:

  def __init__(self, json):
    self.id = json.get("id")
    self.name = json.get("name")
    self.title = json.get("title")
    self.rating = json.get("rating")
    self.provisional = json.get("provisional")
    self.aiLevel = json.get("aiLevel")

  def __str__(self):
    if self.aiLevel:
      return "AI level {}".format(self.aiLevel)
    else:
      rating = "{}{}".format(self.rating, "?" if self.provisional else "")
      return "{}{}({})".format(self.title + " " if self.title else "", self.name, rating)

  def __repr__(self):
    return self.__str__()

class ChatLine:

  def __init__(self, json):
    self.room = json.get("room")
    self.username = json.get("username")
    self.text = json.get("text")

class Games:

  def __init__(self):
    self.queue_count = 0
    self.in_progress = {}

  def enqueue(self):
    before = self.status()
    self.queue_count += 1
    print(f"[Games][Enqueue] {before} -> {self.status()}")

  def starting(self, id):
    if id in self.in_progress:
      print(f"[Games][Starting][{id}] Already tracked (no action)")
      return
    before = self.status()
    if self.queue_count <= 0:
      print(f"[Games][Starting][{id}] Queue count was already zero (no decrement)")
    else:
      self.queue_count -= 1
    self.in_progress[id] = None
    print(f"[Games][Starting][{id}] {before} -> {self.status()}")

  def started(self, game):
    id = game.id
    assert not self.in_progress.get(id, None) # A None in the dictionary is okay (starting) but not a real Game.
    before = self.status()
    self.in_progress[id] = game
    print(f"[Games][Started][{id}] {before} -> {self.status()}")

  def finished(self, id):
    if id not in self.in_progress:
      print(f"[Games][Finished][{id}] Wasn't tracked (no action)")
      return
    before = self.status()
    self.in_progress.pop(id)
    print(f"[Games][Finished][{id}] {before} -> {self.status()}")

  def status(self):
    return f"(queue_count: {self.queue_count}, in_progress: {self.in_progress})"

# --- Protocol ---

ENDPOINTS = {
  "profile": "/api/account",
  "playing": "/api/account/playing",
  "stream": "/api/bot/game/stream/{}",
  "stream_event": "/api/stream/event",
  "game": "/api/bot/game/{}",
  "move": "/api/bot/game/{}/move/{}",
  "chat": "/api/bot/game/{}/chat",
  "abort": "/api/bot/game/{}/abort",
  "accept": "/api/challenge/{}/accept",
  "decline": "/api/challenge/{}/decline",
  "upgrade": "/api/bot/account/upgrade",
  "resign": "/api/bot/game/{}/resign"
}

# docs: https://lichess.org/api
class Lichess:

  def __init__(self, token, url):
    self.header = {
        "Authorization": "Bearer {}".format(token),
        "User-Agent": "ChessCoach",
    }
    self.baseUrl = url
    self.session = requests.Session()
    self.session.headers.update(self.header)

  def is_final(exception):
    return isinstance(exception, HTTPError) and exception.response.status_code < 500

  @backoff.on_exception(backoff.constant,
                        (RemoteDisconnected, ConnectionError, ProtocolError, HTTPError, ReadTimeout),
                        max_time=60,
                        interval=0.1,
                        giveup=is_final)
  def api_get(self, path):
    url = urljoin(self.baseUrl, path)
    print(url)
    response = self.session.get(url, timeout=2)
    response.raise_for_status()
    return response.json()

  @backoff.on_exception(backoff.constant,
                        (RemoteDisconnected, ConnectionError, ProtocolError, HTTPError, ReadTimeout),
                        max_time=60,
                        interval=0.1,
                        giveup=is_final)
  def api_post(self, path, data=None, headers=None):
    url = urljoin(self.baseUrl, path)
    print(url)
    response = self.session.post(url, data=data, headers=headers, timeout=2)
    response.raise_for_status()
    return response.json()

  def get_game(self, game_id):
    return self.api_get(ENDPOINTS["game"].format(game_id))

  def upgrade_to_bot_account(self):
    return self.api_post(ENDPOINTS["upgrade"])

  def make_move(self, game_id, move):
    return self.api_post(ENDPOINTS["move"].format(game_id, move))

  def chat(self, game_id, room, text):
    payload = {'room': room, 'text': text}
    return self.api_post(ENDPOINTS["chat"].format(game_id), data=payload)

  def abort(self, game_id):
    return self.api_post(ENDPOINTS["abort"].format(game_id))

  def get_event_stream(self):
    url = urljoin(self.baseUrl, ENDPOINTS["stream_event"])
    print(url)
    return requests.get(url, headers=self.header, stream=True)

  def get_game_stream(self, game_id):
    url = urljoin(self.baseUrl, ENDPOINTS["stream"].format(game_id))
    print(url)
    return requests.get(url, headers=self.header, stream=True)

  def accept_challenge(self, challenge_id):
    return self.api_post(ENDPOINTS["accept"].format(challenge_id))

  def decline_challenge(self, challenge_id, reason="generic"):
    return self.api_post(ENDPOINTS["decline"].format(challenge_id), data=f"reason={reason}", headers={"Content-Type": "application/x-www-form-urlencoded"})

  def get_profile(self):
    profile = self.api_get(ENDPOINTS["profile"])
    return profile

  def get_ongoing_games(self):
    ongoing_games = self.api_get(ENDPOINTS["playing"])["nowPlaying"]
    return ongoing_games

  def resign(self, game_id):
    self.api_post(ENDPOINTS["resign"].format(game_id))

# --- Chat ---

class Conversation:

  command_prefix = "!"

  def __init__(self, game, lichess, challenge_queues):
    self.game = game
    self.lichess = lichess
    self.challenge_queues = challenge_queues

  def react(self, line, game):
    if (line.text[0] == self.command_prefix):
      self.command(line, game, line.text[1:].lower())

  def command(self, line, game, cmd):
    if cmd == "commands" or cmd == "help" or cmd == "h" or cmd == "?":
      self.send_reply(line, "Supported commands: !eval, !queue")
    elif cmd == "eval" and line.room == "spectator":
      # TODO
      self.send_reply(line, "Still working on the !eval command")
    elif cmd == "queue":
      # TODO
      self.send_reply(line, "Still working on the !queue command")

  def send_reply(self, line, reply):
      self.lichess.chat(self.game.id, line.room, reply)

# --- Bot ---

def is_final(exception):
  return isinstance(exception, HTTPError) and exception.response.status_code < 500

def decode_json(line):
  return json.loads(line.decode("utf-8")) if line else { "type": "ping" }

def watch_control_stream(event_queue, li):
  while True:
    try:
      response = li.get_event_stream()
      lines = response.iter_lines()
      for line in lines:
        event_queue.put(decode_json(line))
    except Exception:
        pass

@backoff.on_exception(backoff.expo, BaseException, max_time=600, giveup=is_final)
def watch_game_stream(li, game_id, event_queue, user_profile):
  watch_game = None
  response = li.get_game_stream(game_id)
  lines = response.iter_lines()
  for line in lines:
    try:
      event = decode_json(line)
      if event["type"] == "gameFull":
        watch_game = Game(event, user_profile["username"], li.baseUrl)
      elif event["type"] == "gameState":
        watch_game.state = event
      event_queue.put(event)
      if watch_game and is_game_over(watch_game):
        break
    except (HTTPError, ReadTimeout, RemoteDisconnected, ChunkedEncodingError, ConnectionError, ProtocolError):
      if game_id not in (ongoing_game["gameId"] for ongoing_game in li.get_ongoing_games()):
        break
      raise
  event_queue.put({ "type": "local_game_done", "id": game_id })

def loop(li, user_profile):
    challenge_queues = [queue.Queue(), queue.Queue()]
    control_stream = threading.Thread(target=watch_control_stream, args=[event_queue, li])
    control_stream.start()

    # We only play one game at a time.
    games = Games()
    game = None

    while True:

      event = event_queue.get()
      if event["type"] != "ping":
        print(event)

      if event["type"] == "local_game_done":
        # We only play one game at a time, so make sure that no games are in progress now.
        games.finished(event["id"])
        game = None
        assert not games.in_progress
        # Stop any search/ponder in progress.
        chesscoach.bot_search(b"", b"", b"", 0, 0, 0, 0, 0, 0)

      elif event["type"] == "local_play_move":
        if game and game.id == event["id"]:
          try:
            li.make_move(game.id, event["move"])
          except Exception:
            pass

      elif event["type"] == "gameFull":
        if game:
          # We reconnected to the game stream.
          # We only play one game at a time, so make sure that it's the same game.
          assert game.id == event["id"]
          game.state = event["state"] # Only update state: maintain "searching_moves".
        else:
          game = Game(event, user_profile["username"], li.baseUrl)
          games.started(game)
        conversation = Conversation(game, li, challenge_queues)
        on_game_state(li, game)

      elif event["type"] == "chatLine":
        if game:
          conversation.react(ChatLine(event), game)

      elif event["type"] == "gameState":
        if game:
          game.state = event
          on_game_state(li, game)

      elif event["type"] == "challenge":
          challenge = Challenge(event["challenge"])
          queue_index, decline_reason = challenge.is_supported()
          if not decline_reason:
            challenge_queues[queue_index].put(challenge)
          else:
            try:
              print("Decline {} for reason '{}'".format(challenge, decline_reason))
              li.decline_challenge(challenge.id, reason=decline_reason)
            except Exception:
              pass

      elif event["type"] == "gameStart":
        game_id = event["game"]["id"]
        if not games.queue_count and games.in_progress and game_id not in games.in_progress:
          # We accidentally accepted too many challenges and can't handle this game.
          try:
            li.abort(game_id)
          except Exception:
            pass
        else:
          games.starting(game_id)
          game_stream = threading.Thread(target=watch_game_stream, args=[li, game_id, event_queue, user_profile])
          game_stream.start()

      # We arrive here after regular, "local" and "ping" events.

      # Handle game upkeep.
      if game and game.should_abort_now():
        print("Aborting {} by lack of activity".format(game.url()))
        try:
          li.abort(game.id)
        except Exception:
          pass

      # Handle control upkeep
      while not games.queue_count and not games.in_progress:
        try:
          challenge = pop_challenge(challenge_queues)
          if not challenge:
            break
          print("Accept {}".format(challenge))
          li.accept_challenge(challenge.id)
          games.enqueue()
        except (HTTPError, ReadTimeout) as exception:
          if isinstance(exception, HTTPError) and exception.response.status_code == 404:  # ignore missing challenge
            print("Skip missing {}".format(challenge))
        except Exception:
          break

      # Send commentary to player and spectator chat (use a separate thread).
      comment = game and game.try_pop_comment()
      if comment:
        max_line = 140
        lines = textwrap.wrap(comment, width=max_line) if (len(comment) > max_line) else [comment]
        def comment_sync():
          try:
            for room in ["player", "spectator"]:
              for line in lines:
                li.chat(game.id, room, line)
          except Exception:
            pass
        comment_async = threading.Thread(target=comment_sync)
        comment_async.start()

def pop_challenge(challenge_queues):
  for challenge_queue in challenge_queues:
    try:
      return challenge_queue.get_nowait()
    except queue.Empty:
      continue
  return None

def on_game_state(li, game):
  # Update the abort-game timeout upon activity.
  game.on_activity()

  # No need to search when the game's over.
  if is_game_over(game):
    return

  # We may already be searching this position (draw offers also trigger game state events).
  if game.searching_moves == game.state["moves"]:
    return
  game.searching_moves = game.state["moves"]

  # C++ expects bytes
  game_id = game.id.encode("utf-8")
  fen = game.initial_fen.encode("utf-8")
  moves = game.state["moves"].encode("utf-8")
  bot_side = (0 if game.is_white else 1)
  limit_seconds = (10 if game.is_abortable() else 0)
  
  # Start the search/ponder.
  status, ply, san, comment = chesscoach.bot_search(game_id, fen, moves, bot_side, limit_seconds,
    game.state["wtime"], game.state["btime"], game.state["winc"], game.state["binc"])
  print(f"*** {status.upper()} *** ply: {ply}, SAN: {san}, comment: {comment}")

  # Store the comment and process it in the main loop (we only comment every 5 seconds).
  if comment:
    assert (ply > 0)
    ply_before_move = (ply - 1)
    move_number = ((ply_before_move // 2) + 1)
    suffix = "." if (ply_before_move % 2 == 0) else "..."
    game.pending_comment = f'{move_number}{suffix} {san}: "{comment}"'
  else:
    game.pending_comment = None

def is_game_over(game):
  return game.state["status"] != "started"

# --- Global state ---

event_queue = queue.Queue()

# --- Public API ---

def run():
  li = Lichess(os.environ["LICHESS_API_KEY"], "https://lichess.org/")
  user_profile = li.get_profile()
  print(user_profile)
  loop(li, user_profile)

def play_move(game_id, move):
  event_queue.put({ "type": "local_play_move", "id": game_id, "move": move })