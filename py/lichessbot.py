# Based on https://github.com/ShailChoksi/lichess-bot
#
# AGPL3-or-later license (see lichess-bot/LICENSE and https://github.com/ShailChoksi/lichess-bot#license)
#
# Modifications by Chris Butner, 2021

import os
import json
import threading
import queue
import time
import random
import backoff
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, HTTPError, ReadTimeout
from urllib3.exceptions import ProtocolError
from http.client import RemoteDisconnected
from urllib.parse import urljoin
import textwrap
from bs4 import BeautifulSoup
import numpy as np

try:
  import chesscoach # See PythonModule.cpp
except:
  pass

# --- Constants ---

WHITE = 0
BLACK = 1
STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# --- Data structures ---

class MissingPieces:

  def __init__(self, below_rating, fens):
    self.below_rating = below_rating
    self.fens = fens

  def is_supported(self, rating, fen):
    return (fen in self.fens) and (rating < self.below_rating)

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

    self.initial_fen = event.get("initialFen")
    self.missing_pieces = [
      # WHITE is missing pieces.
      self.set_up_missing_pieces(lambda x: x.upper()),
      # BLACK is missing pieces.
      self.set_up_missing_pieces(lambda x: x),
    ]

  # Opponent must have 2999 rating or below to allow pawn odds, 1499 rating or below to allow queen odds, etc.
  # Go crazy and allow kingside pieces, any pawn combinations, first move, whatever.
  def set_up_missing_pieces(self, filter):
    return [
        MissingPieces(3000, self.remove_piece(STARTING_POSITION, filter("p"))),
        MissingPieces(2850, self.remove_two_pieces(STARTING_POSITION, filter("p"))),
        MissingPieces(2500, self.remove_piece(STARTING_POSITION, filter("n"))),
        MissingPieces(2500, self.remove_piece(STARTING_POSITION, filter("b"))),
        MissingPieces(2000, self.remove_piece(STARTING_POSITION, filter("r"))),
        MissingPieces(1500, self.remove_piece(STARTING_POSITION, filter("q"))),
      ]

  def remove_piece(self, fen, piece):
    return set((fen[:i] + "1" + fen[i + 1:]).replace("11", "2") for i, existing in enumerate(fen) if existing == piece)

  def remove_two_pieces(self, fen, piece):
    all = ([self.remove_piece(removed, piece) for removed in self.remove_piece(fen, piece)])
    return set([item for sublist in all for item in sublist]) # The set de-duplicates.

  def is_supported_variant(self):
    return (
      # Allow standard chess. Have to support arbitrary colors to allow for rematches.
      (self.variant == "standard") or
      # Allow piece odds, but only when the opponent has all pieces and we're missing some.
      (self.variant == "fromPosition" and self.color == "white" and
        any(m.is_supported(self.challenger_rating_int, self.initial_fen) for m in self.missing_pieces[BLACK])) or
      (self.variant == "fromPosition" and self.color == "black" and
        any(m.is_supported(self.challenger_rating_int, self.initial_fen) for m in self.missing_pieces[WHITE])))

  def is_fast_enough(self):
    # Limit games to 2.5 hours based on 80 moves per side.
    return (2 * (self.base + self.increment * 80)) <= 150 * 60

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

  comment_frequency_seconds = 30.0
  abort_seconds = 30.0 # 10 for first white move, 10 for first black move, 10 buffer

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
    self.abort_at = time.time() + self.abort_seconds

    self.searching_moves = "(not searching yet)"
    self.pending_comment = None
    self.last_comment_time = time.time() - random.uniform(0.0, self.comment_frequency_seconds) - 0.5

  def url(self):
    return urljoin(self.base_url, "{}/{}".format(self.id, self.my_color))

  def is_abortable(self):
    return len(self.state["moves"]) < 6

  def on_activity(self):
    self.abort_at = time.time() + self.abort_seconds

  def should_abort_now(self):
    return self.is_abortable() and time.time() > self.abort_at

  # Only allow commenting every 30 seconds, and delay after 429-errors.
  def try_pop_comment(self):
    comment = self.pending_comment
    if not comment:
      return None
    now = time.time()
    if (now - self.last_comment_time) < self.comment_frequency_seconds:
      return None
    if not throttle.ready():
      return None
    self.pending_comment = None
    self.last_comment_time = now
    return comment

  # Additional optimization layer: only bother generating a comment when it can be immediately posted.
  def can_comment(self):
    now = time.time()
    return (not self.pending_comment) and ((now - self.last_comment_time) >= self.comment_frequency_seconds)

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
    self.queue = set()
    self.in_progress = {}
    self.last_finish_time = time.time()

  def enqueue(self, id):
    before = self.status()
    self.queue.add(id)
    if logging_enabled:
      log(f"[Games][Enqueue][{id}] {before} -> {self.status()}")

  def dequeue(self, id):
    before = self.status()
    try:
      self.queue.remove(id)
    except KeyError:
      if logging_enabled:
        log(f"[Games][Dequeue][{id}] Wasn't in queue (no action)")
      return
    if logging_enabled:
      log(f"[Games][Dequeue][{id}] {before} -> {self.status()}")

  def starting(self, id):
    if id in self.in_progress:
      if logging_enabled:
        log(f"[Games][Starting][{id}] Already tracked (no action)")
      return
    before = self.status()
    try:
      self.queue.remove(id)
    except KeyError:
      if logging_enabled:
        log(f"[Games][Starting][{id}] Wasn't in queue")
      # Unfortunately, there's no direct correlation between a rematch challenge ID (which remains the same as the original game)
      # and the resulting rematch game ID. So, rely on our only playing one game at a time and just dequeue a single game and log
      # when the IDs don't match.
      if len(self.queue) == 1:
        existing_id = next(iter(self.queue))
        self.queue.clear()
        if logging_enabled:
          log(f"[Games][Starting][{id}] Assuming a rematch, dequeuing {existing_id}")
    self.in_progress[id] = None
    if logging_enabled:
      log(f"[Games][Starting][{id}] {before} -> {self.status()}")

  def started(self, game):
    id = game.id
    assert not self.in_progress.get(id, None) # A None in the dictionary is okay (starting) but not a real Game.
    before = self.status()
    self.in_progress[id] = game
    if logging_enabled:
      log(f"[Games][Started][{id}] {before} -> {self.status()}")

  def finished(self, id):
    if id not in self.in_progress:
      if logging_enabled:
        log(f"[Games][Finished][{id}] Wasn't tracked (no action)")
      return
    before = self.status()
    self.in_progress.pop(id)
    self.last_finish_time = time.time()
    if logging_enabled:
      log(f"[Games][Finished][{id}] {before} -> {self.status()}")

  def status(self):
    return f"(queue: {self.queue}, in_progress: {self.in_progress})"

# --- Protocol ---

ENDPOINTS = {
  "profile": "/api/account",
  "playing": "/api/account/playing",
  "stream": "/api/bot/game/stream/{}",
  "stream_event": "/api/stream/event",
  "bots": "/player/bots",
  "status": "/api/users/status",
  "game": "/api/bot/game/{}",
  "move": "/api/bot/game/{}/move/{}",
  "chat": "/api/bot/game/{}/chat",
  "abort": "/api/bot/game/{}/abort",
  "accept": "/api/challenge/{}/accept",
  "decline": "/api/challenge/{}/decline",
  "challenge": "/api/challenge/{}",
  "cancel": "/api/challenge/{}/cancel",
  "upgrade": "/api/bot/account/upgrade",
  "resign": "/api/bot/game/{}/resign",
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
  def api_get(self, path, params=None):
    url = urljoin(self.baseUrl, path)
    if logging_enabled:
      log(url)
    response = self.session.get(url, params=params, timeout=2)
    response.raise_for_status()
    return response.json()

  @backoff.on_exception(backoff.constant,
                        (RemoteDisconnected, ConnectionError, ProtocolError, HTTPError, ReadTimeout),
                        max_time=60,
                        interval=0.1,
                        giveup=is_final)
  def api_post(self, path, data=None, headers=None):
    url = urljoin(self.baseUrl, path)
    if logging_enabled:
      log(url)
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
    if logging_enabled:
      log(url)
    return requests.get(url, headers=self.header, stream=True)

  def get_game_stream(self, game_id):
    url = urljoin(self.baseUrl, ENDPOINTS["stream"].format(game_id))
    if logging_enabled:
      log(url)
    return requests.get(url, headers=self.header, stream=True)

  def get_bots(self):
    url = urljoin(self.baseUrl, ENDPOINTS["bots"])
    if logging_enabled:
      log(url)
    return requests.get(url)

  def get_status(self, ids):
    return self.api_get(ENDPOINTS["status"], params={ "ids": ids })

  def accept_challenge(self, challenge_id):
    return self.api_post(ENDPOINTS["accept"].format(challenge_id))

  def decline_challenge(self, challenge_id, reason="generic"):
    payload = { "reason": reason }
    return self.api_post(ENDPOINTS["decline"].format(challenge_id), data=payload)

  def send_challenge(self, username, clock_limit, clock_increment):
    payload = { "rated": "true", "clock.limit": clock_limit, "clock.increment": clock_increment }
    return self.api_post(ENDPOINTS["challenge"].format(username), data=payload)

  def cancel_challenge(self, challenge_id):
    return self.api_post(ENDPOINTS["cancel"].format(challenge_id))

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
    pass

  def send_reply(self, line, reply):
    self.lichess.chat(self.game.id, line.room, reply)

class Throttle:

  delay_seconds = 65.0

  def __init__(self):
    self.lock = threading.Lock()
    self.ready_time = time.time() - 1000.0

  def on_exception(self, exception):
    with self.lock:
      if isinstance(exception, HTTPError) and (exception.response.status_code == 429):
        if logging_enabled:
          log(f"*** 429 *** for {exception.request.url}: delaying optional requests by {self.delay_seconds} seconds")
        self.ready_time = (time.time() + self.delay_seconds)

  def ready(self):
    with self.lock:
      return (time.time() >= self.ready_time)

# --- Sending challenges when idle ---

class OutgoingChallenges:

  idle_seconds_before_send = 125.0
  unanswered_seconds_before_cancel = 30.0

  minimum_best_rating = 1800
  rating_sample_temperature = 0.5

  time_controls = [
    (30, 3),
    (60, 3),
    (180, 0),
    (180, 2),
    (300, 0),
    (300, 3),
    (600, 0),
    (600, 5),
    (900, 0),
    (900, 10),
    (1800, 0),
    (1800, 20),
  ]

  def __init__(self, lichess, username, games):
    self.challenges_active = {}
    self.challenges_declined = set()
    self.challenges_forbidden = set()
    self.lichess = lichess
    self.username = username
    self.games = games
    self.rng = np.random.default_rng()
    self.last_send_attempt = time.time()

  def upkeep(self):
    # Send a new challenge? (Wait 60 seconds in case players want to challenge us, and delay between sends, and after 429-errors.)
    if (not self.games.queue
        and not self.games.in_progress
        and not self.challenges_active
        and (time.time() >= self.games.last_finish_time + self.idle_seconds_before_send)
        and (time.time() >= self.last_send_attempt + self.idle_seconds_before_send)
        and throttle.ready()):
      key = self.choose()
      if key:
        username, (limit, increment) = key
        try:
          self.last_send_attempt = time.time()
          response = self.lichess.send_challenge(username, limit, increment)
          id = response["challenge"]["id"]
          time_sent = time.time()
          self.challenges_active[id] = key, time_sent
          self.games.enqueue(id)
          if logging_enabled:
            log(f"*** OUTGOING CHALLENGE SENT *** with ID {id} to {username} with time control {limit}+{increment}")
        except HTTPError as e:
          if e.response.content and (b"does not accept challenges" in e.response.content):
            if logging_enabled:
              log(f"Forbidding challenges to {username}: {e.response.content} ({e.response.status_code})")
            self.challenges_forbidden.add(username)
          if e.response.status_code == 429:
            # Assume that all 429s sending challenges are per-day limits and wait an hour.
            self.last_send_attempt = time.time() + 3600.0
          throttle.on_exception(e)
        except Exception:
          pass

    # Cancel an unanswered challenge? (Wait until next upkeep to send another.)
    now = time.time()
    remove = []
    for id, (key, time_sent) in self.challenges_active.items():
      if (now - time_sent) >= self.unanswered_seconds_before_cancel:
        username, (limit, increment) = key
        try:
          self.lichess.cancel_challenge(id)
          if logging_enabled:
            log(f"*** CANCELED OUTGOING CHALLENGE *** with ID {id} to {username} with time control {limit}+{increment}")
        except Exception as e:
          throttle.on_exception(e)
          if logging_enabled:
            log(f"*** ABANDONING OUTGOING CHALLENGE *** with ID {id} to {username} with time control {limit}+{increment} (cancel failed)")
        remove.append(id)
    for id in remove:
      self.games.dequeue(id)
      self.challenges_active.pop(id, None)

  def game_start(self, id):
    # Most details are handled elsewhere: we just need to untrack active outgoing challenges.
    self.challenges_active.pop(id, None)

  def declined(self, event):
    id = event["challenge"]["id"]
    reason = event["challenge"]["declineReason"]
    self.games.dequeue(id)
    challenge = self.challenges_active.pop(id, None)
    if challenge:
      key, _ = challenge
      self.challenges_declined.add(key)
      username, (limit, increment) = key
      if logging_enabled:
        log(f"*** OUTGOING CHALLENGE DECLINED *** with ID {id} to {username} with time control {limit}+{increment}")
      if (("I'm not accepting challenges from bots." in reason) or
        ("I'm not willing to play this variant right now." in reason) or
        ("Please send me a casual challenge instead." in reason) or
        ("only accepts challenges from friends." in reason) or
        ("does not accept challenges." in reason) or
        ("Please register to send challenges." in reason)):
        if logging_enabled:
          log(f"Forbidding challenges to {username}: {reason}")
        self.challenges_forbidden.add(username)

  def list_users(self):
    response = self.lichess.get_bots()
    soup = BeautifulSoup(response.content, "html.parser")
    users = {}

    for link, ratings in zip(soup.select(".user-link"), soup.select(".rating")):
      username = link["href"][3:]
      if username.lower() == self.username.lower() or username in self.challenges_forbidden:
        continue
      best = 0
      for rating in ratings.select("span"):
        text = rating.get_text().replace("-", "").replace("?", "").strip()
        if text:
          best = max(best, int(text))
      if best >= self.minimum_best_rating:
        users[username] = best

    return users

  def filter_out_playing(self, users):
    status_limit = 50
    usernames = list(users.keys())
    for i in range(0, len(usernames), status_limit):
      ids = ",".join(usernames[i:i + status_limit])
      time.sleep(1.0)
      status = self.lichess.get_status(ids)
      for user_status in status:
        if user_status.get("playing", False):
          users.pop(user_status["name"])

  def sample_users(self):
    if not throttle.ready():
      return None
    users = None
    try:
      users = self.list_users()
      self.filter_out_playing(users)
      if not users:
        return None
    except Exception as e:
      throttle.on_exception(e)
      return None

    usernames = list(users.keys())
    bests = list(users.values())

    bests = (bests / np.max(bests)) ** (1.0 / self.rating_sample_temperature)
    bests /= np.sum(bests)
    
    return self.rng.choice(usernames, p=bests).item()

  def sample_time_controls(self):
    return random.choice(self.time_controls)

  def choose(self):
    for _ in range(0, 10):
      username = self.sample_users()
      time_control = self.sample_time_controls()
      if not username or not time_control:
        return None
      key = (username, time_control)
      if key not in self.challenges_declined:
        return key      
    return None

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
    except Exception as e:
        throttle.on_exception(e)
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
    except (HTTPError, ReadTimeout, RemoteDisconnected, ChunkedEncodingError, ConnectionError, ProtocolError) as e:
      throttle.on_exception(e)
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

    outgoing_challenges = OutgoingChallenges(li, user_profile["username"], games)

    while True:

      event = event_queue.get()
      if event["type"] != "ping":
        if logging_enabled:
          log(event)

      if event["type"] == "local_game_done":
        # We only play one game at a time, so make sure that no games are in progress now.
        games.finished(event["id"])
        game = None
        assert not games.in_progress
        # Stop any search/ponder in progress.
        chesscoach.bot_search(b"", b"", b"", 0, False, 0, 0, 0, 0, 0)

      elif event["type"] == "local_play_move":
        if game and game.id == event["id"]:
          try:
            li.make_move(game.id, event["move"])
          except Exception as e:
            throttle.on_exception(e)

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
        # We may have sent the challenge.
        if challenge.challenger_name.lower() != user_profile["username"].lower():
          queue_index, decline_reason = challenge.is_supported()
          if not decline_reason:
            challenge_queues[queue_index].put(challenge)
          else:
            try:
              if logging_enabled:
                log("Decline {} for reason '{}'".format(challenge, decline_reason))
              li.decline_challenge(challenge.id, reason=decline_reason)
            except Exception as e:
              throttle.on_exception(e)

      elif event["type"] == "challengeDeclined":
        challenge = Challenge(event["challenge"])
        # We may *not* have sent the challenge.
        if challenge.challenger_name.lower() == user_profile["username"].lower():
          outgoing_challenges.declined(event)

      elif event["type"] == "gameStart":
        game_id = event["game"]["id"]
        outgoing_challenges.game_start(game_id)
        if games.in_progress and game_id not in games.in_progress:
          # We accidentally sent or accepted too many challenges and can't handle this game.
          try:
            li.abort(game_id)
          except Exception as e:
            throttle.on_exception(e)
        else:
          games.starting(game_id)
          game_stream = threading.Thread(target=watch_game_stream, args=[li, game_id, event_queue, user_profile])
          game_stream.start()

      # We arrive here after regular, "local" and "ping" events.

      # Handle game upkeep.
      if game and game.should_abort_now():
        if logging_enabled:
          log("Aborting {} by lack of activity".format(game.url()))
        try:
          li.abort(game.id)
        except Exception as e:
          throttle.on_exception(e)

      # Handle control upkeep
      while not games.queue and not games.in_progress:
        try:
          challenge = pop_challenge(challenge_queues)
          if not challenge:
            break
          if logging_enabled:
            log("Accept {}".format(challenge))
          li.accept_challenge(challenge.id)
          games.enqueue(challenge.id)
        except (HTTPError, ReadTimeout) as exception:
          if isinstance(exception, HTTPError) and exception.response.status_code == 404:  # ignore missing challenge
            if logging_enabled:
              log("Skip missing {}".format(challenge))
          throttle.on_exception(exception)
        except Exception:
          break

      # Send commentary to player and spectator chat (use a separate thread).
      comment = game and game.try_pop_comment()
      if comment:
        max_line = 140
        lines = textwrap.wrap(comment, width=max_line) if (len(comment) > max_line) else [comment]
        def comment_sync():
          try:
            for line in lines:
              li.chat(game.id, "spectator", line)
          except Exception as e:
            throttle.on_exception(e)
        comment_async = threading.Thread(target=comment_sync)
        comment_async.start()

      # Send and keep track of outgoing challenges when idle.
      outgoing_challenges.upkeep()

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
  bot_side = (WHITE if game.is_white else BLACK)
  can_comment = game.can_comment()
  limit_seconds = (10 if game.is_abortable() else 0)
  
  # Start the search/ponder.
  log("Preparing to search/ponder")
  status, ply, san, comment = chesscoach.bot_search(game_id, fen, moves, bot_side, can_comment, limit_seconds,
    game.state["wtime"], game.state["btime"], game.state["winc"], game.state["binc"])
  if logging_enabled:
    log(f"*** {status.upper()} *** ply: {ply}, SAN: {san}, comment: {comment}")

  # Store the comment and process it in the main loop (we only comment every 30 seconds).
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

# --- Logging ---

logging_enabled = True

def log(message):
  print(round(time.time(), 2), message)

# --- Global state ---

event_queue = queue.Queue()
throttle = Throttle()

# --- Public API ---

def run():
  li = Lichess(os.environ["LICHESS_API_KEY"], "https://lichess.org/")
  user_profile = li.get_profile()
  if logging_enabled:
    log(user_profile)
  loop(li, user_profile)

def play_move(game_id, move):
  if logging_enabled:
    log(f"Posting move, id: {game_id}, move: {move}")
  event_queue.put({ "type": "local_play_move", "id": game_id, "move": move })