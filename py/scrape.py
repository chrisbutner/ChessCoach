import requests
from bs4 import BeautifulSoup
import base64
import os
import unicodedata
import time
import re

# --- Common ---

api_key = "API_KEY_GOES_HERE"
endpoint = "https://app.scrapingbee.com/api/v1"
sleep_seconds = 5
html_cache = "html_cache"
pgn_cache = "pgn_cache"
pgn_header = """[Event ""]
[Site ""]
[Date ""]
[Round ""]
[White ""]
[Black ""]
[Result "?"]
"""

# If you'd like to use something other than ScrapingBee, just replace "do_scrape(url)" with your own implementation,
# including appropriate JS rendering, ad blocking and rate limiting.
def do_scrape(url):
  params = dict(api_key=api_key, url=url, block_ads="true")
  while True:
    print("ScrapingBee:", url)
    response = requests.get(endpoint, params=params)
    content = response.content.decode("utf-8")
    if response.status_code == 200:
      return content
    elif response.status_code == 500:
      print(f"Retrying {url} after {sleep_seconds} seconds ({response.status_code}): {content}")
      time.sleep(sleep_seconds)
    else:
      raise Exception(f"Failed to scrape {url} ({response.status_code}): {content}")

def scrape(url):
  os.makedirs(html_cache, exist_ok=True)
  cache_path = get_cache_path(html_cache, url) + ".html"
  try:
    with open(cache_path, "r", encoding="utf-8") as f: 
      content = f.read()
  except:
    content = do_scrape(url)
    write_file(cache_path, content)
  return content

def normalize(s):
  return unicodedata.normalize("NFKC", s)

def get_cache_path(directory, url):
  path = os.path.join(directory, base64.b32encode(url.encode("utf-8")).decode("utf-8"))
  path = os.path.abspath(path)
  path = path[:240]
  return path

def write_file(path, content):
  temporary_path = path + ".part"
  with open(temporary_path, "w", encoding="utf-8") as f: 
    f.write(content)
  os.rename(temporary_path, path)

# --- gameknot.com ---

class GameKnot:

  url_base = "https://gameknot.com/"
  url_root = "https://gameknot.com/list_annotated.pl?u=all&c=0&sb=0&rm=0&rn=0&rx=9999&sr=0&p=0"

  def find_next_url(self, soup):
    next_url = soup.select("table.paginator a[title='next page']")
    return requests.compat.urljoin(self.url_base, next_url[0]["href"]) if next_url else None
    
  def parse_game_segment(self, content):
    soup = BeautifulSoup(content, 'html.parser')
    pgn = ""

    # Parse this page and produce a .pgn segment.
    cells = soup.select(".dialog > tbody > tr > td")
    expect_moves = True
    for cell in cells:
      if not "vertical-align: top" in cell.get("style", ""):
        continue
      text = cell.get_text().strip()
      if expect_moves:
        pgn += normalize(text.splitlines()[0]) + "\n"
      elif text:
        pgn += "{ " + normalize(text) + " }\n"
      expect_moves = not expect_moves

    # Check for a next page.
    next_url = self.find_next_url(soup)
    return pgn, next_url

  def parse_list(self, content):
    soup = BeautifulSoup(content, 'html.parser')
    games = []

    # Parse this page and produce a list of game URLs.
    links = soup.select("a")
    for link in links:
      if link.string and "Comments" in link.string and "gm=" in link.get("href", ""):
        games.append(requests.compat.urljoin(self.url_base, link["href"]))

    # Check for a next page.
    next_url = self.find_next_url(soup)
    return games, next_url

  def scrape_parse_game(self, url):
    print("Game:", url)
    os.makedirs(pgn_cache, exist_ok=True)
    cache_path = get_cache_path(pgn_cache, url) + ".pgn"
    try:
      with open(cache_path, "r", encoding="utf-8") as f: 
        pgn = f.read()
    except:
      next_url = url
      pgn = pgn_header
      while next_url:
        pgn_segment, next_url = self.parse_game_segment(scrape(next_url))
        pgn += pgn_segment
      write_file(cache_path, pgn)
    return pgn

  def scrape_parse_all_games(self):
    next_url = self.url_root
    while next_url:
      games, next_url = self.parse_list(scrape(next_url))
      for game_url in games:
        self.scrape_parse_game(game_url)

# --- chessgames.com ---

class ChessGamesDotCom:

  url_base = "https://www.chessgames.com/"
  url_root = "https://www.chessgames.com/perl/chess.pl?annotated=1"

  def find_next_url(self, soup):
    next_url = soup.select("img[src='/chessimages/next.gif']")
    return requests.compat.urljoin(self.url_base, next_url[0].parent["href"]) if next_url else None
  
  def parse_list(self, content):
    soup = BeautifulSoup(content, 'html.parser')
    games = []

    # Parse this page and produce a list of game URLs.
    links = soup.select("a")
    for link in links:
      href = link.get("href", "")
      if "gid=" in href and not "comp=1" in href:
        game_id = int(re.match(".*gid=([0-9]+).*", href).group(1))
        games.append(requests.compat.urljoin(self.url_base, f"/perl/nph-chesspgn?text=1&gid={game_id}"))

    # Check for a next page.
    next_url = self.find_next_url(soup)
    return games, next_url

  def scrape_parse_game(self, url):
    print("Game:", url)
    os.makedirs(pgn_cache, exist_ok=True)
    cache_path = get_cache_path(pgn_cache, url) + ".pgn"
    try:
      with open(cache_path, "r", encoding="utf-8") as f: 
        pgn = f.read()
    except:
      content = scrape(url)
      soup = BeautifulSoup(content, 'html.parser')
      pgn = soup.get_text().strip()
      write_file(cache_path, pgn)
    return pgn

  def scrape_parse_all_games(self):
    next_url = self.url_root
    while next_url:
      games, next_url = self.parse_list(scrape(next_url))
      for game_url in games:
        self.scrape_parse_game(game_url)

# --- Run ---

print("Working directory:", os.getcwd())
#GameKnot().scrape_parse_all_games()
#ChessGamesDotCom().scrape_parse_all_games()