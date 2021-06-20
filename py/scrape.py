import requests
from bs4 import BeautifulSoup
import base64
import os
import unicodedata
import time
import re

# --- Common ---

api_key = os.environ["SCRAPINGBEE_API_KEY"]
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

# If you'd like to use something other than ScrapingBee, just replace "do_scrape()" with your own implementation,
# including appropriate ad blocking, rate limiting and error handling.
def do_scrape(url, render_js):
  params = dict(api_key=api_key, url=url, render_js=render_js, block_ads="true")
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

def scrape(url, render_js=False, cache=True):
  if cache:
    cache_path = get_cache_path(html_cache, url) + ".html"
    try:
      with open(cache_path, "r", encoding="utf-8") as f: 
        content = f.read()
    except:
      content = do_scrape(url, render_js)
      write_file(cache_path, content)
  else:
    content = do_scrape(url, render_js)
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

# --- Site1 ---

class Site1:

  url_base = "https://example.com/"
  url_root = "https://example.com/list_annotated.pl?u=all&c=0&sb=0&rm=0&rn=0&rx=9999&sr=0&p=0"

  def find_next_url(self, soup):
    next_url = soup.select("table.paginator a[title='next page']")
    return requests.compat.urljoin(self.url_base, next_url[0]["href"]) if next_url else None
    
  def scrape_parse_game_segment(self, url):
    content = scrape(url)
    soup = BeautifulSoup(content, 'html.parser')
    pgn = ""

    # Parse this page and produce a .pgn segment (need tbody if JS-rendered in cache).
    cells = soup.select("table.dialog > tr > td") or soup.select("table.dialog > tbody > tr > td")
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

  def scrape_parse_list(self, url):
    content = scrape(url)
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
    cache_path = get_cache_path(pgn_cache, url) + ".pgn"
    try:
      with open(cache_path, "r", encoding="utf-8") as f: 
        pgn = f.read()
    except:
      next_url = url
      pgn = pgn_header
      while next_url:
        pgn_segment, next_url = self.scrape_parse_game_segment(next_url)
        pgn += pgn_segment
      write_file(cache_path, pgn)
    return pgn

  def scrape_parse_all_games(self):
    next_url = self.url_root
    while next_url:
      games, next_url = self.scrape_parse_list(next_url)
      for game_url in games:
        self.scrape_parse_game(game_url)

# --- Site2 ---

class Site2:

  url_base = "https://www.example.com/"
  url_root = "https://www.example.com/perl/chess.pl?annotated=1"

  def find_next_url(self, soup):
    next_url = soup.select("img[src='/chessimages/next.gif']")
    return requests.compat.urljoin(self.url_base, next_url[0].parent["href"]) if next_url else None
  
  def scrape_parse_list(self, url):
    content = scrape(url)
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
      games, next_url = self.scrape_parse_list(next_url)
      for game_url in games:
        self.scrape_parse_game(game_url)

# --- Site3 ---

class Site3:

  url_base = "https://www.example.com/"
  url_root = "https://www.example.com/news"

  def has_comments(self, pgn):
    # Ignore games with almost no comments, or with only clock times.
    return (pgn.count("{") - len(re.findall("{\s*\\[", pgn))) >= 3

  def find_next_url(self, soup):
    next_url = soup.select("a.pagination-next")
    return requests.compat.urljoin(self.url_base, next_url[0]["href"]) if next_url else None

  def scrape_parse_list(self, url):
    # Can't cache news, it's most-recent-first.
    content = scrape(url, cache=False)
    soup = BeautifulSoup(content, 'html.parser')

    # Parse this page and produce a list of article URLs, each of which may contain multiple games.
    articles = [requests.compat.urljoin(self.url_base, link["href"]) for link in soup.select("a.post-preview-title")]

    # Check for a next page.
    next_url = self.find_next_url(soup)
    return articles, next_url

  def scrape_parse_game(self, synthetic_url, pgn):
    print("Game:", synthetic_url)
    cache_path = get_cache_path(pgn_cache, synthetic_url) + ".pgn"
    try:
      with open(cache_path, "r", encoding="utf-8"): 
        pass
    except:
      write_file(cache_path, pgn)
    return pgn

  def scrape_parse_article(self, url):
    print("Article:", url)
    content = scrape(url)
    soup = BeautifulSoup(content, 'html.parser')
    diagrams = soup.select(".chessDiagramDiv")
    saved_headers = set()
    game_number = 0
    for diagram in diagrams:
      if not diagram.string:
        print("Broken diagram:", url)
        continue
      games = re.findall("(\\[Event.*?)(\\&-|$)", diagram.string, re.DOTALL)
      for game in games:
        game_number += 1
        synthetic_url = f"{url}#game{game_number}"
        pgn = game[0]
        if not self.has_comments(pgn):
          continue
        headers = re.search("\\[.*\"\\]", pgn, re.DOTALL).group(0)
        if headers in saved_headers:
          print("Duplicate:", synthetic_url)
          continue
        saved_headers.add(headers)
        self.scrape_parse_game(synthetic_url, pgn)

  def scrape_parse_all_games(self):
    next_url = self.url_root
    while next_url:
      articles, next_url = self.scrape_parse_list(next_url)
      for article_url in articles:
        self.scrape_parse_article(article_url)

# --- Utilities ---

# Really rough, final split is shorter. Intended to operate over one-game-per (or equal-per) pgn files.
def combine_split(dir_in, dir_out, split_count):
  os.makedirs(dir_out, exist_ok=True)
  count_in = len(os.listdir(dir_in))
  per_split = (count_in + split_count - 1) // split_count
  count_split = 0
  count_out = 0
  content_out = ""
  for filename_in in os.listdir(dir_in):
    path_in = os.path.join(dir_in, filename_in)
    with open(path_in, "r", encoding="utf-8") as f:
      content_in = f.read()
    content_out += content_in + ("\n" if content_in[-1] == "\n" else "\n\n")
    count_split += 1
    if count_split >= per_split or count_split >= count_in:
      count_in -= count_split
      count_split = 0
      count_out += 1 
      path_out = os.path.join(dir_out, f"split{count_out}.pgn")
      content_out = content_out[:-1]
      with open(path_out, "w", encoding="utf-8") as f:
        f.write(content_out)
      content_out = ""

# --- Run ---

print("Working directory:", os.getcwd())
os.makedirs(html_cache, exist_ok=True)
os.makedirs(pgn_cache, exist_ok=True)
#Site1().scrape_parse_all_games()
#Site2().scrape_parse_all_games()
#Site3().scrape_parse_all_games()

# --- Usage ---

# - Make sure that the working directory is short to avoid any collisions due to MAX_PATH issues with Base32 filenames.
# - Run "combine_split" before copying the resulting PGNs to longer directories to again avoid any MAX_PATH issues.
# - Use a "split_count" that is a multiple of 8 to allow ChessCoachPgnToGames to thread efficiently for large datasets.