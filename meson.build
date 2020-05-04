project('ChessCoach', 'cpp', default_options: ['cpp_std=c++17'])

bindir = get_option('bindir')

cpp_includes = include_directories('cpp')

pymod = import('python')
python_installation = pymod.find_installation('python3')
python_dependency = python_installation.dependency(embed: true)
if not python_dependency.found()
  error('python3 dependency not found')
endif

thread_dependency = dependency('threads')

###############################################################################
# Stockfish
###############################################################################

stockfish_sources = [
  'cpp/Stockfish/benchmark.cpp',
  'cpp/Stockfish/bitbase.cpp',
  'cpp/Stockfish/bitboard.cpp',
  'cpp/Stockfish/endgame.cpp',
  'cpp/Stockfish/evaluate.cpp',
  'cpp/Stockfish/material.cpp',
  'cpp/Stockfish/misc.cpp',
  'cpp/Stockfish/movegen.cpp',
  'cpp/Stockfish/movepick.cpp',
  'cpp/Stockfish/pawns.cpp',
  'cpp/Stockfish/position.cpp',
  'cpp/Stockfish/psqt.cpp',
  'cpp/Stockfish/search.cpp',
  'cpp/Stockfish/thread.cpp',
  'cpp/Stockfish/timeman.cpp',
  'cpp/Stockfish/tt.cpp',
  'cpp/Stockfish/uci.cpp',
  'cpp/Stockfish/ucioption.cpp',
  'cpp/Stockfish/syzygy/tbprobe.cpp',
  ]

stockfish = static_library(
  'stockfish',
  stockfish_sources,
  )

###############################################################################
# ChessCoach
###############################################################################

chesscoach_sources = [
  'cpp/ChessCoach/ChessCoach.cpp',
  'cpp/ChessCoach/Config.cpp',
  'cpp/ChessCoach/Epd.cpp',
  'cpp/ChessCoach/Game.cpp',
  'cpp/ChessCoach/Pgn.cpp',
  'cpp/ChessCoach/Platform.cpp',
  'cpp/ChessCoach/PoolAllocator.cpp',
  'cpp/ChessCoach/PredictionCache.cpp',
  'cpp/ChessCoach/PythonNetwork.cpp',
  'cpp/ChessCoach/SavedGame.cpp',
  'cpp/ChessCoach/SelfPlay.cpp',
  'cpp/ChessCoach/Storage.cpp',
  'cpp/ChessCoach/Threading.cpp',
  ]

chesscoach = static_library(
  'ChessCoach',
  chesscoach_sources,
  include_directories: cpp_includes,
  dependencies: [python_dependency, thread_dependency],
  )

###############################################################################
# ChessCoachUci
###############################################################################

chesscoachuci_sources = [
  'cpp/ChessCoachUci/ChessCoachUci.cpp',
  ]

chesscoach_includes = include_directories('cpp')

chesscoach = executable(
  'ChessCoachUci',
  chesscoachuci_sources,
  include_directories: cpp_includes,
  dependencies: python_dependency,
  link_with: [chesscoach, stockfish],
  install: true,
  )

###############################################################################
# ChessCoachTest
###############################################################################

#gtest_dep = dependency('gtest', main : true, required : false)

###############################################################################
# Install config
###############################################################################

install_data('config.toml')

python_sources = [
  'py/config.py',
  'py/model.py',
  'py/network.py',
  'py/storage.py',
  ]

install_data(python_sources, install_dir: bindir + '/ChessCoach')