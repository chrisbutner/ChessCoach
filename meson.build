project('ChessCoach', 'cpp', version: '0.9.0', license: 'GPL3',
  default_options: [
    'cpp_std=c++17',
  ])

bindir = get_option('bindir')
datadir = get_option('datadir')

cpp_includes = include_directories('cpp')
tclap_includes = include_directories('cpp/tclap/include')

pymod = import('python')
python_installation = pymod.find_installation('python3')
python_dependency = python_installation.dependency(embed: true)
if not python_dependency.found()
  error('python3 dependency not found')
endif

gtest_dependency = dependency('gtest', main: true)
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
# hunspell
###############################################################################

hunspell_sources = [
  'cpp/hunspell/affentry.cxx',
  'cpp/hunspell/affixmgr.cxx',
  'cpp/hunspell/csutil.cxx',
  'cpp/hunspell/filemgr.cxx',
  'cpp/hunspell/hashmgr.cxx',
  'cpp/hunspell/hunspell.cxx',
  'cpp/hunspell/hunzip.cxx',
  'cpp/hunspell/phonet.cxx',
  'cpp/hunspell/replist.cxx',
  'cpp/hunspell/suggestmgr.cxx',
  ]

hunspell = static_library(
  'hunspell',
  hunspell_sources,
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
  'cpp/ChessCoach/Pipeline.cpp',
  'cpp/ChessCoach/Platform.cpp',
  'cpp/ChessCoach/PoolAllocator.cpp',
  'cpp/ChessCoach/PredictionCache.cpp',
  'cpp/ChessCoach/Preprocessing.cpp',
  'cpp/ChessCoach/PythonNetwork.cpp',
  'cpp/ChessCoach/Random.cpp',
  'cpp/ChessCoach/ReplayBuffer.cpp',
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

chesscoachuci = executable(
  'ChessCoachUci',
  chesscoachuci_sources,
  include_directories: cpp_includes,
  link_with: [chesscoach, stockfish, hunspell],
  install: true,
  )

###############################################################################
# ChessCoachPgnToGames
###############################################################################

chesscoachpgntogames_sources = [
  'cpp/ChessCoachPgnToGames/ChessCoachPgnToGames.cpp',
  ]

chesscoachpgntogames = executable(
  'ChessCoachPgnToGames',
  chesscoachpgntogames_sources,
  include_directories: [cpp_includes, tclap_includes],
  link_with: [chesscoach, stockfish, hunspell],
  install: true,
  )

###############################################################################
# ChessCoachStrengthTest
###############################################################################

chesscoachstrengthtest_sources = [
  'cpp/ChessCoachStrengthTest/ChessCoachStrengthTest.cpp',
  ]

chesscoachstrengthtest = executable(
  'ChessCoachStrengthTest',
  chesscoachstrengthtest_sources,
  include_directories: [cpp_includes, tclap_includes],
  link_with: [chesscoach, stockfish, hunspell],
  install: true,
  )

###############################################################################
# ChessCoachTrain
###############################################################################

chesscoachtrain_sources = [
  'cpp/ChessCoachTrain/ChessCoachTrain.cpp',
  ]

chesscoachtrain = executable(
  'ChessCoachTrain',
  chesscoachtrain_sources,
  include_directories: cpp_includes,
  link_with: [chesscoach, stockfish, hunspell],
  install: true,
  )

###############################################################################
# ChessCoachTest
###############################################################################

chesscoachtest_sources = [
  'cpp/ChessCoachTest/ConfigTest.cpp',
  'cpp/ChessCoachTest/GameTest.cpp',
  'cpp/ChessCoachTest/MctsTest.cpp',
  'cpp/ChessCoachTest/NetworkTest.cpp',
  'cpp/ChessCoachTest/PgnTest.cpp',
  'cpp/ChessCoachTest/PoolAllocatorTest.cpp',
  'cpp/ChessCoachTest/PredictionCacheTest.cpp',
  'cpp/ChessCoachTest/StockfishTest.cpp',
  'cpp/ChessCoachTest/StorageTest.cpp',
  ]

chesscoachtest = executable(
  'ChessCoachTest',
  chesscoachtest_sources,
  include_directories: cpp_includes,
  dependencies: [gtest_dependency],
  link_with: [chesscoach, stockfish, hunspell],
)

test('AllTests', chesscoachtest, timeout: 300)

###############################################################################
# Install
###############################################################################

# Access by Platform::InstallationDataPath()
install_data('config.toml')
install_subdir('cpp/StrengthTests', install_dir: datadir + '/ChessCoach')
install_subdir('cpp/Dictionaries', install_dir: datadir + '/ChessCoach')

python_sources = [
  'py/attention.py',
  'py/config.py',
  'py/layers.py',
  'py/model.py',
  'py/network.py',
  'py/storage.py',
  'py/transformer.py',
  ]

# Access by Platform::InstallationScriptPath()
install_data(python_sources, install_dir: bindir + '/ChessCoach')

# Access Games, Networks, etc. by Platform::UserDataPath()