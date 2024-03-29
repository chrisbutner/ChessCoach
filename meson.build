project('ChessCoach', 'cpp', version: '1.0.0', license: 'GPL3',
  default_options: [
    'cpp_std=c++17',
    'werror=true',
  ])

if get_option('buildtype') == 'release'
  compiler = meson.get_compiler('cpp')
  add_project_arguments(compiler.get_supported_arguments(['-march=native', '-O2', '-DNDEBUG', '-flto']), language: 'cpp')
endif

# Support C++17 on GCC8, faster CRC32C and faster Stockfish (no prefetch though: we don't use their transposition table).
add_project_arguments('-lstdc++fs', '-msse', '-msse2', '-msse3', '-msse4.1', '-msse4.2', '-mavx2', '-mpopcnt', '-mbmi2',
  '-DNO_PREFETCH', '-DUSE_POPCNT', '-DUSE_PEXT', '-DIS_64BIT', language: 'cpp')
add_project_link_arguments('-lstdc++fs', language: 'cpp')

# Rely on MSVC's less pedantic signed/unsigned checking, e.g. around loop indices.
add_project_arguments('-Wall', '-Wextra', '-Wpedantic', '-Wno-unknown-pragmas', '-Wno-sign-compare', language: 'cpp')

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
protobuf_dependency = dependency('protobuf', version: '3.13.0', static: true)
zlib_dependency = dependency('zlib', version: '>=1.2.8')

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
  override_options: ['werror=false'],
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
# crc32c
###############################################################################

crc32c_sources = [
  'cpp/crc32c/src/crc32c.cc',
  'cpp/crc32c/src/crc32c_portable.cc',
  'cpp/crc32c/src/crc32c_sse42.cc',
  ]

crc32c = static_library(
  'crc32c',
  crc32c_sources,
  include_directories: include_directories('cpp/crc32c/include'),
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
  'cpp/ChessCoach/Preprocessing.cpp',
  'cpp/ChessCoach/PythonModule.cpp',
  'cpp/ChessCoach/PythonNetwork.cpp',
  'cpp/ChessCoach/Random.cpp',
  'cpp/ChessCoach/SavedGame.cpp',
  'cpp/ChessCoach/SelfPlay.cpp',
  'cpp/ChessCoach/Storage.cpp',
  'cpp/ChessCoach/Syzygy.cpp',
  'cpp/ChessCoach/Threading.cpp',
  'cpp/ChessCoach/WorkerGroup.cpp',
  ]

chesscoach = static_library(
  'ChessCoach',
  chesscoach_sources,
  include_directories: cpp_includes,
  dependencies: [python_dependency, thread_dependency, zlib_dependency, protobuf_dependency],
  )

###############################################################################
# chesscoachprotobuf (generated code in "protobuf" dir, using protoc.cmd/protoc.sh)
###############################################################################

chesscoachprotobuf_sources = [
  'cpp/protobuf/ChessCoach.pb.cc',
]

chesscoachprotobuf = static_library(
  'chesscoachprotobuf',
  chesscoachprotobuf_sources,
  dependencies: [protobuf_dependency],
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
  link_with: [chesscoach, chesscoachprotobuf, stockfish, hunspell, crc32c],
  install: true,
  )

###############################################################################
# ChessCoachBot
###############################################################################

chesscoachbot_sources = [
  'cpp/ChessCoachBot/ChessCoachBot.cpp',
  ]

chesscoachbot = executable(
  'ChessCoachBot',
  chesscoachbot_sources,
  include_directories: cpp_includes,
  link_with: [chesscoach, chesscoachprotobuf, stockfish, hunspell, crc32c],
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
  link_with: [chesscoach, chesscoachprotobuf, stockfish, hunspell, crc32c],
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
  link_with: [chesscoach, chesscoachprotobuf, stockfish, hunspell, crc32c],
  install: true,
  )

###############################################################################
# ChessCoachGui
###############################################################################

chesscoachgui_sources = [
  'cpp/ChessCoachGui/ChessCoachGui.cpp',
  ]

chesscoachgui = executable(
  'ChessCoachGui',
  chesscoachgui_sources,
  include_directories: [cpp_includes, tclap_includes],
  link_with: [chesscoach, chesscoachprotobuf, stockfish, hunspell, crc32c],
  install: true,
  )

###############################################################################
# ChessCoachOptimizeParameters
###############################################################################

chesscoachoptimizeparameters_sources = [
  'cpp/ChessCoachOptimizeParameters/ChessCoachOptimizeParameters.cpp',
  ]

chesscoachoptimizeparameters = executable(
  'ChessCoachOptimizeParameters',
  chesscoachoptimizeparameters_sources,
  include_directories: [cpp_includes, tclap_includes],
  link_with: [chesscoach, chesscoachprotobuf, stockfish, hunspell, crc32c],
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
  link_with: [chesscoach, chesscoachprotobuf, stockfish, hunspell, crc32c],
  install: true,
  )

###############################################################################
# bayeselo
###############################################################################

bayeselo_sources = [
  'tools/deb/bayeselo/bayeselo.cpp',
  ]

bayeselo = executable(
  'bayeselo',
  bayeselo_sources,
  override_options: ['werror=false'],
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
  ]

chesscoachtest = executable(
  'ChessCoachTest',
  chesscoachtest_sources,
  include_directories: cpp_includes,
  dependencies: [gtest_dependency],
  link_with: [chesscoach, chesscoachprotobuf, stockfish, hunspell, crc32c],
)

test('AllTests', chesscoachtest, timeout: 300)

###############################################################################
# Install
###############################################################################

# Install cutechess-cli
install_data('tools/deb/cutechess-cli/cutechess-cli', install_dir: bindir)

# Install Stockfish
install_data('tools/deb/stockfish_13_linux_x64_bmi2/stockfish_13_linux_x64_bmi2', install_dir: bindir)

# Access by Platform::InstallationDataPath()
install_data('config.toml')
install_subdir('cpp/StrengthTests', install_dir: datadir + '/ChessCoach')
install_subdir('cpp/Dictionaries', install_dir: datadir + '/ChessCoach')

python_sources = [
  'py/config.py',
  'py/dataset.py',
  'py/gui.py',
  'py/lichessbot.py',
  'py/model.py',
  'py/network.py',
  'py/optimization.py',
  'py/suites.py',
  'py/tokenization.py',
  'py/training.py',
  'py/transformer.py',
  'py/uci_proxy_client.py',
  'py/uci_proxy_server.py',
  ]

# Access by Platform::InstallationScriptPath()
install_data(python_sources, install_dir: bindir + '/ChessCoach')

# Access Games, Networks, etc. by Platform::UserDataPath()