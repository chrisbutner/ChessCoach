// ChessCoach, a neural network-based chess engine capable of natural-language commentary
// Copyright 2021 Chris Butner
//
// ChessCoach is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ChessCoach is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <functional>
#include <vector>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include <Stockfish/thread.h>
#include <Stockfish/uci.h>
#include <Stockfish/syzygy/tbprobe.h>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/WorkerGroup.h>
#include <ChessCoach/Pgn.h>
#include <ChessCoach/Syzygy.h>

using CommandHandler = std::function<void(std::stringstream&)>;
using CommandHandlerEntry = std::pair<std::string, CommandHandler>;

static constexpr const char OptionTypeSpin[] = "spin";
static constexpr const char OptionTypeFloat[] = "float"; // Not in the UCI spec; used for parameter optimization
static constexpr const char OptionTypeString[] = "string";
static constexpr const char OptionTypeCheck[] = "check";

class ChessCoachUci : public ChessCoach
{
public:

    void Initialize();
    void Finalize();
    void Work();

private:

    bool HandleCommand(std::stringstream& commands, std::string command);

    // UCI commands
    void HandleUci(std::stringstream& commands);
    void HandleDebug(std::stringstream& commands);
    void HandleIsReady(std::stringstream& commands);
    void HandleSetOption(std::stringstream& commands);
    void HandleRegister(std::stringstream& commands);
    void HandleUciNewGame(std::stringstream& commands);
    void HandlePosition(std::stringstream& commands);
    void HandleGo(std::stringstream& commands);
    void HandleStop(std::stringstream& commands);
    void HandleQuit(std::stringstream& commands);

    // Custom commands
    void HandleComment(std::stringstream& commands);
    void HandleGui(std::stringstream& commands);

    // Console
    void HandleConsole(std::stringstream& commands);

    void InitializeNetwork();
    void InitializeWorkers();
    void StopAndReadyWorkers();
    void PropagatePosition();

private:

    bool _quit = false;
    bool _syzygyLoaded = false;
    bool _guiLaunched = false;
    bool _isNewGame = true;
    bool _positionUpdated = true;
    std::string _positionFen = Game::StartingPosition;
    std::vector<Move> _positionMoves = {};
    std::ofstream _commandLog;
    std::vector<CommandHandlerEntry> _commandHandlers;

    std::unique_ptr<INetwork> _network;
    WorkerGroup _workerGroup;
};

int main()
{
    ChessCoachUci chessCoachUci;

    chessCoachUci.PrintExceptions();
    chessCoachUci.Initialize();

    chessCoachUci.Work();

    chessCoachUci.Finalize();

    return 0;
}

void ChessCoachUci::Initialize()
{
    // Suppress all Python/TensorFlow output so that it doesn't interfere with UCI.
    Platform::SetEnvironmentVariable("CHESSCOACH_SILENT", "1");

    InitializePython();
    InitializeStockfish();
    InitializeChessCoach();

    // Validate config.
    const int totalParallelism = (Config::Misc.Search_SearchThreads * Config::Misc.Search_SearchParallelism);
    if (totalParallelism < 256)
    {
        std::cerr << "Warning: Band width (search_threads * search_parallelism) >= 256 required for sufficient exploration" << std::endl;
    }

    // UCI commands
    _commandHandlers.emplace_back("uci", std::bind(&ChessCoachUci::HandleUci, this, std::placeholders::_1));
    _commandHandlers.emplace_back("debug", std::bind(&ChessCoachUci::HandleDebug, this, std::placeholders::_1));
    _commandHandlers.emplace_back("isready", std::bind(&ChessCoachUci::HandleIsReady, this, std::placeholders::_1));
    _commandHandlers.emplace_back("setoption", std::bind(&ChessCoachUci::HandleSetOption, this, std::placeholders::_1));
    _commandHandlers.emplace_back("register", std::bind(&ChessCoachUci::HandleRegister, this, std::placeholders::_1));
    _commandHandlers.emplace_back("ucinewgame", std::bind(&ChessCoachUci::HandleUciNewGame, this, std::placeholders::_1));
    _commandHandlers.emplace_back("position", std::bind(&ChessCoachUci::HandlePosition, this, std::placeholders::_1));
    _commandHandlers.emplace_back("go", std::bind(&ChessCoachUci::HandleGo, this, std::placeholders::_1));
    _commandHandlers.emplace_back("stop", std::bind(&ChessCoachUci::HandleStop, this, std::placeholders::_1));
    _commandHandlers.emplace_back("quit", std::bind(&ChessCoachUci::HandleQuit, this, std::placeholders::_1));

    // Custom commands
    _commandHandlers.emplace_back("comment", std::bind(&ChessCoachUci::HandleComment, this, std::placeholders::_1));
    _commandHandlers.emplace_back("gui", std::bind(&ChessCoachUci::HandleGui, this, std::placeholders::_1));

    // Console (for unsafely-threaded debug info)
    _commandHandlers.emplace_back("`", std::bind(&ChessCoachUci::HandleConsole, this, std::placeholders::_1));
    _commandHandlers.emplace_back("~", std::bind(&ChessCoachUci::HandleConsole, this, std::placeholders::_1));

    std::stringstream commandLogFilename;

    const std::time_t time = std::time(nullptr);
#pragma warning(disable:4996) // Internal buffer is immediately consumed and detached.
    commandLogFilename << std::put_time(std::localtime(&time), "ChessCoachUci_%Y%m%d_%H%M%S.log");
#pragma warning(disable:4996) // Internal buffer is immediately consumed and detached.

    const std::filesystem::path commandLogPath = (Storage::MakeLocalPath(Config::Misc.Paths_Logs) / commandLogFilename.str());
    _commandLog = std::ofstream(commandLogPath, std::ios::out);
}

void ChessCoachUci::Finalize()
{
    FinalizePython();
    FinalizeStockfish();
}

void ChessCoachUci::Work()
{
    std::string line;
    while (!_quit && std::getline(std::cin, line))
    {
        _commandLog << line << std::endl;

        std::stringstream commands(line);
        std::string token;
        do
        {
            commands >> token;
        } while (!commands.fail() && !HandleCommand(commands, token));
    }

    if (_workerGroup.IsInitialized())
    {
        _workerGroup.ShutDown();
    }

    _network.reset();
}

bool ChessCoachUci::HandleCommand(std::stringstream& commands, std::string command)
{
    for (const auto& [key, handler] : _commandHandlers)
    {
        if (command == key)
        {
            handler(commands);
            return true;
        }
    }

    return false;
}

void ChessCoachUci::HandleUci(std::stringstream& /*commands*/)
{
    // Look up default option values.
    std::map<std::string, int> intOptions;
    std::map<std::string, float> floatOptions;
    std::map<std::string, std::string> stringOptions;
    std::map<std::string, bool> boolOptions;
    for (const auto& [name, option] : Config::Misc.UciOptions)
    {
        if (option.Type == OptionTypeSpin)
        {
            intOptions[name] = 0;
        }
        else if (option.Type == OptionTypeFloat)
        {
            floatOptions[name] = 0.f;
        }
        else if (option.Type == OptionTypeString)
        {
            stringOptions[name] = "";
        }
        else if (option.Type == OptionTypeCheck)
        {
            boolOptions[name] = false;
        }
        else
        {
            throw ChessCoachException("Unsupported UCI option type: " + option.Type + " (" + name + ")");
        }
    }
    Config::LookUp(intOptions, floatOptions, stringOptions, boolOptions);

    // Reply.
    std::cout << "id name ChessCoach\n"
        "id author C. Butner\n";
    for (const auto& [name, value] : intOptions)
    {
        std::cout << "option name " << name << " type spin default " << value
            << " min " << Config::Misc.UciOptions[name].Min << " max " << Config::Misc.UciOptions[name].Max << "\n";
    }
    for (const auto& [name, value] : floatOptions)
    {
        // Have to advertise as "string" rather than "float" so that it's recognized as valid; e.g., by cutechess-cli.
        std::cout << "option name " << name << " type string default " << value << "\n";
    }
    for (const auto& [name, value] : stringOptions)
    {
        const std::string& stringValue = (value.empty() ? "none" : value);
        std::cout << "option name " << name << " type string default " << stringValue << "\n";
    }
    for (const auto& [name, value] : boolOptions)
    {
        const std::string& boolAsString = (value ? "true" : "false");
        std::cout << "option name " << name << " type check default " << boolAsString << "\n";
    }
    std::cout << "uciok" << std::endl;
}

void ChessCoachUci::HandleDebug(std::stringstream& commands)
{
    std::string setting;

    commands >> setting;

    if (setting == "on")
    {
        _workerGroup.searchState.debug = true;
    }
    else if (setting == "off")
    {
        _workerGroup.searchState.debug = false;
    }
}

void ChessCoachUci::HandleIsReady(std::stringstream& /*commands*/)
{
    InitializeWorkers();

    // "This command must always be answered with "readyok" and can be sent also when the engine is calculating
    // in which case the engine should also immediately answer with "readyok" without stopping the search."
    if (!_workerGroup.workCoordinator->CheckWorkItemsExist())
    {
        _workerGroup.workCoordinator->WaitForWorkers();

        // Propagate the position if updated.
        PropagatePosition();
    }

    std::cout << "readyok" << std::endl;
}

void ChessCoachUci::HandleSetOption(std::stringstream& commands)
{
    // This parsing uses built-in whitespace tokenization, which makes looking for "name"/"value" easy,
    // but collapses whitespace into a single space - should be fine usually.

    if (_workerGroup.IsInitialized() && _workerGroup.workCoordinator->CheckWorkItemsExist())
    {
        std::cout << "info string Cannot set options while searching" << std::endl;
        return;
    }

    // Ignore tokens before "name".
    std::string token;
    while (commands >> token)
    {
        if (token == "name")
        {
            break;
        }
    }

    // Collect the name.
    std::string name;
    while (commands >> token)
    {
        if (token == "value")
        {
            break;
        }
        if (!name.empty())
        {
            name += " ";
        }
        name += token;
    }

    // Collect the value and update.
    const auto match = Config::Misc.UciOptions.find(name);
    if (match == Config::Misc.UciOptions.end())
    {
        std::cout << "info string Unknown option name" << std::endl;
        return;
    }
    if (match->second.Type == OptionTypeSpin)
    {
        int intValue = 0;
        if (!(commands >> intValue))
        {
            std::cout << "info string Invalid spin value" << std::endl;
            return;
        }
        else if (intValue < match->second.Min)
        {
            std::cout << "info string Invalid spin value: " << intValue << " is below minimum of " << match->second.Min << std::endl;
            return;
        }
        else if (intValue > match->second.Max)
        {
            std::cout << "info string Invalid spin value: " << intValue << " is above maximum of " << match->second.Max << std::endl;
            return;
        }
        Config::Update({ { name, intValue } }, {},  {}, {});
    }
    else if (match->second.Type == OptionTypeFloat)
    {
        float floatValue = 0.f;
        if (!(commands >> floatValue))
        {
            std::cout << "info string Invalid float value" << std::endl;
            return;
        }
        Config::Update({}, { { name, floatValue } }, {}, {});
    }
    else if (match->second.Type == OptionTypeString)
    {
        std::string stringValue;
        while (commands >> token)
        {
            if (!stringValue.empty())
            {
                stringValue += " ";
            }
            stringValue += token;
        }
        if ((stringValue == "none") || (stringValue == "\"\""))
        {
            stringValue = "";
        }
        Config::Update({}, {}, { { name, stringValue } }, {});
    }
    else if (match->second.Type == OptionTypeCheck)
    {
        std::string stringValue;
        if (!(commands >> stringValue) || !((stringValue == "true") || (stringValue == "false")))
        {
            std::cout << "info string Invalid check value" << std::endl;
            return;
        }
        Config::Update({}, {}, {}, { { name, (stringValue != "false") } });
    }
    else
    {
        throw ChessCoachException("Unsupported UCI option type: " + match->second.Type + " (" + name + ")");
    }

    // Handle custom updates.
    if (((name == "search_threads") || (name == "search_parallelism") || (name == "network_weights")) && _workerGroup.IsInitialized())
    {
        throw ChessCoachException("Threads, parallelism, and weights need to be set before readying/searching");
    }
    else if (name == "network_weights")
    {
        InitializeNetwork();
        _network->UpdateNetworkWeights(Config::Network.SelfPlay.NetworkWeights);
    }
    else if (name == "syzygy")
    {
        Syzygy::Reload();
        _syzygyLoaded = true;
    }
    else if (name == "Hash")
    {
        InitializePredictionCache();
    }
}

void ChessCoachUci::HandleRegister(std::stringstream& /*commands*/)
{
}

void ChessCoachUci::HandleUciNewGame(std::stringstream& /*commands*/)
{
    // Use this as a signal not to reuse the current MCTS tree, even for a compatible position.
    _isNewGame = true;
    _positionUpdated = true;
    _positionFen = Game::StartingPosition;
    _positionMoves.clear();

    // Also clear the prediction cache, for repeatability/consistency during analysis.
    PredictionCache::Instance.Clear();

    // Work around problems with swapping out memory-mapped pages by reinitializing tablebases each game.
    Syzygy::Reload();
    _syzygyLoaded = true;
}

void ChessCoachUci::HandlePosition(std::stringstream& commands)
{
    std::string token;

    _positionUpdated = true;
    _positionFen.clear();
    _positionMoves.clear();

    commands >> token;
    if (token == "fen")
    {
        // Trust the provided FEN.
        _positionFen.reserve(128);
        while ((commands >> token) && (token != "moves"))
        {
            _positionFen += token + " ";
        }
    }
    else
    {
        // Instead of "fen" we got "startpos" or something invalid.
        _positionFen = Game::StartingPosition;
        while ((commands >> token) && (token != "moves"))
        {
        }
    }

    Game game(_positionFen, {});

    // If "moves" wasn't seen then we already consumed the rest of the line.
    while (commands >> token)
    {
        const Move move = UCI::to_move(game.GetPosition(), token);
        if (move == MOVE_NONE)
        {
            break;
        }

        game.ApplyMove(move);
    }
    _positionMoves = std::move(game.Moves());

    // It's only safe to update the controller worker's game/position while search is stopped,
    // because it involves pruning the shared Node tree. To keep this thread responsive, only
    // update _positionFen/_positionMoves here, and propagate to the controller worker on a
    // "go" command via _positionUpdated.
}

void ChessCoachUci::HandleGo(std::stringstream& commands)
{
    TimeControl timeControl = {};
    std::vector<Move> searchMoves;

    std::string token;
    while (commands >> token)
    {
        if ((token == "infinite") || (token == "inf"))
        {
            timeControl.infinite = true;
        }
        else if ((token == "nodes"))
        {
            commands >> timeControl.nodes;
        }
        else if ((token == "mate"))
        {
            commands >> timeControl.mate;
        }
        else if (token == "movetime")
        {
            commands >> timeControl.moveTimeMs;
        }
        else if (token == "wtime")
        {
            commands >> timeControl.timeRemainingMs[WHITE];
        }
        else if (token == "btime")
        {
            commands >> timeControl.timeRemainingMs[BLACK];
        }
        else if (token == "winc")
        {
            commands >> timeControl.incrementMs[WHITE];
        }
        else if (token == "binc")
        {
            commands >> timeControl.incrementMs[BLACK];
        }
        else if (token == "movestogo")
        {
            commands >> timeControl.movesToGo;
        }
        else if (token == "searchmoves")
        {
            // Set up the last "position" specified to give proper context to the "searchmoves".
            Game game(_positionFen, _positionMoves);

            // Unlike in "position ... moves ...", these moves don't depend on each other, so it's okay
            // to skip over any invalid/illegal moves and keep parsing.
            while (commands >> token)
            {
                const Move move = UCI::to_move(game.GetPosition(), token);
                if (move != MOVE_NONE)
                {
                    searchMoves.push_back(move);
                }
            }
        }
    }

    InitializeWorkers();
    StopAndReadyWorkers();

    // Launch the GUI if not yet shown.
    if (_workerGroup.searchState.gui && !_guiLaunched)
    {
        _guiLaunched = true;
        _network->LaunchGui("push");
    }

    // Clear the tree before and after "searchmoves" searches.
    if (!_workerGroup.searchState.searchMoves.empty() || !searchMoves.empty())
    {
        _isNewGame = true;
        _positionUpdated = true;
    }

    // Propagate the position if updated.
    PropagatePosition();

    // It would be more accurate to capture the start time at the top of this method, in case ChessCoach
    // is asked to play where it can lose on time, but is not given "isready" commands to prepare
    // (since initializing workers and "PropagatePosition" -> "SearchUpdatePosition" can be expensive, especially on TPU).
    //
    // However, not receiving "isready" breaks the spec and shouldn't be common. Instead, capture here, so that ad-hoc
    // usage makes more sense, rather than giving zero search time after long initializaiton.
    const auto searchStart = std::chrono::high_resolution_clock::now();

    _workerGroup.searchState.Reset(timeControl, searchStart);
    _workerGroup.searchState.searchMoves = std::move(searchMoves);

    PredictionCache::Instance.ResetProbeMetrics();

    _workerGroup.workCoordinator->ResetWorkItemsRemaining(1);
}

void ChessCoachUci::HandleStop(std::stringstream& /*commands*/)
{
    if (_workerGroup.IsInitialized())
    {
        StopAndReadyWorkers();
    }
}

void ChessCoachUci::HandleQuit(std::stringstream& /*commands*/)
{
    if (_workerGroup.IsInitialized())
    {
        StopAndReadyWorkers();
    }
    _quit = true;
}

void ChessCoachUci::HandleComment(std::stringstream& /*commands*/)
{
    InitializeWorkers();
    StopAndReadyWorkers();

    // Propagate the position if updated.
    PropagatePosition();

    _workerGroup.controllerWorker->CommentOnPosition(_network.get());
}

void ChessCoachUci::HandleGui(std::stringstream& /*commands*/)
{
    _workerGroup.searchState.gui = true;
}

void ChessCoachUci::HandleConsole(std::stringstream& commands)
{
    std::string token;
    if (!(commands >> token))
    {
        return;
    }

    if ((token == "ucb") || (token == "puct"))
    {
        bool csv = false;
        while (commands >> token)
        {
            if (token == "csv")
            {
                csv = true;
            }
            else if (token == "moves")
            {
                break;
            }
        }

        SelfPlayGame* game;
        _workerGroup.controllerWorker->DebugGame(0, &game, nullptr, nullptr, nullptr);
        SelfPlayGame puctGame = *game;

        // If "moves" wasn't seen then we already consumed the rest of the line.
        while (commands >> token)
        {
            const Move move = UCI::to_move(puctGame.GetPosition(), token);
            if (move == MOVE_NONE)
            {
                break;
            }

            puctGame.ApplyMoveWithRoot(move, puctGame.Root()->Child(move));
        }

        if (csv)
        {
            std::cout << "move,prior,value,puct,visits,weight" << std::endl;
        }

        Node* root = puctGame.Root();
        PuctContext puctContext(&_workerGroup.searchState, root);
        for (const Node& child : *root)
        {
            if (csv)
            {
                std::cout << Pgn::San(puctGame.GetPosition(), Move(child.move), true /* showCheckmate */)
                    << "," << child.Prior()
                    << "," << child.Value()
                    << "," << puctContext.CalculatePuctScoreAdHoc(&child)
                    << "," << child.visitCount
                    << "," << child.valueWeight
                    << std::endl;
            }
            else
            {
                std::cout << Pgn::San(puctGame.GetPosition(), Move(child.move), true /* showCheckmate */)
                    << " prior=" << child.Prior()
                    << " value=" << child.Value()
                    << " puct=" << puctContext.CalculatePuctScoreAdHoc(&child)
                    << " visits=" << child.visitCount
                    << " weight=" << child.valueWeight
                    << std::endl;
            }
        }
    }
    else if (token == "fen")
    {
        // Convert the last "position" specified to a standalone FEN.
        const std::string fen = Game(_positionFen, _positionMoves).GetPosition().fen();
        std::cout << fen << std::endl;
    }
}

void ChessCoachUci::InitializeNetwork()
{
    if (!_network)
    {
        _network.reset(CreateNetwork());
    }
}

void ChessCoachUci::InitializeWorkers()
{
    if (_workerGroup.IsInitialized())
    {
        return;
    }

    InitializeNetwork();
    _workerGroup.Initialize(_network.get(), nullptr /* storage */, Config::Network.SelfPlay.PredictionNetworkType,
        Config::Misc.Search_SearchThreads, Config::Misc.Search_SearchParallelism, &SelfPlayWorker::LoopSearch);

    // Delay initializing the prediction cache until the Hash option is set or isready/go/comment, to keep input responsive early.
    InitializePredictionCache();

    // Let the GUI call back in to show requested lines.
    InitializePythonModule(nullptr /* storage */, _network.get(), &_workerGroup);

    // Initialize tablebases if never done.
    if (!_syzygyLoaded)
    {
        Syzygy::Reload();
        _syzygyLoaded = true;
    }
}

void ChessCoachUci::StopAndReadyWorkers()
{
    _workerGroup.workCoordinator->ResetWorkItemsRemaining(0);
    _workerGroup.workCoordinator->WaitForWorkers();
}

void ChessCoachUci::PropagatePosition()
{
    // Propagate the position if updated.
    if (_positionUpdated)
    {
        _workerGroup.controllerWorker->SearchUpdatePosition(_positionFen, _positionMoves, _isNewGame /* forceNewPosition */);
        _isNewGame = false;
        _positionUpdated = false;
    }
}