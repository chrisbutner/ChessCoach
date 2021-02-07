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

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/WorkerGroup.h>
#include <ChessCoach/Pgn.h>

typedef std::function<void(std::stringstream&)> CommandHandler;
typedef std::pair<std::string, CommandHandler> CommandHandlerEntry;

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

    void InitializeWorkers();
    void StopAndReadyWorkers();

private:

    bool _quit = false;
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
        throw std::invalid_argument("Band width (search_threads * search_parallelism) >= 256 required for sufficient exploration");
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

    const std::filesystem::path commandLogPath = (Storage().LocalLogPath() / commandLogFilename.str());
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
    std::cout << "id name ChessCoach\n"
        "id author C. Butner\n"
        // Options are listed here when they exist.
        "uciok" << std::endl;
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
    }

    std::cout << "readyok" << std::endl;
}

void ChessCoachUci::HandleSetOption(std::stringstream& /*commands*/)
{
}

void ChessCoachUci::HandleRegister(std::stringstream& /*commands*/)
{
}

void ChessCoachUci::HandleUciNewGame(std::stringstream& /*commands*/)
{
    InitializeWorkers();
}

void ChessCoachUci::HandlePosition(std::stringstream& commands)
{
    std::string token;
    std::string fen;
    std::vector<Move> moves;
    StateListPtr positionStates(new std::deque<StateInfo>(1));
    Position position;

    commands >> token;
    if (token == "fen")
    {
        // Trust the provided FEN.
        fen.reserve(128);
        while ((commands >> token) && (token != "moves"))
        {
            fen += token + " ";
        }
    }
    else
    {
        // Instead of "fen" we got "startpos" or something invalid.
        fen = Config::StartingPosition;
        while ((commands >> token) && (token != "moves"))
        {
        }
    }

    position.set(fen, false /* isChess960 */, &positionStates->back(), Threads.main());

    // If "moves" wasn't seen then we already consumed the rest of the line.
    while (commands >> token)
    {
        Move move = UCI::to_move(position, token);
        if (move == MOVE_NONE)
        {
            break;
        }

        position.do_move(move, positionStates->emplace_back());

        moves.push_back(move);
    }

    InitializeWorkers();
    StopAndReadyWorkers();
    _workerGroup.controllerWorker->SearchUpdatePosition(std::move(fen), std::move(moves), false /* forceNewPosition */);
}

void ChessCoachUci::HandleGo(std::stringstream& commands)
{
    // TODO: searchmoves
    // TODO: movestogo

    TimeControl timeControl = {};

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
    }

    InitializeWorkers();
    StopAndReadyWorkers();

    // Use the starting position if none ever specified.
    if (!_workerGroup.searchState.position)
    {
        _workerGroup.controllerWorker->SearchUpdatePosition(Config::StartingPosition, {}, false /* forceNewPosition */);
    }

    _workerGroup.searchState.Reset(timeControl);

    PredictionCache::Instance.ResetProbeMetrics();

    _workerGroup.workCoordinator->ResetWorkItemsRemaining(1);
}

void ChessCoachUci::HandleStop(std::stringstream& /*commands*/)
{
    InitializeWorkers();
    StopAndReadyWorkers();
}

void ChessCoachUci::HandleQuit(std::stringstream& /*commands*/)
{
    _quit = true;
}

void ChessCoachUci::HandleComment(std::stringstream& /*commands*/)
{
    InitializeWorkers();
    StopAndReadyWorkers();
    _workerGroup.controllerWorker->CommentOnPosition(_network.get());
}

void ChessCoachUci::HandleGui(std::stringstream& /*commands*/)
{
    InitializeWorkers();
    if (!_workerGroup.searchState.gui)
    {
        _workerGroup.searchState.gui = true;
        _network->LaunchGui("push");
    }
}

void ChessCoachUci::HandleConsole(std::stringstream& commands)
{
    std::string token;
    if (!(commands >> token))
    {
        return;
    }

    if (token == "ucb")
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
        SelfPlayGame ucbGame = *game;

        // If "moves" wasn't seen then we already consumed the rest of the line.
        while (commands >> token)
        {
            Move move = UCI::to_move(ucbGame.GetPosition(), token);
            if (move == MOVE_NONE)
            {
                break;
            }

            ucbGame.ApplyMoveWithRoot(move, ucbGame.Root()->Child(move));
        }

        if (csv)
        {
            std::cout << "move,prior,value,ucb,visits,weight" << std::endl;
        }

        Node* root = ucbGame.Root();
        for (const Node& child : *root)
        {
            if (csv)
            {
                std::cout << Pgn::San(ucbGame.GetPosition(), Move(child.move), true /* showCheckmate */)
                    << "," << child.prior
                    << "," << child.Value()
                    << "," << _workerGroup.controllerWorker->CalculatePuctScore<false>(root, &child).first
                    << "," << child.visitCount
                    << "," << child.valueWeight
                    << std::endl;
            }
            else
            {
                std::cout << Pgn::San(ucbGame.GetPosition(), Move(child.move), true /* showCheckmate */)
                    << " prior=" << child.prior
                    << " value=" << child.Value()
                    << " ucb=" << _workerGroup.controllerWorker->CalculatePuctScore<false>(root, &child).first
                    << " visits=" << child.visitCount
                    << " weight=" << child.valueWeight
                    << std::endl;
            }
        }
    }
}

void ChessCoachUci::InitializeWorkers()
{
    if (_workerGroup.IsInitialized())
    {
        return;
    }

    // Use the faster student network for UCI.
    _network.reset(CreateNetwork());
    _workerGroup.Initialize(_network.get(), NetworkType_Student, Config::Misc.Search_SearchThreads, Config::Misc.Search_SearchParallelism, &SelfPlayWorker::LoopSearch);
}

void ChessCoachUci::StopAndReadyWorkers()
{
    _workerGroup.workCoordinator->ResetWorkItemsRemaining(0);
    _workerGroup.workCoordinator->WaitForWorkers();
}