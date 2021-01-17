#include <filesystem>

#include <tclap/CmdLine.h>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/SelfPlay.h>

class ChessCoachStrengthTest : public ChessCoach
{
public:

    ChessCoachStrengthTest(NetworkType networkType, const std::filesystem::path& epdPath, int moveTimeMs, float slopeArg, float interceptArg);

    void Initialize();

    void StrengthTest();

private:

    NetworkType _networkType;
    std::filesystem::path _epdPath;
    int _moveTimeMs;
    float _slope;
    float _intercept;
};

int main(int argc, char* argv[])
{
    std::string network;
    NetworkType networkType = NetworkType_Count;
    std::string epdPath;
    int moveTimeMs;
    float slope;
    float intercept;

    try
    {
        TCLAP::CmdLine cmd("ChessCoachStrengthTest: Tests ChessCoach using a provided .epd file to generate a score and optionally a rating", ' ', "0.9");

        TCLAP::ValueArg<std::string> networkArg("n", "network", "Network to test, teacher or student", false /* req */, "student", "string");
        TCLAP::ValueArg<std::string> epdArg("e", "epd", "Path to the .epd file to test", true /* req */, "", "string");
        TCLAP::ValueArg<int> moveTimeArg("t", "movetime", "Move time per position (ms)", true /* req */, 0, "whole number");
        TCLAP::ValueArg<float> slopeArg("s", "slope", "Slope for linear rating calculation based on score", false /* req */, 0.f, "decimal");
        TCLAP::ValueArg<float> interceptArg("i", "intercept", "Intercept for linear rating calculation based on score", false /* req */, 0.f, "decimal");

        // Usage/help seems to reverse this order.
        cmd.add(interceptArg);
        cmd.add(slopeArg);
        cmd.add(moveTimeArg);
        cmd.add(epdArg);
        cmd.add(networkArg);

        cmd.parse(argc, argv);

        network = networkArg.getValue();
        if (network == "teacher")
        {
            networkType = NetworkType_Teacher;
        }
        else if (network == "student")
        {
            networkType = NetworkType_Student;
        }
        else
        {
            std::cerr << "Expected 'teacher' or 'student' for 'network argument" << std::endl;
            return 1;
        }

        epdPath = epdArg.getValue();
        moveTimeMs = moveTimeArg.getValue();
        slope = slopeArg.getValue();
        intercept = interceptArg.getValue();
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
        return 1;
    }

    ChessCoachStrengthTest strengthTest(networkType, epdPath, moveTimeMs, slope, intercept);

    strengthTest.PrintExceptions();
    strengthTest.Initialize();

    strengthTest.StrengthTest();

    strengthTest.Finalize();

    return 0;
}

ChessCoachStrengthTest::ChessCoachStrengthTest(NetworkType networkType, const std::filesystem::path& epdPath, int moveTimeMs, float slope, float intercept)
    : _networkType(networkType)
    , _epdPath(epdPath)
    , _moveTimeMs(moveTimeMs)
    , _slope(slope)
    , _intercept(intercept)
{
}

void ChessCoachStrengthTest::Initialize()
{
    // Suppress all Python/TensorFlow output so that output is readable, especially when running
    // multiple strength tests back-to-back.
    Platform::SetEnvironmentVariable("CHESSCOACH_SILENT", "1");

    InitializePython();
    InitializeStockfish();
    InitializeChessCoach();

    PredictionCache::Instance.Allocate(Config::Misc.PredictionCache_RequestGibibytes, Config::Misc.PredictionCache_MinGibibytes);
}

static void PrintProgress(const std::string& fen, const std::string& target, const std::string& chosen, int score, int total)
{
    std::cout << fen << ", " << target << ", " << chosen << ", " << score << ", " << total << std::endl;
}

void ChessCoachStrengthTest::StrengthTest()
{
    std::cout << "Preparing network..." << std::endl;

    // Use the UCI network for standalone strength tests.
    std::unique_ptr<INetwork> network(CreateNetwork(Config::UciNetwork));
    SelfPlayWorker worker(Config::UciNetwork, nullptr /* storage */);

    std::cout << "Testing " << _epdPath.stem() << "...\n\nPosition, Target, Chosen, Score, Total" << std::endl;

    const auto start = std::chrono::high_resolution_clock::now();

    const auto [score, total, positions, totalNodesRequired] = worker.StrengthTestEpd(network.get(), _networkType, _epdPath,
        _moveTimeMs, 0 /* nodes */, 0 /* failureNodes */, 0 /* positionLimit */, PrintProgress);

    const float secondsExpected = std::chrono::duration<float>(std::chrono::duration<float, std::milli>(_moveTimeMs * positions)).count();
    const float secondsTaken = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();

    std::cout << "\nTested " << positions << " positions in " << secondsTaken << " seconds, expected " << secondsExpected << " seconds." << std::endl;
    std::cout << "Score: " << score << " out of " << total << std::endl;

    // Use score/positions (not score/total) with slope and intercept to match STS.
    if ((_slope != 0.f) || (_intercept != 0.f))
    {
        const int rating = static_cast<int>((_slope * score / positions) + _intercept);
        std::cout << "Rating: " << rating << std::endl;
    }
}