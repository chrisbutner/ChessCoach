#include <functional>
#include <fstream>

#include <Stockfish/position.h>
#include <tclap/CmdLine.h>

#include <ChessCoach/ChessCoach.h>
#include <ChessCoach/Storage.h>
#include <ChessCoach/Pgn.h>

class ChessCoachPgnToGames : public ChessCoach
{
public:

    ChessCoachPgnToGames(const std::filesystem::path& inputDirectory, const std::filesystem::path& outputDirectory);

    void InitializeLight();
    void FinalizeLight();
    void ConvertAll();

private:

    std::filesystem::path _inputDirectory;
    std::filesystem::path _outputDirectory;
};

int main(int argc, char* argv[])
{
    std::string inputDirectory;
    std::string outputDirectory;

    try
    {
        TCLAP::CmdLine cmd("ChessCoachPgnToGames: Converts PGN databases to games to use in training and testing ChessCoach", ' ', "0.9");

        TCLAP::ValueArg<std::string> inputDirectoryArg("i", "input", "Input directory where PGN files are located", true /* req */, "", "string");
        TCLAP::ValueArg<std::string> outputDirectoryArg("o", "output", "Output directory where game files should be placed", true /* req */, "", "string");

        // Usage/help seems to reverse this order.
        cmd.add(outputDirectoryArg);
        cmd.add(inputDirectoryArg);

        cmd.parse(argc, argv);

        inputDirectory = inputDirectoryArg.getValue();
        outputDirectory = outputDirectoryArg.getValue();
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    }

    ChessCoachPgnToGames pgnToGames(inputDirectory, outputDirectory);

    pgnToGames.InitializeLight();

    pgnToGames.ConvertAll();

    pgnToGames.FinalizeLight();

    return 0;
}

ChessCoachPgnToGames::ChessCoachPgnToGames(const std::filesystem::path& inputDirectory, const std::filesystem::path& outputDirectory)
    : _inputDirectory(inputDirectory)
    , _outputDirectory(outputDirectory)
{
}

void ChessCoachPgnToGames::InitializeLight()
{
    InitializeStockfish();
    InitializeChessCoach();
}

void ChessCoachPgnToGames::FinalizeLight()
{
    FinalizeStockfish();
}

void ChessCoachPgnToGames::ConvertAll()
{
    int latestGameNumber = 0;

    for (const auto& entry : std::filesystem::directory_iterator(_inputDirectory))
    {
        if (entry.path().extension().string() == ".pgn")
        {
            std::ifstream pgnFile = std::ifstream(entry.path(), std::ios::in);
         
            std::cout << "Converting " << entry.path().filename() << ": ";
            int gamesConverted = 0;

            Pgn::ParsePgn(pgnFile, [&](const SavedGame& game)
                {
                    const std::filesystem::path gamePath = (_outputDirectory / Storage::GenerateGameName(++latestGameNumber));

                    Storage::SaveToDisk(gamePath, game);
                    gamesConverted++;
                });

            std::cout << gamesConverted << " games" << std::endl;
        }
    }
    std::cout << "Finished" << std::endl;
}