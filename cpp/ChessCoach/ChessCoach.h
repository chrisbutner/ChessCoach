#include "Network.h"

class ChessCoach
{
public:

    void PrintExceptions();
    void Initialize();
    void Finalize();

    INetwork* CreateNetwork(const NetworkConfig& networkConfig) const;

protected:

    int InitializePython();
    void InitializeStockfish();
    void InitializeChessCoach();
    void InitializePredictionCache();

    void FinalizePython();
    void FinalizeStockfish();
};