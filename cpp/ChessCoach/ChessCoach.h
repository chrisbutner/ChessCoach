#include "Network.h"

class ChessCoach
{
public:

    void Initialize();
    void Finalize();

    INetwork* CreateNetwork() const;

private:

    int InitializePython();
    void InitializeStockfish();
    void InitializeChessCoach();

    void FinalizePython();
    void FinalizeStockfish();
};