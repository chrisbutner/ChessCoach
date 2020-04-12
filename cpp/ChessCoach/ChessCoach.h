#include "Network.h"

class ChessCoach
{
public:

    void Initialize();
    void Finalize();

    INetwork* CreateNetwork() const;

protected:

    int InitializePython();
    void InitializeStockfish();
    void InitializeChessCoach();

    void FinalizePython();
    void FinalizeStockfish();
};