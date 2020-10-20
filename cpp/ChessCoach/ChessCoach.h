#ifndef _CHESSCOACH_H_
#define _CHESSCOACH_H_

#include "Network.h"

class ChessCoach
{
public:

    void PrintExceptions();
    void Initialize();
    void Finalize();

    INetwork* CreateNetwork(const NetworkConfig& networkConfig) const;
    INetwork* CreateNetworkWithInfo(const NetworkConfig& networkConfig, int& stepCountOut) const;

protected:

    void InitializePython();
    void InitializeStockfish();
    void InitializeChessCoach();
    void InitializePredictionCache();

    void FinalizePython();
    void FinalizeStockfish();
};

#endif // _CHESSCOACH_H_