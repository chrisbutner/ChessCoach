#ifndef _CHESSCOACH_H_
#define _CHESSCOACH_H_

#include "Network.h"

class Storage;
class SelfPlayWorker;

class ChessCoach
{
public:

    void PrintExceptions();
    void Initialize();
    void Finalize();

    INetwork* CreateNetwork(const NetworkConfig& networkConfig) const;

protected:

    void InitializePython();
    void InitializeStockfish();
    void InitializeChessCoach();
    void InitializePredictionCache();
    void InitializePythonModule(Storage* storage, SelfPlayWorker* worker, INetwork* network);

    void FinalizePython();
    void FinalizeStockfish();
};

#endif // _CHESSCOACH_H_