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

    INetwork* CreateNetwork() const;

protected:

    void InitializePython();
    void InitializeStockfish();
    void InitializeChessCoach();
    void InitializePredictionCache();
    void InitializePythonModule(Storage* storage, INetwork* network, SelfPlayWorker* worker);

    void FinalizePython();
    void FinalizeStockfish();

    void OptimizeParameters();
};

#endif // _CHESSCOACH_H_