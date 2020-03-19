//#include "Mcts.h"
//
//Node::Node(float prior)
//{
//
//}
//
//bool Game::IsTerminal()
//{
//    return blah || (History.size() >= Config.MaxMoves);
//}
//
//// TODO: Write a custom allocator for nodes (work out very maximum, then do some kind of ring/tree - important thing is all same size, capped number)
//
//void Game::Play()
//{
//    Game game;
//
//    _root = new Node(0.f);
//    Evaluate();
//
//    while (!game.IsTerminal())
//    {
//        Action action = RunMcts();
//        Apply(action);
//        StoreSearchStatistics();
//    }
//}
//
//void Game::RunMcts()
//{
//    PrepareSubtree();
//    AddExplorationNoise();
//
//    for (int i = 0; i < _config.NumSimulations; i++)
//    {
//
//    }
//}
//
//void Game::Evaluate()
//{
//
//}