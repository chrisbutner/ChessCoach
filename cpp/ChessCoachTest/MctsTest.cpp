#include <gtest/gtest.h>

#include <functional>

#include <ChessCoach/SelfPlay.h>
#include <ChessCoach/ChessCoach.h>

SelfPlayGame& PlayGame(SelfPlayWorker& selfPlayWorker, std::function<void (SelfPlayGame&)> tickCallback)
{
    const int index = 0;
    SelfPlayGame* game;
    SelfPlayState* state;
    float* values;
    INetwork::OutputPlanes* policies;

    selfPlayWorker.DebugGame(index, &game, &state, &values, &policies);

    selfPlayWorker.SetUpGame(index);

    while (true)
    {
        // CPU work
        selfPlayWorker.Play(index);

        if (*state == SelfPlayState::Finished)
        {
            return *game;
        }

        // "GPU" work. Pretend to predict for a batch.
        const int gameCount = 1;
        std::fill(values, values + gameCount, CHESSCOACH_VALUE_DRAW);

        INetwork::PlanesPointerFlat policiesPtr = reinterpret_cast<INetwork::PlanesPointerFlat>(policies);
        const int policyCount = (gameCount * INetwork::OutputPlanesFloatCount);
        std::fill(policiesPtr, policiesPtr + policyCount, 0.f);

        tickCallback(*game);
    }
}

std::vector<Node*> GeneratePrincipleVariation(const SelfPlayWorker& selfPlayWorker, const SelfPlayGame& game)
{
    Node* node = game.Root();
    std::vector<Node*> principleVariation;

    while (node)
    {
        for (const Node& child : *node)
        {
            if (child.visitCount > 0)
            {
                const bool bestIsNotBest = selfPlayWorker.WorseThan(node->bestChild, &child);
                if (bestIsNotBest) throw std::runtime_error("bestIsNotBest");
            }
        }
        if (node->bestChild)
        {
            principleVariation.push_back(node->bestChild);
        }
        node = node->bestChild;
    }

    return principleVariation;
}

void MockExpand(Node* node, int count)
{
    const float prior = (1.f / count);

    node->children = new Node[count]{};
    node->childCount = count;
    for (int i = 0; i < count; i++)
    {
        node->children[i].move = static_cast<uint16_t>(i);
        node->children[i].prior = prior;
    }
}

void CheckMateN(Node* node, int n)
{
    assert(n >= 1);

    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsImmediate(), (n == 1));
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).ImmediateValue(), (n == 1) ? CHESSCOACH_VALUE_WIN : CHESSCOACH_VALUE_DRAW);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsMateInN(), true);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsOpponentMateInN(), false);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).MateN(), n);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).OpponentMateN(), 0);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).EitherMateN(), n);
}

void CheckOpponentMateN(Node* node, int n)
{
    assert(n >= 1);

    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsImmediate(), false);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).ImmediateValue(), CHESSCOACH_VALUE_DRAW);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsMateInN(), false);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsOpponentMateInN(), true);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).MateN(), 0);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).OpponentMateN(), n);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).EitherMateN(), -n);
}

void CheckDraw(Node* node)
{
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsImmediate(), true);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).ImmediateValue(), CHESSCOACH_VALUE_DRAW);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsMateInN(), false);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsOpponentMateInN(), false);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).MateN(), 0);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).OpponentMateN(), 0);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).EitherMateN(), 0);
}

void CheckNonTerminal(Node* node)
{
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsImmediate(), false);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).ImmediateValue(), CHESSCOACH_VALUE_DRAW);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsMateInN(), false);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).IsOpponentMateInN(), false);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).MateN(), 0);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).OpponentMateN(), 0);
    EXPECT_EQ(node->terminalValue.load(std::memory_order_relaxed).EitherMateN(), 0);
}

TEST(Mcts, StateLeaks)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SearchState searchState{};
    SelfPlayWorker selfPlayWorker(nullptr /* storage */, &searchState, 1 /* gameCount */);

    // Allocations are only tracked with DEBUG.
#ifdef _DEBUG
    auto [currentBefore, peakBefore] = Game::StateAllocator.DebugAllocations();
#endif

    PlayGame(selfPlayWorker, [](auto&) {});

    // Clear the scratch game.
    selfPlayWorker.DebugResetGame(0);

    // Allocations are only tracked with DEBUG.
#ifdef _DEBUG
    auto [currentAfter, peakAfter] = Game::StateAllocator.DebugAllocations();
    EXPECT_EQ(currentAfter, currentBefore);
    EXPECT_GT(peakAfter, 0);
    EXPECT_GE(peakAfter, peakBefore);
#endif
}

TEST(Mcts, PrincipleVariation)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SearchState searchState{};
    SelfPlayWorker selfPlayWorker(nullptr /* storage */, &searchState, 1 /* gameCount */);

    int latestNodeCount = 0;
    std::vector<Node*> latestPrincipleVariation;
    PlayGame(selfPlayWorker, [&](SelfPlayGame& game)
        {
            std::vector<Node*> principleVariation = GeneratePrincipleVariation(selfPlayWorker, game);
            if (searchState.principleVariationChanged)
            {
                // If multiple nodes run, through terminal nodes or cache hits, the principle variation
                // may flip back to its previous value and look the same. Just test the single node case.
                if (searchState.nodeCount == (latestNodeCount + 1))
                {
                    EXPECT_NE(principleVariation, latestPrincipleVariation);
                }
                searchState.principleVariationChanged = false;
            }
            else
            {
                EXPECT_EQ(principleVariation, latestPrincipleVariation);
            }
            latestNodeCount = searchState.nodeCount;
            latestPrincipleVariation = principleVariation;
        });
}

TEST(Mcts, MateComparisons)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SearchState searchState{};
    SelfPlayWorker selfPlayWorker(nullptr /* storage */, &searchState, 1 /* gameCount */);

    // Set up nodes from expected worst to best.
    const int nodeCount = 7;
    Node nodes[nodeCount] = {};
    nodes[0].terminalValue = TerminalValue::OpponentMateIn<2>();
    nodes[1].terminalValue = TerminalValue::OpponentMateIn<4>();
    nodes[2].visitCount = 10;
    nodes[3].terminalValue = TerminalValue::Draw();
    nodes[3].visitCount = 15;
    nodes[4].visitCount = 100;
    nodes[5].terminalValue = TerminalValue::MateIn<3>();
    nodes[6].terminalValue = TerminalValue::MateIn<1>();

    // Check all pairs.
    for (int i = 0; i < nodeCount - 1; i++)
    {
        EXPECT_FALSE(selfPlayWorker.WorseThan(&nodes[i], &nodes[i]));

        EXPECT_TRUE(selfPlayWorker.WorseThan(nullptr, &nodes[i]));

        for (int j = i + 1; j < nodeCount; j++)
        {
            EXPECT_TRUE(selfPlayWorker.WorseThan(&nodes[i], &nodes[j]));
            EXPECT_FALSE(selfPlayWorker.WorseThan(&nodes[j], &nodes[i]));
        }
    }
}

TEST(Mcts, MateProving)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SearchState searchState{};
    SelfPlayWorker selfPlayWorker(nullptr /* storage */, &searchState, 1 /* gameCount */);
    SelfPlayGame* game;
    selfPlayWorker.SetUpGame(0);
    selfPlayWorker.DebugGame(0, &game, nullptr, nullptr, nullptr);

    // Expand a small tree (1 root, 3 ply1, 9 ply2).
    MockExpand(game->Root(), 3);
    MockExpand(game->Root()->Child(Move(0)), 3);
    MockExpand(game->Root()->Child(Move(1)), 3);
    MockExpand(game->Root()->Child(Move(2)), 3);

    // Selectively deepen two leaves.
    MockExpand(game->Root()->Child(Move(1))->Child(Move(1)), 1);
    MockExpand(game->Root()->Child(Move(1))->Child(Move(1))->Child(Move(0)), 1);
    MockExpand(game->Root()->Child(Move(2))->Child(Move(2)), 1);
    MockExpand(game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0)), 1);
    MockExpand(game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0))->Child(Move(0)), 1);
    MockExpand(game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0))->Child(Move(0))->Child(Move(0)), 1);

    // Expect that root and ply2child0 are non-terminal.
    CheckNonTerminal(game->Root());
    CheckNonTerminal(game->Root()->Child(Move(0))->Child(Move(0)));

    // Make ply2child0 a mate-in-1 (M1) and backpropagate.
    game->Root()->Child(Move(0))->Child(Move(0))->terminalValue = TerminalValue::MateIn<1>();
    selfPlayWorker.BackpropagateMate({
        { game->Root(), 1 },
        { game->Root()->Child(Move(0)), 1 },
        { game->Root()->Child(Move(0))->Child(Move(0)), 1 }});
    CheckMateN(game->Root()->Child(Move(0))->Child(Move(0)), 1);
    CheckOpponentMateN(game->Root()->Child(Move(0)), 1);
    CheckNonTerminal(game->Root());

    // Make ply2child1 a draw.
    game->Root()->Child(Move(0))->Child(Move(1))->terminalValue = TerminalValue::Draw();
    CheckDraw(game->Root()->Child(Move(0))->Child(Move(1)));

    // Make ply2child5 a mate-in-2 (M2) and backpropagate.
    game->Root()->Child(Move(1))->Child(Move(1))->Child(Move(0))->Child(Move(0))->terminalValue = TerminalValue::MateIn<1>();
    selfPlayWorker.BackpropagateMate({
        { game->Root(), 1 },
        { game->Root()->Child(Move(1)), 1 },
        { game->Root()->Child(Move(1))->Child(Move(1)), 1 },
        { game->Root()->Child(Move(1))->Child(Move(1))->Child(Move(0)), 1 },
        { game->Root()->Child(Move(1))->Child(Move(1))->Child(Move(0))->Child(Move(0)), 1 }});
    CheckMateN(game->Root()->Child(Move(1))->Child(Move(1))->Child(Move(0))->Child(Move(0)), 1);
    CheckOpponentMateN(game->Root()->Child(Move(1))->Child(Move(1))->Child(Move(0)), 1);
    CheckMateN(game->Root()->Child(Move(1))->Child(Move(1)), 2);
    CheckOpponentMateN(game->Root()->Child(Move(1)), 2);
    CheckNonTerminal(game->Root());

    // Make ply2child8 a mate-in-3 (M3) and backpropagate.
    // This should cause the root to get recognized as a mate-in-4 (M4).
    game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0))->Child(Move(0))->Child(Move(0))->Child(Move(0))->terminalValue = TerminalValue::MateIn<1>();
    selfPlayWorker.BackpropagateMate({
        { game->Root(), 1 },
        { game->Root()->Child(Move(2)), 1 },
        { game->Root()->Child(Move(2))->Child(Move(2)), 1 },
        { game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0)), 1 },
        { game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0))->Child(Move(0)), 1 },
        { game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0))->Child(Move(0))->Child(Move(0)), 1 },
        { game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0))->Child(Move(0))->Child(Move(0))->Child(Move(0)), 1 }});
    CheckMateN(game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0))->Child(Move(0))->Child(Move(0))->Child(Move(0)), 1);
    CheckOpponentMateN(game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0))->Child(Move(0))->Child(Move(0)), 1);
    CheckMateN(game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0))->Child(Move(0)), 2);
    CheckOpponentMateN(game->Root()->Child(Move(2))->Child(Move(2))->Child(Move(0)), 2);
    CheckMateN(game->Root()->Child(Move(2))->Child(Move(2)), 3);
    CheckOpponentMateN(game->Root()->Child(Move(2)), 3);
    CheckMateN(game->Root(), 4);

    game->PruneAll();
}

TEST(Mcts, TwofoldRepetition)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SearchState searchState{};
    SelfPlayWorker selfPlayWorker(nullptr /* storage */, &searchState, 1 /* gameCount */);
    SelfPlayGame* game;
    selfPlayWorker.SetUpGame(0);
    selfPlayWorker.DebugGame(0, &game, nullptr, nullptr, nullptr);

    // Set up a simple 2-repetition.
    std::vector<Move> moves{ make_move(SQ_E2, SQ_E4), make_move(SQ_D7, SQ_D6),
        make_move(SQ_D1, SQ_G4), make_move(SQ_G8, SQ_F6),
        make_move(SQ_G4, SQ_D1), make_move(SQ_F6, SQ_G8),
        make_move(SQ_D1, SQ_G4) };
    std::vector<Node*> nodes{};
    Node* node = game->Root();
    for (Move move : moves)
    {
        node->childCount = 1;
        node->children = new Node[1]{};
        node = &node->children[0];
        node->move = static_cast<uint16_t>(move);
        node->prior = 1.f;
        nodes.push_back(node);
    }

    // Apply the moves and evaluate the 2-repetition as a draw using the
    // starting position as the search root.
    {
        SelfPlayGame searchRoot = *game;
        for (int i = 0; i < moves.size(); i++)
        {
            searchRoot.ApplyMoveWithRoot(moves[i], nodes[i]);
        }

        SelfPlayState state = SelfPlayState::Working;
        PredictionCacheChunk* cacheStore = nullptr;
        const float value = searchRoot.ExpandAndEvaluate(state, cacheStore, &searchState, true /* isSearchRoot */);
        EXPECT_EQ(value, CHESSCOACH_VALUE_DRAW);
    }

    // Apply 6 moves, snap off a search root, then evaluate the final
    // position as a non-draw since it's not a 2-repetition past the
    // search root.
    {
        SelfPlayGame progress = *game;
        for (int i = 0; i < 6; i++)
        {
            progress.ApplyMoveWithRoot(moves[i], nodes[i]);
        }

        SelfPlayGame searchRoot = progress;
        for (int i = 6; i < moves.size(); i++)
        {
            searchRoot.ApplyMoveWithRoot(moves[i], nodes[i]);
        }

        SelfPlayState state = SelfPlayState::Working;
        PredictionCacheChunk* cacheStore = nullptr;
        const float value = searchRoot.ExpandAndEvaluate(state, cacheStore, &searchState, true /* isSearchRoot */);
        EXPECT_NE(value, CHESSCOACH_VALUE_DRAW);
        EXPECT_TRUE(std::isnan(value)); // A non-terminal position requires a network evaluation.
    }

    game->PruneAll();
}

TEST(Mcts, SamplingSelfPlay)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SearchState searchState{};
    SelfPlayWorker selfPlayWorker(nullptr /* storage */, &searchState, 1 /* gameCount */);
    SelfPlayGame* game;
    selfPlayWorker.SetUpGame(0, Config::StartingPosition, {}, false /* tryHard */); // SamplingSelfPlay means tryHard is false.
    selfPlayWorker.DebugGame(0, &game, nullptr, nullptr, nullptr);

    // Set up visit counts for four moves.
    game->Root()->childCount = 4;
    game->Root()->children = new Node[4]{};
    game->Root()->children[0].move = static_cast<uint16_t>(make_move(SQ_E2, SQ_E4));
    game->Root()->children[0].prior = 0.25f;
    game->Root()->children[0].visitCount = 350;
    game->Root()->children[1].move = static_cast<uint16_t>(make_move(SQ_D2, SQ_D4));
    game->Root()->children[1].prior = 0.25f;
    game->Root()->children[1].visitCount = 250;
    game->Root()->children[2].move = static_cast<uint16_t>(make_move(SQ_E2, SQ_E3));
    game->Root()->children[2].prior = 0.25f;
    game->Root()->children[2].visitCount = 150;
    game->Root()->children[3].move = static_cast<uint16_t>(make_move(SQ_C2, SQ_C4));
    game->Root()->children[3].prior = 0.25f;
    game->Root()->children[3].visitCount = 50;
    game->Root()->visitCount = (game->Root()->children[0].visitCount + game->Root()->children[1].visitCount + game->Root()->children[2].visitCount + game->Root()->children[3].visitCount);
    game->Root()->bestChild = &game->Root()->children[0];

    // Validate self-play sampling. Just shove samples in "valueWeight".
    // Expect visits in proportion to visit count.
    const int epsilon = 50;
    for (int i = 0; i < game->Root()->visitCount; i++)
    {
        Node* selected = selfPlayWorker.SelectMove(*game, true /* allowDiversity */);
        selected->valueWeight++;
    }
    EXPECT_NEAR(game->Root()->children[0].valueWeight, game->Root()->children[0].visitCount, epsilon);
    EXPECT_NEAR(game->Root()->children[1].valueWeight, game->Root()->children[1].visitCount, epsilon);
    EXPECT_NEAR(game->Root()->children[2].valueWeight, game->Root()->children[2].visitCount, epsilon);
    EXPECT_NEAR(game->Root()->children[3].valueWeight, game->Root()->children[3].visitCount, epsilon);

    game->PruneAll();
}

TEST(Mcts, SamplingUci)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SearchState searchState{};
    SelfPlayWorker selfPlayWorker(nullptr /* storage */, &searchState, 1 /* gameCount */);
    SelfPlayGame* game;
    selfPlayWorker.SetUpGame(0, Config::StartingPosition, {}, true /* tryHard */); // SamplingUci means tryHard is true.
    selfPlayWorker.DebugGame(0, &game, nullptr, nullptr, nullptr);

    // Set up visit counts for four moves.
    game->Root()->childCount = 4;
    game->Root()->children = new Node[4]{};
    game->Root()->children[0].move = static_cast<uint16_t>(make_move(SQ_E2, SQ_E4));
    game->Root()->children[0].prior = 0.25f;
    game->Root()->children[0].visitCount = 350;
    game->Root()->children[1].move = static_cast<uint16_t>(make_move(SQ_D2, SQ_D4));
    game->Root()->children[1].prior = 0.25f;
    game->Root()->children[1].visitCount = 250;
    game->Root()->children[2].move = static_cast<uint16_t>(make_move(SQ_E2, SQ_E3));
    game->Root()->children[2].prior = 0.25f;
    game->Root()->children[2].visitCount = 150;
    game->Root()->children[3].move = static_cast<uint16_t>(make_move(SQ_C2, SQ_C4));
    game->Root()->children[3].prior = 0.25f;
    game->Root()->children[3].visitCount = 50;
    game->Root()->visitCount = (game->Root()->children[0].visitCount + game->Root()->children[1].visitCount + game->Root()->children[2].visitCount + game->Root()->children[3].visitCount);
    game->Root()->bestChild = &game->Root()->children[0];

    // Back up temperature.
    const float temperatureBackup = Config::Network.SelfPlay.MoveDiversityTemperature;
    Config::Network.SelfPlay.MoveDiversityTemperature = 0.75f;

    // Validate UCI sampling. Just shove samples in "valueWeight".
    // Expect visits in proportion to visit counts re-exponentiated with temperature 0.75.
    const int sampleCount = 100000;
    const float epsilon = 0.005f;
    for (int i = 0; i < sampleCount; i++)
    {
        Node* selected = selfPlayWorker.SelectMove(*game, true /* allowDiversity */);
        selected->valueWeight++;
    }
    EXPECT_NEAR(static_cast<float>(game->Root()->children[0].valueWeight) / sampleCount, 0.4911f, epsilon);
    EXPECT_NEAR(static_cast<float>(game->Root()->children[1].valueWeight) / sampleCount, 0.3136f, epsilon);
    EXPECT_NEAR(static_cast<float>(game->Root()->children[2].valueWeight) / sampleCount, 0.1587f, epsilon);
    EXPECT_NEAR(static_cast<float>(game->Root()->children[3].valueWeight) / sampleCount, 0.0367f, epsilon);

    // Restore temperature.
    Config::Network.SelfPlay.MoveDiversityTemperature = temperatureBackup;

    game->PruneAll();
}

TEST(Mcts, PrepareExpandedRoot)
{
    ChessCoach chessCoach;
    chessCoach.Initialize();

    SearchState searchState{};
    SelfPlayWorker selfPlayWorker(nullptr /* storage */, &searchState, 1 /* gameCount */);
    
    const int index = 0;
    SelfPlayGame* game;
    SelfPlayState* state;
    float* values;
    INetwork::OutputPlanes* policies;

    selfPlayWorker.DebugGame(index, &game, &state, &values, &policies);
    selfPlayWorker.SetUpGame(index);

    // "GPU" work, in advance.
    const int gameCount = 1;
    std::fill(values, values + gameCount, CHESSCOACH_VALUE_DRAW);
    INetwork::PlanesPointerFlat policiesPtr = reinterpret_cast<INetwork::PlanesPointerFlat>(policies);
    const int policyCount = (gameCount * INetwork::OutputPlanesFloatCount);
    std::fill(policiesPtr, policiesPtr + policyCount, 0.f);

    bool coverageA = false;
    bool coverageB = false;
    bool coverageC = false;
    int lastPly = 0;
    while ((*state != SelfPlayState::Finished) && (lastPly < 10))
    {
        // CPU work
        selfPlayWorker.Play(index);

        if ((game->Ply() > lastPly) && game->Root()->IsExpanded())
        {
            // Expect exploration noise at the root.
            if (game->Root()->childCount >= 2)
            {
                EXPECT_NE(game->Root()->children[0].prior, game->Root()->children[1].prior);
                coverageA = true;
            }

            for (const Node& child : *game->Root())
            {
                // Expect win-FPU for children of the root. We overwrite any default or draw-sibling-FPU.
                EXPECT_TRUE(
                    ((child.visitCount > 0) && (child.valueAverage == CHESSCOACH_VALUE_DRAW)) || // Backprops
                    ((child.visitCount == 0) && (child.valueAverage == CHESSCOACH_FIRST_PLAY_URGENCY_ROOT)) // Win-FPU
                    );
                coverageB = true;
                
                if (child.IsExpanded() && (child.childCount >= 2))
                {
                    // No win-FPU or exploration noise deeper in the tree.
                    EXPECT_FALSE((child.children[0].visitCount == 0) && (child.children[0].valueAverage == CHESSCOACH_FIRST_PLAY_URGENCY_ROOT));
                    EXPECT_EQ(child.children[0].prior, child.children[1].prior);
                    coverageC = true;
                }
            }

            lastPly = game->Ply();
        }
    }

    EXPECT_TRUE(coverageA);
    EXPECT_TRUE(coverageB);
    EXPECT_TRUE(coverageC);
}