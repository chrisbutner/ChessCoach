/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (c) 2013 Ronald de Man
  Copyright (C) 2016-2020 Marco Costalba, Lucas Braesch

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Based on Stockfish tbprobe.cpp, adapted for ChessCoach
// Modifications by Chris Butner, 2021

#include "Syzygy.h"

#include <limits>
#include <cmath>
#include <limits>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>
#include <iomanip>

#include <Stockfish/thread.h>
#include <Stockfish/uci.h>
#include <Stockfish/syzygy/tbprobe.h>

#include "SelfPlay.h"
#include "Storage.h"

void Syzygy::Reload()
{
    Tablebases::init(Storage::MakeLocalPath(Config::Misc.Paths_Syzygy).string());
}

bool Syzygy::ProbeTablebasesAtRoot(SelfPlayGame& game)
{
    bool RootInTB = false;
    bool dtz_available = true;
    bool attemptedProbe = false;
    const Position& position = game.GetPosition();
    game.TablebaseCardinality() = Tablebases::MaxCardinality;

    // "ShouldProbeTablebases" may be non-deterministic for self-play games.
    // Decide once for each position (here) whether to use tablebases at all, and if not,
    // clear cardinality so that deeper positions don't probe via "ProbeWdl".
    if (!game.ShouldProbeTablebases())
    {
        game.TablebaseCardinality() = 0;
    }

    if (game.TablebaseCardinality() &&
        (game.TablebaseCardinality() >= position.count<ALL_PIECES>()) &&
        !position.can_castle(ANY_CASTLING))
    {
        // Rank moves using DTZ tables
        attemptedProbe = true;
        RootInTB = ProbeDtzAtRoot(game);

        if (!RootInTB)
        {
            // DTZ tables are missing; try to rank moves using WDL tables
            dtz_available = false;
            RootInTB = ProbeWdlAtRoot(game);
        }
    }

    if (RootInTB)
    {
        // Probe during search only if DTZ is not available.
        if (dtz_available)
        {
            game.TablebaseCardinality() = 0;
        }
    }
    else if (attemptedProbe)
    {
        // Data is too intermingled. Just quit if ProbeDtzAtRoot() and ProbeWdlAtRoot() fail.
        throw ChessCoachException("Failed to probe tablebases at the search root");
    }
    
    return RootInTB;
}

int dtz_before_zeroing(Tablebases::WDLScore wdl) {
    return wdl == Tablebases::WDLWin ? 1 :
        wdl == Tablebases::WDLCursedWin ? 101 :
        wdl == Tablebases::WDLBlessedLoss ? -101 :
        wdl == Tablebases::WDLLoss ? -1 : 0;
}

// Use the DTZ tables to rank root moves.
//
// A return value false indicates that not all probes were successful.
bool Syzygy::ProbeDtzAtRoot(SelfPlayGame& game)
{
    Tablebases::ProbeState result;
    StateInfo stateInfo;
    Position& position = game.GetPosition();

    // Obtain 50-move counter for the root position
    const int rootNoProgressCount = position.rule50_count();

    // Check whether a position was repeated since the last zeroing move.
    const bool hasRepeated = position.has_repeated();

    // Always use 50-move rule.
    const int bound = 900;

    // Probe and rank each move
    for (Node& child : *game.Root())
    {
        const Move move = Move(child.move);
        position.do_move(move, stateInfo);

        // Calculate dtz for the current move counting from the root position
        int dtz;
        if (position.rule50_count() == 0)
        {
            // In case of a zeroing move, dtz is one of -101/-1/0/1/101
            Tablebases::WDLScore wdl = Tablebases::WDLScore(-Tablebases::probe_wdl(position, &result));
            dtz = dtz_before_zeroing(wdl);
        }
        else
        {
            // Otherwise, take dtz for the new position and correct by 1 ply
            dtz = -Tablebases::probe_dtz(position, &result);
            dtz = dtz > 0 ? dtz + 1
                : dtz < 0 ? dtz - 1 : dtz;
        }

        // Make sure that a mating move is assigned a dtz value of 1
        if (position.checkers()
            && dtz == 2
            && MoveList<LEGAL>(position).size() == 0)
        {
            dtz = 1;
        }

        position.undo_move(move);

        if (result == Tablebases::ProbeState::FAIL)
        {
            return false;
        }

        // Better moves are ranked higher. Certain wins are ranked equally.
        // Losing moves are ranked equally unless a 50-move draw is in sight.
        int r = dtz > 0 ? (dtz + rootNoProgressCount <= 99 && !hasRepeated ? 1000 : 1000 - (dtz + rootNoProgressCount))
            : dtz < 0 ? (-dtz * 2 + rootNoProgressCount < 100 ? -1000 : -1000 + (-dtz + rootNoProgressCount))
            : 0;

        const Bound tablebaseBound =
            r >= bound ? BOUND_LOWER
            : r <= -bound ? BOUND_UPPER
            : BOUND_EXACT;
        
        // Sacrifice cursed win/blessed loss differentiation. This can be added back using signal bits in "tablebaseRankBound"
        // if needed.
        child.SetTablebaseRankBound(r, tablebaseBound);
    }

    return true;
}

// Use the WDL tables to rank root moves.
// This is a fallback for the case that some or all DTZ tables are missing.
//
// A return value false indicates that not all probes were successful.
bool Syzygy::ProbeWdlAtRoot(SelfPlayGame& game)
{
    constexpr int WDL_to_rank[] = { -1000, -899, 0, 899, 1000 };

    Position& position = game.GetPosition();
    Tablebases::ProbeState result;
    StateInfo stateInfo;

    // Probe and rank each move
    for (Node& child : *game.Root())
    {
        const Move move = Move(child.move);
        position.do_move(move, stateInfo);

        Tablebases::WDLScore wdl = Tablebases::WDLScore(-Tablebases::probe_wdl(position, &result));

        position.undo_move(move);

        if (result == Tablebases::ProbeState::FAIL)
        {
            return false;
        }

        const Bound tablebaseBound =
            wdl > 1 ? BOUND_LOWER
            : wdl < -1 ? BOUND_UPPER
            : BOUND_EXACT;

        // Sacrifice cursed win/blessed loss differentiation. This can be added back using signal bits in "tablebaseRankBound"
        // if needed.
        child.SetTablebaseRankBound(WDL_to_rank[wdl + 2], tablebaseBound);
    }

    return true;
}

bool Syzygy::ProbeWdl(SelfPlayGame& game, bool isSearchRoot)
{
    Position& position = game.GetPosition();

    // In the specific case where we have WDL tables, but DTZ tables are missing, and one of the immediate children
    // of the search root is a zeroing move, we may proceed to probe WDL here despite having already set score/bound
    // on this node during the root probe. Since this will happen so rarely, and only cause a small number of duplicate
    // probes at the start of search, with no harm done, don't waste any runtime checking for the case.
    //
    // No need to check "SelfPlayGame::ShouldProbeTablebases" here: see in "ProbeTablebasesAtRoot".
    if (!(!isSearchRoot &&
        game.TablebaseCardinality() &&
        game.TablebaseCardinality() >= position.count<ALL_PIECES>() &&
        (position.rule50_count() == 0) &&
        !position.can_castle(ANY_CASTLING)))
    {
        return false;
    }

    // Always value from parent's perspective.
    Tablebases::ProbeState result;
    Tablebases::WDLScore wdl = Tablebases::WDLScore(-Tablebases::probe_wdl(position, &result));
    if (result == Tablebases::ProbeState::FAIL)
    {
        return false;
    }

    // Always use 50-move rule.
    const int drawScore = 1;

    const Bound tablebaseBound =
        wdl < -drawScore ? BOUND_UPPER
        : wdl >  drawScore ? BOUND_LOWER
        : BOUND_EXACT;

    // Sacrifice cursed win/blessed loss differentiation. This can be added back using signal bits in "tablebaseRankBound"
    // if needed.
    game.Root()->SetTablebaseRankBound(game.Root()->TablebaseRank(), tablebaseBound);
    return true;
}