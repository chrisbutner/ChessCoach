/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (c) 2013 Ronald de Man
  Copyright (C) 2016-2020 Marco Costalba, Lucas Braesch
  Copyright (C) 2021 Chris Butner

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

// Based on Stockfish tbprobe.h, adapted for ChessCoach.

#ifndef _SYZYGY_H_
#define _SYZYGY_H_

class SelfPlayGame;
struct Node;

class Syzygy
{
public:

    static void Reload();
    static bool ProbeTablebasesAtRoot(SelfPlayGame& game);
    static bool ProbeDtzAtRoot(SelfPlayGame& game);
    static bool ProbeWdlAtRoot(SelfPlayGame& game);
    static bool ProbeWdl(SelfPlayGame& game, bool isSearchRoot);

private:

    static void UpdateRootChildValue(Node* node);
};

#endif // _SYZYGY_H_