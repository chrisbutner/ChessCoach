// ChessCoach, a neural network-based chess engine capable of natural-language commentary
// Copyright 2021 Chris Butner
//
// ChessCoach is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ChessCoach is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ChessCoach. If not, see <https://www.gnu.org/licenses/>.

#ifndef _EPD_H_
#define _EPD_H_

#include <iostream>
#include <vector>
#include <filesystem>

struct StrengthTestSpec
{
    std::string fen;
    std::vector<std::string> pointSans;
    std::vector<int> points;
    std::vector<std::string> avoidSans;
};


class Epd
{
public:

    static std::vector<StrengthTestSpec> ParseEpds(const std::filesystem::path& path);

private:

    static StrengthTestSpec ParseEpd(const std::string& epd);
    static std::vector<std::string> ReadMoves(std::istream& tokenizer, std::string& token);
    static void Expect(std::istream& tokenizer, const unsigned char expected);
    static bool Find(std::istream& tokenizer, std::string& token, const std::string& find);
    static std::string StripLeft(const std::string& string, char expected);
    static std::string StripRight(const std::string& string, char expected);
};

#endif // _EPD_H_