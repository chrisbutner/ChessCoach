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

#ifndef _PREPROCESSING_H_
#define _PREPROCESSING_H_

#include <string>
#include <memory>

#define HUNSPELL_STATIC
#include <hunspell/hunspell.hxx>

class Preprocessor
{
public:

    Preprocessor();

    void PreprocessComment(std::string& comment) const;

public:

    static void Trim(std::string& text);

private:

    static constexpr const char EnglishPunctuationPlusSlash[] = ".,;:-?!'\"()[]{}/";

private:

    bool WordTokenize(std::istream& content, std::string& tokenOut) const;
    bool IsBadComment(const std::string& comment) const;
    bool IsSufficientlyEnglish(const std::string& comment) const;
    bool IsNotEnglish(const std::string& token) const;
    bool Unwrap(std::string& token, char left, char right) const;
    void StripStartingNoise(std::string& comment) const;

private:

    std::unique_ptr<Hunspell> _hunspell;
};

#endif // _PREPROCESSING_H_