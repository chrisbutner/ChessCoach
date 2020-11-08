#include "Epd.h"

#include <fstream>
#include <sstream>
#include <cassert>

std::vector<StrengthTestSpec> Epd::ParseEpds(const std::filesystem::path& path)
{
    std::ifstream epdFile(path);
    std::vector<StrengthTestSpec> specs;
    std::string line;
    while (std::getline(epdFile, line))
    {
        specs.emplace_back(ParseEpd(line));
    }
    return specs;
}

// This is a really quick-and-dirty .epd "parser" written specifically for STS, ERET and Arasan20 test suites.
// It's basically trying hard not to be an actual parser. Now that it's done I'm thinking that was the wrong choice.
// Too-many-dashes in ERET is fixed in the .epd file rather than handled here in code (two instances IIRC).
StrengthTestSpec Epd::ParseEpd(const std::string& epd)
{
    StrengthTestSpec spec;
    std::vector<std::string> bestMoveSans;

    std::stringstream tokenizer(epd);
    std::string token;

    // Parse the partial FEN.
    spec.fen.reserve(128);
    for (int i = 0; i < 4; i++)
    {
        tokenizer >> token;
        spec.fen += token + " ";
    }
    assert(tokenizer);
    spec.fen += "0 1"; // Move clocks are optional in EPDs, usually 0/1 for test suites.

    // Parse the best move(s) or avoid move(s).
    tokenizer >> token;
    assert(tokenizer);
    if (token == "bm")
    {
        bestMoveSans = ReadMoves(tokenizer, token);
    }
    else if (token == "am")
    {
        spec.avoidSans = ReadMoves(tokenizer, token);
    }
    else
    {
        assert(false);
    }

    // Parse SANs for best + alternatives (optional).
    if (Find(tokenizer, token, "c7"))
    {
        spec.pointSans = ReadMoves(tokenizer, token /* may gut token */);
    }

    // Parse points for best + alternatives (optional).
    if (Find(tokenizer, token, "c8"))
    {
        int points;
        Expect(tokenizer, '"');
        while (tokenizer >> points)
        {
            spec.points.push_back(points);
        }
        tokenizer.clear();
        Expect(tokenizer, '"');
        Expect(tokenizer, ';');
    }

    // Validate SANs/points.
    if (spec.pointSans.size() != spec.points.size())
    {
        assert(false);
        spec.pointSans.clear();
        spec.points.clear();
    }

    // If best-moves exist and no points, populate points.
    if (!bestMoveSans.empty() && spec.pointSans.empty())
    {
        spec.pointSans = std::move(bestMoveSans);
        spec.points.resize(spec.pointSans.size());
        std::fill(spec.points.begin(), spec.points.end(), 1);
    }

    return spec;
}

std::vector<std::string> Epd::ReadMoves(std::istream& tokenizer, std::string& token)
{
    std::vector<std::string> moves;
    bool quoted = false;

    while ((tokenizer >> token) && !token.empty())
    {
        if (token[0] == '"')
        {
            token = StripLeft(token, '"');
            quoted = true;
        }

        if (token.back() == ';')
        {
            moves.emplace_back(quoted ?
                StripRight(StripRight(token, ';'), '"') :
                StripRight(token, ';')
                );
            break;
        }
        else
        {
            moves.emplace_back(std::move(token));
        }
    }

    return moves;
}

void Epd::Expect(std::istream& tokenizer, [[maybe_unused]] const unsigned char expected)
{
    unsigned char c;
    tokenizer >> c;
    assert(tokenizer);
    assert(c == expected);
}

bool Epd::Find(std::istream& tokenizer, std::string& token, const std::string& expected)
{
    while (tokenizer >> token)
    {
        if (token == expected)
        {
            return true;
        }
    }
    return false;
}

std::string Epd::StripLeft(const std::string& string, [[maybe_unused]] char expected)
{
    assert(!string.empty() && (string[0] == expected));
    return string.substr(1, string.size() - 1);
}

std::string Epd::StripRight(const std::string& string, [[maybe_unused]] char expected)
{
    assert(!string.empty() && (string.back() == expected));
    return string.substr(0, string.size() - 1);
}