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