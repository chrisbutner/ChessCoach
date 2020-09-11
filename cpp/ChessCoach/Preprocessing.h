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