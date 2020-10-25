#include "Preprocessing.h"

#include <algorithm>
#include <filesystem>
#include <sstream>

#include "Platform.h"

Preprocessor::Preprocessor()
{
    const std::filesystem::path englishPath = (Platform::InstallationDataPath() / "Dictionaries" / "en");
    const std::filesystem::path affixPath = (englishPath / "en_US.aff");
    const std::filesystem::path dictionaryPath = (englishPath / "en_US.dic");
    
    _hunspell.reset(new Hunspell(affixPath.string().c_str(), dictionaryPath.string().c_str()));
}

// Assume English in UTF-8, so for the most part C++ characters == text characters.
void Preprocessor::PreprocessComment(std::string& comment) const
{
    // Apply simple rules to find comments that aren't useful.
    if (IsBadComment(comment))
    {
        comment.clear();
        return;
    }

    // Strip off purely wrapping punctuation.
    while (true)
    {
        if (Unwrap(comment, '(', ')') ||
            Unwrap(comment, '[', ']') ||
            Unwrap(comment, '{', '}') ||
            Unwrap(comment, '"', '"') ||
            Unwrap(comment, '\'', '\''))
        {
            continue;
        }

        break;
    }

    // Strip off starting noise like punctuation, symbols, etc. that are too far from English.
    StripStartingNoise(comment);

    // Comments should contain some number of English words and not be dominated by symbols/names/other languages.
    if (!IsSufficientlyEnglish(comment))
    {
        comment.clear();
        return;
    }

    // Convert newlines to spaces and re-trim.
    std::replace(comment.begin(), comment.end(), '\n', ' ');
    comment.erase(std::remove(comment.begin(), comment.end(), '\r'), comment.end());
    Trim(comment);
}

void Preprocessor::Trim(std::string& text)
{
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](char c) {
        return !::isspace(static_cast<unsigned char>(c));
        }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [](char c) {
        return !::isspace(static_cast<unsigned char>(c));
        }).base(), text.end());
}

bool Preprocessor::WordTokenize(std::istream& content, std::string& tokenOut) const
{
    tokenOut.clear();
    char c;
    const std::string delimiters(EnglishPunctuationPlusSlash);
    while (content >> std::noskipws >> c)
    {
        if (::isspace(static_cast<unsigned char>(c)) || (delimiters.find(c) != std::string::npos))
        {
            if (!tokenOut.empty())
            {
                return true;
            }
        }
        else
        {
            tokenOut += c;
        }
    }

    return !tokenOut.empty();
}

bool Preprocessor::IsSufficientlyEnglish(const std::string& comment) const
{
    std::string token;
    int tokenCount = 0;
    int englishCount = 0;

    // Tokenize into word candidates, delimited by English punctuation plus slash (/).
    std::stringstream tokenizer(comment);
    while (WordTokenize(tokenizer, token))
    {
        tokenCount++;

        // Ensure first character is lowercase (for latin characters) so that
        // proper nouns don't count: strings of names aren't useful comments.
        token[0] = std::tolower(token[0], std::locale());

        if (!IsNotEnglish(token) &&
            _hunspell->spell(token))
        {
            englishCount++;
        }
    }

    const int characterCount = static_cast<int>(comment.size());
    const int minEnglishCount = std::max({ 2, (tokenCount + 2) / 3, (characterCount + 20) / 21 });
    return (englishCount >= minEnglishCount);
}

bool Preprocessor::IsNotEnglish(const std::string& token) const
{
    // Don't allow words with digits.
    if (token.find_first_of("0123456789") != std::string::npos)
    {
        return true;
    }

    // Don't allow single-letter words except for A/a/I/i (Hunspell is very generous for some reason).
    if (token.size() <= 1)
    {
        return !((token == "A") || (token == "a") || (token == "I") || (token == "i"));
    }

    return false;
}

bool Preprocessor::IsBadComment(const std::string& comment) const
{
    // Throw away comments with 500+ characters.
    if (comment.size() >= 500)
    {
        return true;
    }

    // Throw away pointless references, quizes, etc.
    if ((comment.find("details you can see") != std::string::npos) ||
        (comment.find("details, you can see") != std::string::npos) ||
        (comment.find("information you can see") != std::string::npos) ||
        (comment.find("information, you can see") != std::string::npos) ||
        (comment.find("details see the notes") != std::string::npos) ||
        (comment.find("can see my comments") != std::string::npos) ||
        (comment.find("can find in annotations") != std::string::npos) ||
        (comment.find("can find in my annotations") != std::string::npos) ||
        (comment.find("can see annotations") != std::string::npos) ||
        (comment.find("can see comments") != std::string::npos) ||
        (comment.find("can see my annotations") != std::string::npos) ||
        (comment.find("can see my comments") != std::string::npos) ||
        (comment.find("[% tqu") != std::string::npos) ||
        (comment.find("[%tqu") != std::string::npos) ||
        (comment.find("converted with error") != std::string::npos))
    {
        return true;
    }

    return false;
}

bool Preprocessor::Unwrap(std::string& token, char left, char right) const
{
    if (!token.empty() &&
        (token.front() == left) &&
        (token.back() == right) &&
        (token.find(left, 1) == std::string::npos) &&
        (token.rfind(right, token.size() - 2) == std::string::npos))
    {
        token.pop_back();
        token.erase(token.begin());
        Trim(token);
        return true;
    }

    return false;
}

void Preprocessor::StripStartingNoise(std::string& comment) const
{
    int retainFrom = 0;
    std::string token;

    // Grab space-delimited tokens.
    std::stringstream tokenizer(comment);
    while (tokenizer >> token)
    {
        // To survive stripping, the token must:
        // (a) contain only A-Za-z, digits, SAN characters and English punctuation plus slash (/),
        // (b) contain at least one A-Za-z
        // (e.g. "Karpov good, hi good, Nf3+ good, e8=Q# good, _|_ bad, [%mdl bad, café bad (unfortunately), 100% bad (unfortunately))
        //
        // It would be better to also allow for accented latin characters but it involves
        // checking a few ranges in both ASCII and UTF-8, which is better done via a library,
        // so additional complexity for almost no gain based on data I'm seeing.
        //
        // It also be better to allow for some less common punctuation like "100%" but that
        // allows in too much junk while not keeping much useful extra.
        const std::string az = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        const std::string digitsSan = "0123456789+#=";
        const std::string englishPunctuationPlusSlash(EnglishPunctuationPlusSlash);
        bool illegalFound = false;
        bool azFound = false;
        for (char c : token)
        {
            const bool isAz = (az.find(c) != std::string::npos);
            azFound |= isAz;
            if (!isAz && (digitsSan.find(c) == std::string::npos) && (englishPunctuationPlusSlash.find(c) == std::string::npos))
            {
                illegalFound = true;
                break;
            }
        }

        if (!illegalFound && azFound)
        {
            // This token is fine, so break out and use "retainFrom".
            break;
        }

        // We're stripping this token, so move "retainFrom" up.
        retainFrom = ((tokenizer && !tokenizer.eof()) ? static_cast<int>(tokenizer.tellg()) : static_cast<int>(comment.size()));
    }

    if (retainFrom > 0)
    {
        comment.erase(comment.begin(), comment.begin() + retainFrom);
        Trim(comment);
    }
}