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

#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#include <string>
#include <filesystem>

// Treat everything else as Linux + gcc, rather than forcing a failure. If it works, it works.
#ifdef _WIN32
#define CHESSCOACH_WINDOWS
#endif

class ChessCoachException : public std::runtime_error
{

public:

    explicit ChessCoachException(const std::string& message)
        : std::runtime_error(message)
    {
    }
};

class Platform
{
public:

    static std::filesystem::path InstallationScriptPath();
    static std::filesystem::path InstallationDataPath();
    static std::filesystem::path UserDataPath();
    static std::filesystem::path GetExecutableDirectory();

    static std::string GetEnvironmentVariable(const char* name);
    static void SetEnvironmentVariable(const char* name, const char* value);

    static void DebugBreak();

};

class PosixFile
{
public:

    PosixFile(const std::filesystem::path& path, bool write);
    ~PosixFile();
    int FileDescriptor() const;

private:

    int _fileDescriptor;
};

#endif // _PLATFORM_H_