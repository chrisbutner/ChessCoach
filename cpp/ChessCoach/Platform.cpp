#include "Platform.h"

#include <cstdlib>

std::filesystem::path Platform::InstallationScriptPath()
{
#ifdef CHESSCOACH_WINDOWS
    return std::filesystem::current_path();
#else
    return "/usr/local/bin/ChessCoach";
#endif
}

std::filesystem::path Platform::InstallationDataPath()
{
#ifdef CHESSCOACH_WINDOWS
    return std::filesystem::current_path();
#else
    return "/usr/local/share/ChessCoach";
#endif
}

std::filesystem::path Platform::UserDataPath()
{
#ifdef CHESSCOACH_WINDOWS
    return std::filesystem::path(GetEnvironmentVariable("localappdata")) / "ChessCoach";
#else
    const std::string xdgDataHome = GetEnvironmentVariable("XDG_DATA_HOME");
    return !xdgDataHome.empty() ?
        (std::filesystem::path(xdgDataHome) / "ChessCoach") :
        (std::filesystem::path(GetEnvironmentVariable("HOME")) / ".local/share/ChessCoach");
#endif
}

std::string Platform::GetEnvironmentVariable(const char* name)
{
#pragma warning(disable:4996) // Internal buffer is immediately consumed and detached.
    const char* value = ::getenv(name);
    return (value ? value : "");
#pragma warning(default:4996) // Internal buffer is immediately consumed and detached.
}

void Platform::SetEnvironmentVariable(const char* name, const char* value)
{
#ifdef CHESSCOACH_WINDOWS
    _putenv_s(name, value);
#else
    ::setenv(name, value, 1);
#endif
}