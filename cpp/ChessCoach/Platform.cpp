#include "Platform.h"

#include <cstdlib>
#include <fcntl.h>
#include <cassert>

#ifdef CHESSCOACH_WINDOWS
#include <io.h>
#include <Windows.h> // Required for <libloaderapi.h> for GetModuleFileNameA
#undef GetEnvironmentVariable
#undef SetEnvironmentVariable
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define O_BINARY 0
#endif

std::filesystem::path Platform::InstallationScriptPath()
{
#ifdef CHESSCOACH_WINDOWS
    return GetExecutableDirectory();
#else
    return "/usr/local/bin/ChessCoach";
#endif
}

std::filesystem::path Platform::InstallationDataPath()
{
#ifdef CHESSCOACH_WINDOWS
    return GetExecutableDirectory();
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

std::filesystem::path Platform::GetExecutableDirectory()
{
#ifdef CHESSCOACH_WINDOWS
    CHAR path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    return std::filesystem::path(path).parent_path();
#else
    throw std::exception("Not implemented");
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
    ::_putenv_s(name, value);
#else
    ::setenv(name, value, 1);
#endif
}

void Platform::DebugBreak()
{
#ifdef CHESSCOACH_WINDOWS
    ::__debugbreak();
#else
    __asm__ __volatile__("int3");
#endif
}

PosixFile::PosixFile(const std::filesystem::path& path, bool write)
{
#pragma warning(disable:4996) // It's all fine.
    _fileDescriptor = ::open(path.string().c_str(), (write ? (O_CREAT | O_WRONLY) : O_RDONLY) | O_BINARY, 0644);
    assert(_fileDescriptor != -1);
#pragma warning(default:4996) // It's all fine.
}

PosixFile::~PosixFile()
{
#pragma warning(disable:4996) // Don't worry about conformant name.
    ::close(_fileDescriptor);
#pragma warning(default:4996) // Don't worry about conformant name.
}

int PosixFile::FileDescriptor() const
{
    return _fileDescriptor;
}