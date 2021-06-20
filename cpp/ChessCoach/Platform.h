#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#include <string>
#include <filesystem>

// Treat everything else as Linux + gcc, rather than forcing a failure. If it works, it works.
#ifdef _WIN32
#define CHESSCOACH_WINDOWS
#endif

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