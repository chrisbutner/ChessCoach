#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#include <string>
#include <filesystem>

// Treat everything else as Linux + gcc/clang, rather than forcing a failure. If it works, it works.
#ifdef _WIN32
#define CHESSCOACH_WINDOWS
#endif

class Platform
{
public:

    static std::filesystem::path InstallationScriptPath();
    static std::filesystem::path InstallationDataPath();
    static std::filesystem::path UserDataPath();

    static std::string GetEnvironmentVariable(const char* name);
    static void SetEnvironmentVariable(const char* name, const char* value);

};

#endif // _PLATFORM_H_