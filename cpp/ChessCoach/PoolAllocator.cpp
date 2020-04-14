#include "PoolAllocator.h"

#include <windows.h>

size_t LargePageAllocator::LargePageMinimum = 0;

void* LargePageAllocator::Allocate(size_t byteCount)
{
    Initialize();

    byteCount = ((byteCount + LargePageMinimum - 1) / LargePageMinimum) * LargePageMinimum;
    return ::VirtualAlloc(nullptr, byteCount, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
}

void LargePageAllocator::Free(void* memory)
{
    ::VirtualFree(memory, 0, MEM_RELEASE);
}

void LargePageAllocator::Initialize()
{
    if (LargePageMinimum > 0)
    {
        return;
    }

    HANDLE hToken;
    TOKEN_PRIVILEGES tokenPrivileges;

    BOOL opened = ::OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken);
    assert(opened);

    BOOL gotLuid = ::LookupPrivilegeValue(nullptr, L"SeLockMemoryPrivilege", &tokenPrivileges.Privileges[0].Luid);
    assert(gotLuid);

    tokenPrivileges.PrivilegeCount = 1;
    tokenPrivileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    BOOL adjusted = ::AdjustTokenPrivileges(hToken, FALSE, &tokenPrivileges, 0, nullptr, 0);
    assert(adjusted);

    LargePageMinimum = ::GetLargePageMinimum();
}