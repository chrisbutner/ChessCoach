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

#include "PoolAllocator.h"

#include <cstdlib>

#include "Platform.h"

#ifdef CHESSCOACH_WINDOWS
#include <windows.h>
#else
#include <sys/mman.h>
#endif

size_t LargePageAllocator::LargePageMinimum = 0;

void* LargePageAllocator::Allocate(size_t byteCount)
{
    Initialize();

    byteCount = ((byteCount + LargePageMinimum - 1) / LargePageMinimum) * LargePageMinimum;

#ifdef CHESSCOACH_WINDOWS
    return ::VirtualAlloc(nullptr, byteCount, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
#else
    {
        // The madvise() isn't strictly necessary outside of embedded but it doesn't hurt.
        void* memory = std::aligned_alloc(LargePageMinimum, byteCount);
        ::madvise(memory, byteCount, MADV_HUGEPAGE);
        return memory;
    }
#endif
}

void LargePageAllocator::Free(void* memory)
{
#ifdef CHESSCOACH_WINDOWS
    ::VirtualFree(memory, 0, MEM_RELEASE);
#else
    std::free(memory);
#endif
}

void LargePageAllocator::Initialize()
{
    if (LargePageMinimum > 0)
    {
        return;
    }

#ifdef CHESSCOACH_WINDOWS
    HANDLE hToken;
    TOKEN_PRIVILEGES tokenPrivileges;

    BOOL opened = ::OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken);
    assert(opened);
    if (!opened)
    {
        throw std::bad_alloc();
    }

    BOOL gotLuid = ::LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tokenPrivileges.Privileges[0].Luid);
    assert(gotLuid);
    if (!gotLuid)
    {
        throw std::bad_alloc();
    }

    tokenPrivileges.PrivilegeCount = 1;
    tokenPrivileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    BOOL adjusted = ::AdjustTokenPrivileges(hToken, FALSE, &tokenPrivileges, 0, nullptr, 0);
    assert(adjusted);
    if (!adjusted)
    {
        throw std::bad_alloc();
    }

    // Close using RAII if reusing as library code.
    ::CloseHandle(hToken);

    LargePageMinimum = ::GetLargePageMinimum();
#else
    // x86_64 apparently has two possibilities, 2 MiB and 1 GiB, but we'll probably only ever see 2 MiB.
    // If we happen to allocate a multiple of 1 GiB (e.g. maybe prediction cache) then so be it.
    LargePageMinimum = 2 * 1024 * 1024;
#endif
}