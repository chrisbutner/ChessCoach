#include <iostream>

#include "bitboard.h"
#include "position.h"
#include "thread.h"
#include "tt.h"

int main(int argc, char* argv[])
{
    Bitboards::init();
    Position::init();
    Bitbases::init();

    return 0;
}
