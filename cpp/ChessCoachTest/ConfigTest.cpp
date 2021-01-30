#include <gtest/gtest.h>

#include <ChessCoach/Config.h>
#include <toml11/toml.hpp>

TEST(Config, Basic)
{
    // Mainly just want Initialize not to crash. Check a couple arbitrary things.
    Config::Initialize();
    EXPECT_TRUE(!Config::TrainingNetwork.Name.empty());
    EXPECT_TRUE(!Config::UciNetwork.Name.empty());
}

TEST(Config, ParameterAssignment)
{
    Config::Initialize();

    Config::UpdateParameters({ { "batch_size", 512.f } });

    EXPECT_THROW(Config::UpdateParameters({ { "batch_sizez", 512.f } }), std::runtime_error);
}