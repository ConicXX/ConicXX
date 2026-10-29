#include "conicxx/version.h"

#include <gtest/gtest.h>

#include <regex>

TEST(Version, StringMatchesMacroAndSemverFormat) {
  EXPECT_STREQ(conicxx::version(), CONICXX_VERSION_STRING);
  EXPECT_TRUE(std::regex_match(CONICXX_VERSION_STRING, std::regex(R"(\d+\.\d+\.\d+)")));
}

TEST(Version, ComponentMacrosMatchVersionString) {
  const std::string expected = std::to_string(CONICXX_VERSION_MAJOR) + "." +
                                std::to_string(CONICXX_VERSION_MINOR) + "." +
                                std::to_string(CONICXX_VERSION_PATCH);
  EXPECT_EQ(expected, CONICXX_VERSION_STRING);
}
