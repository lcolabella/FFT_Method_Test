#pragma once

#include <fstream>
#include <string>

namespace common {

class Logger {
public:
    static Logger& instance();

    void open(const std::string& filePath);
    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);

private:
    Logger() = default;

    void write(const std::string& level, const std::string& message);
    std::string timestamp() const;

    std::ofstream file_;
};

} // namespace common
