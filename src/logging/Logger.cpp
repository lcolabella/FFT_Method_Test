#include "logging/Logger.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace common {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::open(const std::string& filePath) {
    const std::filesystem::path path(filePath);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    file_.open(filePath, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot open log file: " + filePath);
    }
}

void Logger::info(const std::string& message) {
    write("INFO", message);
}

void Logger::warn(const std::string& message) {
    write("WARN", message);
}

void Logger::error(const std::string& message) {
    write("ERROR", message);
}

void Logger::write(const std::string& level, const std::string& message) {
    const std::string line = "[" + timestamp() + "] [" + level + "] " + message;

    std::cout << line << '\n';
    if (file_.is_open()) {
        file_ << line << '\n';
    }
}

std::string Logger::timestamp() const {
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);

    std::tm tmTime{};
#if defined(_WIN32)
    localtime_s(&tmTime, &nowTime);
#else
    localtime_r(&nowTime, &tmTime);
#endif

    std::ostringstream output;
    output << std::put_time(&tmTime, "%Y-%m-%d %H:%M:%S");
    return output.str();
}

} // namespace common
