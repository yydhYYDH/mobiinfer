//
//  ProfileExecutionInfo.cpp
//  MNN
//
//  Created by MNN on 2026/06/24.
//

#include "core/ProfileExecutionInfo.hpp"

#include <map>
#include <mutex>

namespace MNN {
namespace {

std::mutex gProfileExecutionInfoMutex;
std::map<std::string, std::string> gProfileExecutionInfo;
std::map<std::string, std::string> gCurrentProfileExecutionInfo;
std::map<std::string, uint64_t> gProfileExecutionBytes;

} // namespace

void setProfileExecutionInfo(const std::string& opName, const std::string& info) {
    if (opName.empty() || info.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(gProfileExecutionInfoMutex);
    auto& current = gProfileExecutionInfo[opName];
    if (current.empty()) {
        current = info;
        gCurrentProfileExecutionInfo[opName] = info;
        return;
    }
    if (current.find(info) != std::string::npos) {
        gCurrentProfileExecutionInfo[opName] = info;
        return;
    }
    current += " | ";
    current += info;
    gCurrentProfileExecutionInfo[opName] = info;
}

std::string getProfileExecutionInfo(const std::string& opName) {
    std::lock_guard<std::mutex> lock(gProfileExecutionInfoMutex);
    auto iter = gProfileExecutionInfo.find(opName);
    if (iter == gProfileExecutionInfo.end()) {
        return "";
    }
    return iter->second;
}

std::string getCurrentProfileExecutionInfo(const std::string& opName) {
    std::lock_guard<std::mutex> lock(gProfileExecutionInfoMutex);
    auto iter = gCurrentProfileExecutionInfo.find(opName);
    if (iter == gCurrentProfileExecutionInfo.end()) {
        return "";
    }
    return iter->second;
}

void setProfileExecutionBytes(const std::string& opName, uint64_t bytes) {
    if (opName.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(gProfileExecutionInfoMutex);
    gProfileExecutionBytes[opName] = bytes;
}

uint64_t getProfileExecutionBytes(const std::string& opName) {
    std::lock_guard<std::mutex> lock(gProfileExecutionInfoMutex);
    auto iter = gProfileExecutionBytes.find(opName);
    if (iter == gProfileExecutionBytes.end()) {
        return 0;
    }
    return iter->second;
}

void clearProfileExecutionInfo() {
    std::lock_guard<std::mutex> lock(gProfileExecutionInfoMutex);
    gProfileExecutionInfo.clear();
    gCurrentProfileExecutionInfo.clear();
    gProfileExecutionBytes.clear();
}

} // namespace MNN
