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
thread_local std::string gCurrentProfilePhase;

static bool _endsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string _stripSuffix(std::string name, const std::string& suffix) {
    if (_endsWith(name, suffix)) {
        name.resize(name.size() - suffix.size());
    }
    return name;
}

static std::string _profileNameKey(const std::string& name) {
    auto key = name;
    key = _stripSuffix(key, "_matmul_converted");
    key = _stripSuffix(key, "__matmul_converted");
    key = _stripSuffix(key, "_converted");
    return key;
}

static bool _sameProfileName(const std::string& lhs, const std::string& rhs) {
    if (lhs == rhs) {
        return true;
    }
    auto lhsKey = _profileNameKey(lhs);
    auto rhsKey = _profileNameKey(rhs);
    if (lhsKey == rhsKey) {
        return true;
    }
    // Converted matmul names sometimes append an intermediate output segment after the logical op name.
    if (lhs.find("_raster_") != std::string::npos || rhs.find("_raster_") != std::string::npos) {
        return false;
    }
    return lhsKey.find(rhsKey + "/") == 0 || rhsKey.find(lhsKey + "/") == 0;
}

template <typename T>
static typename std::map<std::string, T>::const_iterator _findProfileValue(const std::map<std::string, T>& values,
                                                                           const std::string& opName) {
    auto iter = values.find(opName);
    if (iter != values.end()) {
        return iter;
    }
    for (iter = values.begin(); iter != values.end(); ++iter) {
        if (_sameProfileName(iter->first, opName)) {
            return iter;
        }
    }
    return values.end();
}

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
    auto iter = _findProfileValue(gProfileExecutionInfo, opName);
    if (iter == gProfileExecutionInfo.end()) {
        return "";
    }
    return iter->second;
}

std::string getCurrentProfileExecutionInfo(const std::string& opName) {
    std::lock_guard<std::mutex> lock(gProfileExecutionInfoMutex);
    auto iter = _findProfileValue(gCurrentProfileExecutionInfo, opName);
    if (iter != gCurrentProfileExecutionInfo.end()) {
        return iter->second;
    }
    iter = _findProfileValue(gProfileExecutionInfo, opName);
    if (iter != gProfileExecutionInfo.end()) {
        return iter->second;
    }
    return "";
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
    auto iter = _findProfileValue(gProfileExecutionBytes, opName);
    if (iter == gProfileExecutionBytes.end()) {
        return 0;
    }
    return iter->second;
}

void setCurrentProfilePhase(const std::string& phase) {
    gCurrentProfilePhase = phase;
}

std::string getCurrentProfilePhase() {
    return gCurrentProfilePhase;
}

void clearProfileExecutionInfo() {
    std::lock_guard<std::mutex> lock(gProfileExecutionInfoMutex);
    gProfileExecutionInfo.clear();
    gCurrentProfileExecutionInfo.clear();
    gProfileExecutionBytes.clear();
}

} // namespace MNN
