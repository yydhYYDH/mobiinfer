//
//  Profiler.cpp
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include <algorithm>
#include <fstream>
#include <string>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif
#include "Profiler.hpp"
#include "core/Macro.h"
#include "core/ProfileExecutionInfo.hpp"

#define MFLOPS (1e6)

namespace MNN {
    
static inline int64_t getTime() {
    uint64_t time;
#if defined(_MSC_VER)
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    uint64_t sec = now.QuadPart / freq.QuadPart;
    uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time = sec * 1000000 + usec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
    return time;
}

static std::string toString(float value) {
    char typeString[100] = {};
    sprintf(typeString, "%f", value);
    return std::string(typeString);
}

static std::string toString(const std::vector<int>& shape) {
    char content[100] = {};
    auto current      = content;
    for (auto s : shape) {
        current = current + sprintf(current, "%d,", s);
    }
    return std::string(current);
}

static uint64_t tensorBytes(const std::vector<Tensor*>& tensors) {
    uint64_t bytes = 0;
    for (auto tensor : tensors) {
        if (tensor == nullptr) {
            continue;
        }
        bytes += tensor->usize();
    }
    return bytes;
}

static std::string readField(const std::string& content, const std::string& key, const std::string& fallback = "") {
    auto pos = content.find(key);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos += key.size();
    auto end = content.find(' ', pos);
    if (end == std::string::npos) {
        return content.substr(pos);
    }
    return content.substr(pos, end - pos);
}

static std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) {
        return value;
    }
    std::string result = "\"";
    for (auto c : value) {
        if (c == '"') {
            result += "\"\"";
        } else {
            result += c;
        }
    }
    result += "\"";
    return result;
}

static bool containsText(const std::string& value, const std::string& token) {
    return value.find(token) != std::string::npos;
}

static bool isVisionMarkerName(const std::string& name) {
    return containsText(name, "patch_embed") || containsText(name, "/merger/") ||
           containsText(name, "deepstack_merger") || containsText(name, "deepstack_feature") ||
           containsText(name, "image_embeds");
}

static std::string inferPhase(const std::string& name, const std::string& type, const std::string& phase,
                              const std::string& execution) {
    auto explicitPhase = getCurrentProfilePhase();
    if (!explicitPhase.empty()) {
        return explicitPhase;
    }
    if (!phase.empty() && phase != "unknown") {
        return phase;
    }
    auto execPhase = readField(execution, "phase=");
    if (!execPhase.empty() && execPhase != "unknown") {
        return execPhase;
    }
    if (containsText(name, "/lm/lm_head/")) {
        return "decode";
    }
    if (name.rfind("/layers.", 0) == 0 || containsText(name, "/lm/")) {
        return "llm_prefill_or_decode";
    }
    if (isVisionMarkerName(name)) {
        return "vision_prefill";
    }
    if (type == "Attention") {
        return "attention_unknown";
    }
    return "unknown";
}

Profiler* Profiler::gInstance = nullptr;
Profiler* Profiler::getInstance() {
    if (gInstance == nullptr) {
        gInstance = new Profiler;
    }
    return gInstance;
}

void Profiler::reset() {
    mStartTime = 0;
    mEndTime = 0;
    mProfileOriginTime = 0;
    mTotalTime = 0.0f;
    mTotalMFlops = 0.0f;
    mMapByType.clear();
    mMapByName.clear();
    mMapByNamePhase.clear();
    mActiveNamePhaseKey.clear();
    mEvents.clear();
    mActiveEventIndex.clear();
    mSequence = 0;
}

std::vector<Profiler::Event> Profiler::refinePhasesByTimeline() const {
    std::vector<Event> events = mEvents;
    if (events.empty()) {
        return events;
    }
    int firstLmHead = -1;
    int lastVisionMarker = -1;
    for (int i = 0; i < events.size(); ++i) {
        if (firstLmHead < 0 && containsText(events[i].name, "/lm/lm_head/")) {
            firstLmHead = i;
        }
        if (isVisionMarkerName(events[i].name)) {
            lastVisionMarker = i;
        }
    }
    int visionEnd = lastVisionMarker;
    while (visionEnd >= 0 && visionEnd + 1 < events.size() && !containsText(events[visionEnd + 1].name, "/lm/lm_head/")) {
        const auto& name = events[visionEnd + 1].name;
        if (name.rfind("_raster_", 0) == 0 || containsText(name, "image_embeds") ||
            containsText(name, "deepstack_feature")) {
            ++visionEnd;
            continue;
        }
        break;
    }
    for (int i = 0; i < events.size(); ++i) {
        if (events[i].inferredPhase == "eagle") {
            continue;
        }
        if (visionEnd >= 0 && i <= visionEnd) {
            events[i].inferredPhase = "vision_prefill";
        } else if (firstLmHead >= 0 && i >= firstLmHead) {
            events[i].inferredPhase = "decode";
        } else {
            events[i].inferredPhase = "prefill";
        }
    }
    return events;
}

Profiler::Record& Profiler::getTypedRecord(const OperatorInfo* op) {
    auto typeStr = op->type();
    auto iter = mMapByType.find(typeStr);
    if (iter != mMapByType.end()) {
        return iter->second;
    }

    // create new
    mMapByType.insert(std::make_pair(typeStr, Record()));
    Record& record     = mMapByType.find(typeStr)->second;
    record.costTime    = 0.0f;
    record.dispatchTime = 0.0f;
    record.waitTime    = 0.0f;
    record.calledTimes = 0;
    record.type        = op->type();
    record.flops       = 0.0f;
    record.inputBytes  = 0;
    record.outputBytes = 0;

    return record;
}

Profiler::Record& Profiler::getNamedRecord(const OperatorInfo* op) {
    auto name = op->name();
    auto iter = mMapByName.find(name);
    if (iter != mMapByName.end()) {
        return iter->second;
    }

    // create new
    mMapByName.insert(std::make_pair(name, Record()));
    Record& record     = mMapByName.find(name)->second;
    record.costTime    = 0.0f;
    record.name        = op->name();
    record.type        = op->type();
    record.flops       = 0.0f;
    record.dispatchTime = 0.0f;
    record.waitTime    = 0.0f;
    record.inputBytes  = 0;
    record.outputBytes = 0;

    return record;
}

Profiler::Record& Profiler::getNamePhaseRecord(const OperatorInfo* op) {
    auto execution = getCurrentProfileExecutionInfo(op->name());
    auto phase = readField(execution, "phase=", "unknown");
    auto key = op->name() + "\t" + phase + "\t" + execution;
    auto iter = mMapByNamePhase.find(key);
    if (iter != mMapByNamePhase.end()) {
        return iter->second;
    }

    mMapByNamePhase.insert(std::make_pair(key, Record()));
    Record& record     = mMapByNamePhase.find(key)->second;
    record.costTime    = 0.0f;
    record.name        = op->name();
    record.type        = op->type();
    record.phase       = phase;
    record.execution   = execution;
    record.flops       = 0.0f;
    record.dispatchTime = 0.0f;
    record.waitTime    = 0.0f;
    record.inputBytes  = 0;
    record.outputBytes = 0;
    return record;
}

void Profiler::start(const OperatorInfo* info) {
    start(std::vector<Tensor*>(), info);
}

void Profiler::start(const std::vector<Tensor*>& inputs, const OperatorInfo* info) {
    mStartTime = getTime();
    if (mProfileOriginTime == 0) {
        mProfileOriginTime = mStartTime;
    }
    mTotalMFlops += info->flops();
    auto& typed = getTypedRecord(info);
    typed.calledTimes++;
    typed.flops += info->flops();
    typed.inputBytes += tensorBytes(inputs);
    auto& named = getNamedRecord(info);
    named.flops += info->flops();
    named.inputBytes += tensorBytes(inputs);
    auto& namedPhase = getNamePhaseRecord(info);
    namedPhase.calledTimes++;
    namedPhase.flops += info->flops();
    namedPhase.inputBytes += tensorBytes(inputs);
    mActiveNamePhaseKey = info->name() + "\t" + namedPhase.phase + "\t" + namedPhase.execution;

    Event event;
    event.sequence = mSequence++;
    event.name = info->name();
    event.type = info->type();
    event.phase = namedPhase.phase;
    event.inferredPhase = inferPhase(event.name, event.type, event.phase, namedPhase.execution);
    event.execution = namedPhase.execution;
    event.startTime = (float)(mStartTime - mProfileOriginTime) / 1000.0f;
    event.endTime = event.startTime;
    event.costTime = 0.0f;
    event.dispatchTime = 0.0f;
    event.waitTime = 0.0f;
    event.flops = info->flops();
    event.inputBytes = tensorBytes(inputs);
    event.outputBytes = 0;
    event.staticBytes = getProfileExecutionBytes(info->name());
    mEvents.emplace_back(event);
    mActiveEventIndex[std::string(info->type()) + ":" + info->name()] = static_cast<int64_t>(mEvents.size() - 1);
}

void Profiler::end(const OperatorInfo* info) {
    mEndTime   = getTime();
    float cost = (float)(mEndTime - mStartTime) / 1000.0f;
    mMapByType[info->type()].costTime += cost;
    mMapByType[info->type()].dispatchTime += cost;
    mMapByName[info->name()].costTime += cost;
    mMapByName[info->name()].dispatchTime += cost;
    mMapByNamePhase[mActiveNamePhaseKey].costTime += cost;
    mMapByNamePhase[mActiveNamePhaseKey].dispatchTime += cost;
    auto activeEvent = mActiveEventIndex.find(std::string(info->type()) + ":" + info->name());
    if (activeEvent != mActiveEventIndex.end() && activeEvent->second >= 0 &&
        activeEvent->second < static_cast<int64_t>(mEvents.size())) {
        auto& event = mEvents[activeEvent->second];
        event.endTime = (float)(mEndTime - mProfileOriginTime) / 1000.0f;
        event.costTime = cost;
        event.dispatchTime = cost;
        event.waitTime = 0.0f;
    }
    mTotalTime += cost;
}

void Profiler::end(const std::vector<Tensor*>& outputs, const OperatorInfo* info) {
    auto dispatchEnd = getTime();
    for (auto o : outputs) {
        o->wait(MNN::Tensor::MAP_TENSOR_READ, true);
    }
    auto waitEnd = getTime();
    float dispatchCost = (float)(dispatchEnd - mStartTime) / 1000.0f;
    float waitCost = (float)(waitEnd - dispatchEnd) / 1000.0f;
    float cost = dispatchCost + waitCost;
    mMapByType[info->type()].costTime += cost;
    mMapByType[info->type()].dispatchTime += dispatchCost;
    mMapByType[info->type()].waitTime += waitCost;
    mMapByType[info->type()].outputBytes += tensorBytes(outputs);
    mMapByName[info->name()].costTime += cost;
    mMapByName[info->name()].dispatchTime += dispatchCost;
    mMapByName[info->name()].waitTime += waitCost;
    mMapByName[info->name()].outputBytes += tensorBytes(outputs);
    mMapByNamePhase[mActiveNamePhaseKey].costTime += cost;
    mMapByNamePhase[mActiveNamePhaseKey].dispatchTime += dispatchCost;
    mMapByNamePhase[mActiveNamePhaseKey].waitTime += waitCost;
    mMapByNamePhase[mActiveNamePhaseKey].outputBytes += tensorBytes(outputs);
    auto activeEvent = mActiveEventIndex.find(std::string(info->type()) + ":" + info->name());
    if (activeEvent != mActiveEventIndex.end() && activeEvent->second >= 0 &&
        activeEvent->second < static_cast<int64_t>(mEvents.size())) {
        auto& event = mEvents[activeEvent->second];
        event.endTime = (float)(waitEnd - mProfileOriginTime) / 1000.0f;
        event.costTime = cost;
        event.dispatchTime = dispatchCost;
        event.waitTime = waitCost;
        event.outputBytes = tensorBytes(outputs);
    }
    mTotalTime += cost;
}

static void printTable(const char* title, const std::vector<std::string>& header,
                       const std::vector<std::vector<std::string>>& data) {
    MNN_PRINT("%s\n", title);

    // calc column width
    std::vector<size_t> maxLength(header.size());
    for (int i = 0; i < header.size(); ++i) {
        size_t max = header[i].size();
        for (auto& row : data) {
            max = std::max(max, row[i].size());
        }
        maxLength[i] = max + 1;
    }

    // print header
    for (int i = 0; i < header.size(); ++i) {
        auto expand = header[i];
        expand.resize(maxLength[i], ' ');
        MNN_PRINT("%s\t", expand.c_str());
    }
    MNN_PRINT("\n");
    
    // print rows
    for (auto& row : data) {
        for (int i = 0; i < header.size(); ++i) {
            auto expand = row[i];
            expand.resize(maxLength[i], ' ');
            MNN_PRINT("%s\t", expand.c_str());
        }
        MNN_PRINT("\n");
    }
}
    
void Profiler::printTimeByType(int loops) {
    // sort by time cost
    std::vector<std::pair<float, std::string>> sorted;
    for (auto iter : mMapByType) {
        sorted.push_back(std::make_pair(iter.second.costTime, iter.first));
    }
    std::sort(sorted.begin(), sorted.end());
    
    // fill in columns
    const std::vector<std::string> header = {"Node Type",   "Avg(ms)",   "Dispatch(ms)", "Wait(ms)",
                                             "%",           "Called times", "Flops Rate", "Act BW(GB/s)",
                                             "GFLOPS"};
    std::vector<std::vector<std::string>> rows;
    for (auto iter : sorted) {
        auto record = mMapByType.find(iter.second)->second;
        std::vector<std::string> columns;
        columns.push_back(iter.second);
        columns.push_back(toString(record.costTime / (float)loops));
        columns.push_back(toString(record.dispatchTime / (float)loops));
        columns.push_back(toString(record.waitTime / (float)loops));
        columns.push_back(toString((record.costTime / (float)mTotalTime) * 100));
        columns.push_back(toString(record.calledTimes / loops));
        columns.push_back(toString((record.flops / (float)mTotalMFlops) * 100));
        auto activationBytes = record.inputBytes + record.outputBytes;
        columns.push_back(toString(record.costTime > 0.0f ? (float)activationBytes / record.costTime / 1000000.0f : 0.0f));
        columns.push_back(toString(record.costTime > 0.0f ? record.flops / record.costTime : 0.0f));
        rows.emplace_back(columns);
    }
    printTable("Sort by time cost !", header, rows);
    float totalAvgTime = mTotalTime / (float)loops;
    MNN_PRINT("total time : %f ms, total mflops : %f \n", totalAvgTime, mTotalMFlops / loops);
}

void Profiler::printTimeByName(int loops) {
    const std::vector<std::string> header = {"Node Name", "Op Type", "Avg(ms)", "Dispatch(ms)", "Wait(ms)",
                                             "%", "Flops Rate", "IO Bytes", "Static Bytes", "Act BW(GB/s)",
                                             "Total BW(GB/s)", "GFLOPS", "Execution"};
    std::vector<std::vector<std::string>> rows;
    // sort by name
    for (auto iter: mMapByName) {
        auto record = iter.second;
        std::vector<std::string> columns;
        columns.push_back(iter.first);
        columns.push_back(record.type);
        columns.push_back(toString(record.costTime / (float)loops));
        columns.push_back(toString(record.dispatchTime / (float)loops));
        columns.push_back(toString(record.waitTime / (float)loops));
        columns.push_back(toString((record.costTime / (float)mTotalTime) * 100));
        columns.push_back(toString((record.flops / (float)mTotalMFlops) * 100));
        auto activationBytes = record.inputBytes + record.outputBytes;
        auto staticBytes = getProfileExecutionBytes(iter.first) * (uint64_t)record.calledTimes;
        auto totalBytes = activationBytes + staticBytes;
        columns.push_back(std::to_string(activationBytes / (uint64_t)loops));
        columns.push_back(std::to_string(staticBytes / (uint64_t)loops));
        columns.push_back(toString(record.costTime > 0.0f ? (float)activationBytes / record.costTime / 1000000.0f : 0.0f));
        columns.push_back(toString(record.costTime > 0.0f ? (float)totalBytes / record.costTime / 1000000.0f : 0.0f));
        columns.push_back(toString(record.costTime > 0.0f ? record.flops / record.costTime : 0.0f));
        columns.push_back(getProfileExecutionInfo(iter.first));
        rows.emplace_back(columns);
    }
    printTable("Sort by node name !", header, rows);

    std::vector<std::vector<std::string>> phaseRows;
    for (auto iter : mMapByNamePhase) {
        auto record = iter.second;
        std::vector<std::string> columns;
        auto activationBytes = record.inputBytes + record.outputBytes;
        auto staticBytes = getProfileExecutionBytes(record.name) * (uint64_t)record.calledTimes;
        auto totalBytes = activationBytes + staticBytes;
        columns.push_back(record.name);
        columns.push_back(record.type);
        columns.push_back(record.phase);
        columns.push_back(toString(record.costTime / (float)loops));
        columns.push_back(toString(record.dispatchTime / (float)loops));
        columns.push_back(toString(record.waitTime / (float)loops));
        columns.push_back(toString((record.costTime / (float)mTotalTime) * 100));
        columns.push_back(std::to_string(record.calledTimes / loops));
        columns.push_back(std::to_string(activationBytes / (uint64_t)loops));
        columns.push_back(std::to_string(staticBytes / (uint64_t)loops));
        columns.push_back(toString(record.costTime > 0.0f ? (float)activationBytes / record.costTime / 1000000.0f : 0.0f));
        columns.push_back(toString(record.costTime > 0.0f ? (float)totalBytes / record.costTime / 1000000.0f : 0.0f));
        columns.push_back(toString(record.costTime > 0.0f ? record.flops / record.costTime : 0.0f));
        columns.push_back(record.execution);
        phaseRows.emplace_back(columns);
    }
    const std::vector<std::string> phaseHeader = {"Node Name", "Op Type", "Phase", "Avg(ms)", "Dispatch(ms)",
                                                  "Wait(ms)", "%", "Called times", "IO Bytes", "Static Bytes",
                                                  "Act BW(GB/s)", "Total BW(GB/s)", "GFLOPS", "Execution"};
    printTable("Sort by node name and phase !", phaseHeader, phaseRows);
}

void Profiler::dumpCSV(const std::string& file, int loops) {
    std::ofstream ofs(file.c_str());
    if (!ofs.good()) {
        MNN_ERROR("Can't open profile csv: %s\n", file.c_str());
        return;
    }
    ofs << "node_name,op_type,phase,avg_ms,dispatch_ms,wait_ms,percent,called_times,flops_rate,"
           "io_bytes,static_bytes,act_bw_gbps,total_bw_gbps,gflops,gemm_m,gemm_k,gemm_n,batch,ic,oc,"
           "weight_bits,block_num,kernel,repack_via_int8,online_repack_for_prefill,execution\n";
    for (auto iter : mMapByNamePhase) {
        auto record = iter.second;
        auto activationBytes = record.inputBytes + record.outputBytes;
        auto staticBytes = getProfileExecutionBytes(record.name) * (uint64_t)record.calledTimes;
        auto totalBytes = activationBytes + staticBytes;
        auto execution = record.execution;
        ofs << csvEscape(record.name) << ","
            << csvEscape(record.type) << ","
            << csvEscape(record.phase) << ","
            << (record.costTime / (float)loops) << ","
            << (record.dispatchTime / (float)loops) << ","
            << (record.waitTime / (float)loops) << ","
            << ((record.costTime / (float)mTotalTime) * 100) << ","
            << (record.calledTimes / loops) << ","
            << ((record.flops / (float)mTotalMFlops) * 100) << ","
            << (activationBytes / (uint64_t)loops) << ","
            << (staticBytes / (uint64_t)loops) << ","
            << (record.costTime > 0.0f ? (float)activationBytes / record.costTime / 1000000.0f : 0.0f) << ","
            << (record.costTime > 0.0f ? (float)totalBytes / record.costTime / 1000000.0f : 0.0f) << ","
            << (record.costTime > 0.0f ? record.flops / record.costTime : 0.0f) << ","
            << csvEscape(readField(execution, "gemmM=")) << ","
            << csvEscape(readField(execution, "gemmK=")) << ","
            << csvEscape(readField(execution, "gemmN=")) << ","
            << csvEscape(readField(execution, "batch=")) << ","
            << csvEscape(readField(execution, "ic=")) << ","
            << csvEscape(readField(execution, "oc=")) << ","
            << csvEscape(readField(execution, "weightBits=")) << ","
            << csvEscape(readField(execution, "blockNum=")) << ","
            << csvEscape(readField(execution, "gemmKernel=")) << ","
            << csvEscape(readField(execution, "repackViaInt8=")) << ","
            << csvEscape(readField(execution, "onlineRepackForPrefill=")) << ","
            << csvEscape(execution) << "\n";
    }
    MNN_PRINT("Profile csv saved: %s\n", file.c_str());
}

void Profiler::dumpTimelineCSV(const std::string& file, int loops) {
    std::ofstream ofs(file.c_str());
    if (!ofs.good()) {
        MNN_ERROR("Can't open profile timeline csv: %s\n", file.c_str());
        return;
    }
    std::vector<Event> events = refinePhasesByTimeline();
    ofs << "seq,start_ms,end_ms,node_name,op_type,phase,inferred_phase,total_ms,dispatch_ms,wait_ms,flopsM,io_bytes,static_bytes,"
           "act_bw_gbps,total_bw_gbps,gflops,gemm_m,gemm_k,gemm_n,batch,ic,oc,weight_bits,block_num,kernel,"
           "repack_via_int8,execution\n";
    for (auto event : events) {
        auto activationBytes = (event.inputBytes + event.outputBytes) / (uint64_t)loops;
        auto staticBytes = event.staticBytes / (uint64_t)loops;
        auto totalMs = event.costTime / (float)loops;
        auto flopsM = event.flops / (float)loops;
        auto totalBytes = activationBytes + staticBytes;
        double actBw = totalMs > 0.0f ? (double)activationBytes / totalMs / 1000000.0 : 0.0;
        double totalBw = totalMs > 0.0f ? (double)totalBytes / totalMs / 1000000.0 : 0.0;
        double gflops = totalMs > 0.0f ? flopsM / totalMs : 0.0;
        auto execution = event.execution;
        ofs << event.sequence << ","
            << event.startTime << ","
            << event.endTime << ","
            << csvEscape(event.name) << ","
            << csvEscape(event.type) << ","
            << csvEscape(event.phase) << ","
            << csvEscape(event.inferredPhase) << ","
            << totalMs << ","
            << (event.dispatchTime / (float)loops) << ","
            << (event.waitTime / (float)loops) << ","
            << flopsM << ","
            << activationBytes << ","
            << staticBytes << ","
            << actBw << ","
            << totalBw << ","
            << gflops << ","
            << csvEscape(readField(execution, "gemmM=")) << ","
            << csvEscape(readField(execution, "gemmK=")) << ","
            << csvEscape(readField(execution, "gemmN=")) << ","
            << csvEscape(readField(execution, "batch=")) << ","
            << csvEscape(readField(execution, "ic=")) << ","
            << csvEscape(readField(execution, "oc=")) << ","
            << csvEscape(readField(execution, "weightBits=")) << ","
            << csvEscape(readField(execution, "blockNum=")) << ","
            << csvEscape(readField(execution, "gemmKernel=")) << ","
            << csvEscape(readField(execution, "repackViaInt8=")) << ","
            << csvEscape(execution) << "\n";
    }
    MNN_PRINT("Profile timeline csv saved: %s\n", file.c_str());
}

void Profiler::dumpPhaseCSV(const std::string& file, int loops) {
    std::map<std::string, Record> phaseMap;
    std::vector<Event> events = refinePhasesByTimeline();
    for (auto event : events) {
        auto key = event.type + "\t" + event.name + "\t" + event.inferredPhase + "\t" + event.execution;
        auto& record = phaseMap[key];
        record.name = event.name;
        record.type = event.type;
        record.phase = event.inferredPhase;
        record.execution = event.execution;
        record.calledTimes += 1;
        record.costTime += event.costTime;
        record.dispatchTime += event.dispatchTime;
        record.waitTime += event.waitTime;
        record.flops += event.flops;
        record.inputBytes += event.inputBytes;
        record.outputBytes += event.outputBytes;
    }
    std::vector<Record> rows;
    for (auto iter : phaseMap) {
        rows.emplace_back(iter.second);
    }
    std::sort(rows.begin(), rows.end(), [](const Record& left, const Record& right) {
        return left.costTime > right.costTime;
    });

    std::ofstream ofs(file.c_str());
    if (!ofs.good()) {
        MNN_ERROR("Can't open profile phase csv: %s\n", file.c_str());
        return;
    }
    ofs << "node_name,op_type,inferred_phase,total_ms,dispatch_ms,wait_ms,avg_ms,count,flopsM,"
           "io_bytes,static_bytes,total_bw_gbps,gflops,gemm_m,gemm_k,gemm_n,batch,ic,oc,"
           "weight_bits,block_num,kernel,repack_via_int8,execution\n";
    for (auto record : rows) {
        auto activationBytes = record.inputBytes + record.outputBytes;
        auto staticBytes = getProfileExecutionBytes(record.name) * (uint64_t)record.calledTimes;
        auto totalBytes = activationBytes + staticBytes;
        auto execution = record.execution;
        ofs << csvEscape(record.name) << ","
            << csvEscape(record.type) << ","
            << csvEscape(record.phase) << ","
            << (record.costTime / (float)loops) << ","
            << (record.dispatchTime / (float)loops) << ","
            << (record.waitTime / (float)loops) << ","
            << (record.calledTimes > 0 ? record.costTime / (float)record.calledTimes : 0.0f) << ","
            << (record.calledTimes / loops) << ","
            << (record.flops / (float)loops) << ","
            << (activationBytes / (uint64_t)loops) << ","
            << (staticBytes / (uint64_t)loops) << ","
            << (record.costTime > 0.0f ? (float)totalBytes / record.costTime / 1000000.0f : 0.0f) << ","
            << (record.costTime > 0.0f ? record.flops / record.costTime : 0.0f) << ","
            << csvEscape(readField(execution, "gemmM=")) << ","
            << csvEscape(readField(execution, "gemmK=")) << ","
            << csvEscape(readField(execution, "gemmN=")) << ","
            << csvEscape(readField(execution, "batch=")) << ","
            << csvEscape(readField(execution, "ic=")) << ","
            << csvEscape(readField(execution, "oc=")) << ","
            << csvEscape(readField(execution, "weightBits=")) << ","
            << csvEscape(readField(execution, "blockNum=")) << ","
            << csvEscape(readField(execution, "gemmKernel=")) << ","
            << csvEscape(readField(execution, "repackViaInt8=")) << ","
            << csvEscape(execution) << "\n";
    }
    MNN_PRINT("Profile phase csv saved: %s\n", file.c_str());
}

void Profiler::printSlowOp(const std::string& type, int topK, float rate) {
    MNN_PRINT("Print <=%d slowest Op for %s, larger than %.2f\n", topK, type.c_str(), rate * 100.0f);
    std::vector<std::pair<std::string, float>> result;
    for (auto& iter : mMapByName) {
        if (iter.second.type == type || type.empty()) {
            if (iter.second.flops > 0.0f && iter.second.costTime / mTotalTime >= rate) {
                result.emplace_back(std::make_pair(iter.second.name, iter.second.costTime / iter.second.flops));
            }
        }
    }
    if (result.size() < topK) {
        topK = result.size();
    }
    std::partial_sort(result.begin(), result.begin() + topK, result.end(), [&](const std::pair<std::string, float>& left, std::pair<std::string, float>& right) {
        return left.second > right.second;
    });
    for (int i=0; i<topK; ++i) {
        const auto& record = mMapByName[result[i].first];
        MNN_PRINT("%s -  %f GFlops, %.2f rate\n", record.name.c_str(), record.flops / record.costTime, record.costTime / mTotalTime * 100.0f);
    }
    MNN_PRINT("\n");
}


} // namespace MNN
