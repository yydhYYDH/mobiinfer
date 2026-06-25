//
//  Profiler.hpp
//  MNN
//
//  Created by MNN on 2019/01/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Profiler_hpp
#define Profiler_hpp

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <vector>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

namespace MNN {

/** Profiler for Ops */
class Profiler {
public:
    /**
     * @brief get shared instance.
     */
    static Profiler* getInstance();
    /**
    * @brief start profiler with op, name and inout tensors.
    * @param op        given op.
    */
    void start(const OperatorInfo* info);
    void start(const std::vector<Tensor*>& inputs, const OperatorInfo* info);
    /**
     * @brief end profiler with op name and type.
     * @param name      op name.
     */
    void end(const OperatorInfo* info);
    void end(const std::vector<Tensor*>& outputs, const OperatorInfo* info);
    /**
     * print profiler time result, grouped by type and sorter by time cost.
     * @param loops     loop count.
     */
    void printTimeByType(int loops = 1);
    /**
     * print profiler time result, grouped and sorter by op name.
     * @param loops     loop count.
     */
    void printTimeByName(int loops = 1);

    void reset();
    void dumpCSV(const std::string& file, int loops = 1);
    void dumpTimelineCSV(const std::string& file, int loops = 1);
    void dumpPhaseCSV(const std::string& file, int loops = 1);

    /**
     * print op that flops / time is slow
     */
    void printSlowOp(const std::string& type, int topk, float limitRate);
private:
    ~Profiler() = default;

private:
    struct Record {
        std::string name;
        std::string type;
        std::string phase;
        std::string execution;
        int64_t order;
        int64_t calledTimes;
        float costTime;
        float dispatchTime;
        float waitTime;
        float flops;
        uint64_t inputBytes;
        uint64_t outputBytes;
    };
    struct Event {
        int64_t sequence;
        std::string name;
        std::string type;
        std::string phase;
        std::string inferredPhase;
        std::string execution;
        float startTime;
        float endTime;
        float costTime;
        float dispatchTime;
        float waitTime;
        float flops;
        uint64_t inputBytes;
        uint64_t outputBytes;
        uint64_t staticBytes;
    };

    static Profiler* gInstance;
    uint64_t mStartTime = 0;
    uint64_t mEndTime   = 0;
    uint64_t mProfileOriginTime = 0;
    float mTotalTime    = 0.0f;
    float mTotalMFlops  = 0.0f;
    std::map<std::string, Record> mMapByType;
    std::map<std::string, Record> mMapByName;
    std::map<std::string, Record> mMapByNamePhase;
    std::string mActiveNamePhaseKey;
    std::vector<Event> mEvents;
    std::map<std::string, int64_t> mActiveEventIndex;
    int64_t mSequence = 0;

private:
    Record& getTypedRecord(const OperatorInfo* info);
    Record& getNamedRecord(const OperatorInfo* info);
    Record& getNamePhaseRecord(const OperatorInfo* info);
    std::vector<Event> refinePhasesByTimeline() const;
};

} // namespace MNN

#endif /* Profiler_hpp */
