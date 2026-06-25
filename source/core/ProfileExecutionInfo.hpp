//
//  ProfileExecutionInfo.hpp
//  MNN
//
//  Created by MNN on 2026/06/24.
//

#ifndef ProfileExecutionInfo_hpp
#define ProfileExecutionInfo_hpp

#include <MNN/MNNDefine.h>

#include <stdint.h>
#include <string>

namespace MNN {

MNN_PUBLIC void setProfileExecutionInfo(const std::string& opName, const std::string& info);
MNN_PUBLIC std::string getProfileExecutionInfo(const std::string& opName);
MNN_PUBLIC std::string getCurrentProfileExecutionInfo(const std::string& opName);
MNN_PUBLIC void setProfileExecutionBytes(const std::string& opName, uint64_t bytes);
MNN_PUBLIC uint64_t getProfileExecutionBytes(const std::string& opName);
MNN_PUBLIC void setCurrentProfilePhase(const std::string& phase);
MNN_PUBLIC std::string getCurrentProfilePhase();
MNN_PUBLIC void clearProfileExecutionInfo();

} // namespace MNN

#endif /* ProfileExecutionInfo_hpp */
