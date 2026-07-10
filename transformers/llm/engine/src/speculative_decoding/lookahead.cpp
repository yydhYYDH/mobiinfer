//
//  lookahead.cpp
//
//  Created by MNN on 2025/04/09.
//

#include "ngram.hpp"
#include "lookahead.hpp"
#include "generate.hpp"
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

//#define DUMP_PROFILE_INFO

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

namespace {

static bool hasSuffix(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool parseInt(const std::string& text, int* value) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}


static const uint32_t kNgramBinaryVersion = 2;
static const char kNgramBinaryMagic[8] = {'M', 'N', 'N', 'N', 'G', 'R', 'M', '2'};
static const uint32_t kNgramHashBinaryVersion = 3;
static const uint32_t kNgramHashBinaryFlagsTop1 = 1;
static const uint64_t kNgramHashSeed = 1469598103934665603ull;
static const uint64_t kNgramHashPrime = 1099511628211ull;
static const char kNgramHashBinaryMagic[8] = {'M', 'N', 'N', 'N', 'G', 'R', 'M', '3'};

static bool isMmapHashNgramTable(const std::string& path) {
    return hasSuffix(path, ".mnnngram3");
}

static bool isBinaryNgramTable(const std::string& path) {
    return hasSuffix(path, ".bin") || hasSuffix(path, ".mnnngram");
}

class ReadOnlyMappedFile {
public:
    explicit ReadOnlyMappedFile(const std::string& path) {
#ifndef _WIN32
        mFd = ::open(path.c_str(), O_RDONLY);
        if (mFd < 0) {
            return;
        }
        struct stat st;
        if (::fstat(mFd, &st) != 0 || st.st_size <= 0) {
            close();
            return;
        }
        mSize = static_cast<size_t>(st.st_size);
        void* mapped = ::mmap(nullptr, mSize, PROT_READ, MAP_PRIVATE, mFd, 0);
        if (mapped == MAP_FAILED) {
            close();
            return;
        }
        mData = static_cast<const uint8_t*>(mapped);
        mMapped = true;
#else
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return;
        }
        file.seekg(0, std::ios::end);
        auto size = file.tellg();
        if (size <= 0) {
            return;
        }
        mBuffer.resize(static_cast<size_t>(size));
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char*>(mBuffer.data()), size);
        if (file.gcount() != size) {
            mBuffer.clear();
            return;
        }
        mData = mBuffer.data();
        mSize = mBuffer.size();
#endif
    }

    ~ReadOnlyMappedFile() {
        close();
    }

    const uint8_t* data() const {
        return mData;
    }

    size_t size() const {
        return mSize;
    }

    bool valid() const {
        return mData != nullptr && mSize > 0;
    }

private:
    void close() {
#ifndef _WIN32
        if (mMapped && mData != nullptr) {
            ::munmap(const_cast<uint8_t*>(mData), mSize);
        }
        if (mFd >= 0) {
            ::close(mFd);
        }
        mFd = -1;
        mMapped = false;
#else
        mBuffer.clear();
#endif
        mData = nullptr;
        mSize = 0;
    }

    const uint8_t* mData = nullptr;
    size_t mSize = 0;
#ifndef _WIN32
    int mFd = -1;
    bool mMapped = false;
#else
    std::vector<uint8_t> mBuffer;
#endif
};

static bool readU32(const uint8_t*& ptr, const uint8_t* end, uint32_t* value) {
    if (ptr + sizeof(uint32_t) > end) {
        return false;
    }
    uint32_t v = 0;
    v |= static_cast<uint32_t>(ptr[0]);
    v |= static_cast<uint32_t>(ptr[1]) << 8;
    v |= static_cast<uint32_t>(ptr[2]) << 16;
    v |= static_cast<uint32_t>(ptr[3]) << 24;
    ptr += sizeof(uint32_t);
    *value = v;
    return true;
}

static bool readI32(const uint8_t*& ptr, const uint8_t* end, int* value) {
    uint32_t raw = 0;
    if (!readU32(ptr, end, &raw)) {
        return false;
    }
    *value = static_cast<int32_t>(raw);
    return true;
}


static bool readU64(const uint8_t*& ptr, const uint8_t* end, uint64_t* value) {
    if (ptr + sizeof(uint64_t) > end) {
        return false;
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= static_cast<uint64_t>(ptr[i]) << (8 * i);
    }
    ptr += sizeof(uint64_t);
    *value = v;
    return true;
}

static bool readU32At(const uint8_t* data, size_t size, uint64_t offset, uint32_t* value) {
    if (offset > size || size - static_cast<size_t>(offset) < sizeof(uint32_t)) {
        return false;
    }
    const uint8_t* ptr = data + static_cast<size_t>(offset);
    uint32_t v = 0;
    v |= static_cast<uint32_t>(ptr[0]);
    v |= static_cast<uint32_t>(ptr[1]) << 8;
    v |= static_cast<uint32_t>(ptr[2]) << 16;
    v |= static_cast<uint32_t>(ptr[3]) << 24;
    *value = v;
    return true;
}

static bool readI32At(const uint8_t* data, size_t size, uint64_t offset, int* value) {
    uint32_t raw = 0;
    if (!readU32At(data, size, offset, &raw)) {
        return false;
    }
    *value = static_cast<int32_t>(raw);
    return true;
}

static uint64_t hashNgramKey(const ngram_key& key, int keyLen) {
    uint64_t hash = kNgramHashSeed;
    for (int i = 0; i < keyLen; i++) {
        hash ^= static_cast<uint32_t>(key.keys[i]);
        hash *= kNgramHashPrime;
    }
    return hash;
}

class MmapHashNgramTable {
public:
    bool load(const std::string& path, int maxKeyLen) {
        mFile.reset(new ReadOnlyMappedFile(path));
        if (!mFile->valid()) {
            mFile.reset();
            return false;
        }
        const uint8_t* ptr = mFile->data();
        const uint8_t* end = mFile->data() + mFile->size();
        if (ptr + 40 > end || std::memcmp(ptr, kNgramHashBinaryMagic, sizeof(kNgramHashBinaryMagic)) != 0) {
            MNN_PRINT("Warning: invalid mmap hash ngram table header: %s\n", path.c_str());
            mFile.reset();
            return false;
        }
        ptr += sizeof(kNgramHashBinaryMagic);
        uint32_t version = 0;
        uint32_t flags = 0;
        uint32_t storedMaxKeyLen = 0;
        uint32_t sectionCount = 0;
        uint32_t descOffset = 0;
        uint32_t dataOffset = 0;
        uint32_t hashSeedLow = 0;
        uint32_t reserved = 0;
        if (!readU32(ptr, end, &version) || !readU32(ptr, end, &flags) || !readU32(ptr, end, &storedMaxKeyLen) ||
            !readU32(ptr, end, &sectionCount) || !readU32(ptr, end, &descOffset) || !readU32(ptr, end, &dataOffset) ||
            !readU32(ptr, end, &hashSeedLow) || !readU32(ptr, end, &reserved)) {
            mFile.reset();
            return false;
        }
        if (version != kNgramHashBinaryVersion || (flags & kNgramHashBinaryFlagsTop1) == 0 || storedMaxKeyLen == 0 ||
            storedMaxKeyLen > MNN_NGRAM_KEY_MAX || sectionCount > MNN_NGRAM_KEY_MAX) {
            MNN_PRINT("Warning: unsupported mmap hash ngram table header: %s\n", path.c_str());
            mFile.reset();
            return false;
        }
        (void)dataOffset;
        (void)hashSeedLow;
        (void)reserved;
        const size_t sectionStride = 40;
        if (descOffset > mFile->size() || mFile->size() - descOffset < sectionCount * sectionStride) {
            mFile.reset();
            return false;
        }
        int loaded = 0;
        mMaxKeyLen = std::min(static_cast<int>(storedMaxKeyLen), maxKeyLen);
        for (uint32_t i = 0; i < sectionCount; i++) {
            const uint8_t* sptr = mFile->data() + descOffset + i * sectionStride;
            uint32_t n = 0;
            uint32_t entryCount = 0;
            uint32_t bucketCount = 0;
            uint32_t entryStride = 0;
            uint64_t bucketOffsetsOffset = 0;
            uint64_t entriesOffset = 0;
            uint32_t sectionFlags = 0;
            uint32_t sectionReserved = 0;
            if (!readU32(sptr, end, &n) || !readU32(sptr, end, &entryCount) || !readU32(sptr, end, &bucketCount) ||
                !readU32(sptr, end, &entryStride) || !readU64(sptr, end, &bucketOffsetsOffset) ||
                !readU64(sptr, end, &entriesOffset) || !readU32(sptr, end, &sectionFlags) ||
                !readU32(sptr, end, &sectionReserved)) {
                mFile.reset();
                return false;
            }
            (void)sectionFlags;
            (void)sectionReserved;
            if (n == 0 || n > static_cast<uint32_t>(mMaxKeyLen) || bucketCount == 0 ||
                (bucketCount & (bucketCount - 1)) != 0 || entryStride != sizeof(int32_t) * (n + 1)) {
                continue;
            }
            uint64_t offsetsBytes = static_cast<uint64_t>(bucketCount + 1) * sizeof(uint32_t);
            uint64_t entriesBytes = static_cast<uint64_t>(entryCount) * entryStride;
            if (bucketOffsetsOffset > mFile->size() || offsetsBytes > mFile->size() - bucketOffsetsOffset ||
                entriesOffset > mFile->size() || entriesBytes > mFile->size() - entriesOffset) {
                continue;
            }
            Section& section = mSections[n];
            section.valid = true;
            section.n = n;
            section.entryCount = entryCount;
            section.bucketCount = bucketCount;
            section.entryStride = entryStride;
            section.bucketOffsetsOffset = bucketOffsetsOffset;
            section.entriesOffset = entriesOffset;
            loaded += entryCount;
        }
        if (loaded <= 0) {
            mFile.reset();
            return false;
        }
        MNN_PRINT("Loaded %d mmap hash ngram table entries from %s\n", loaded, path.c_str());
        return true;
    }

    bool lookup(const ngram_key& key, int keyLen, int* nextToken) const {
        if (!mFile || keyLen <= 0 || keyLen > MNN_NGRAM_KEY_MAX) {
            return false;
        }
        const Section& section = mSections[keyLen];
        if (!section.valid) {
            return false;
        }
        uint32_t bucket = static_cast<uint32_t>(hashNgramKey(key, keyLen) & (section.bucketCount - 1));
        uint32_t begin = 0;
        uint32_t end = 0;
        uint64_t offsetsPos = section.bucketOffsetsOffset + static_cast<uint64_t>(bucket) * sizeof(uint32_t);
        if (!readU32At(mFile->data(), mFile->size(), offsetsPos, &begin) ||
            !readU32At(mFile->data(), mFile->size(), offsetsPos + sizeof(uint32_t), &end) || begin > end ||
            end > section.entryCount) {
            return false;
        }
        for (uint32_t i = begin; i < end; i++) {
            uint64_t row = section.entriesOffset + static_cast<uint64_t>(i) * section.entryStride;
            bool matched = true;
            for (int j = 0; j < keyLen; j++) {
                int token = 0;
                if (!readI32At(mFile->data(), mFile->size(), row + static_cast<uint64_t>(j) * sizeof(int32_t), &token) ||
                    token != key.keys[j]) {
                    matched = false;
                    break;
                }
            }
            if (matched) {
                return readI32At(mFile->data(), mFile->size(), row + static_cast<uint64_t>(keyLen) * sizeof(int32_t),
                                 nextToken);
            }
        }
        return false;
    }

private:
    struct Section {
        bool valid = false;
        uint32_t n = 0;
        uint32_t entryCount = 0;
        uint32_t bucketCount = 0;
        uint32_t entryStride = 0;
        uint64_t bucketOffsetsOffset = 0;
        uint64_t entriesOffset = 0;
    };

    std::unique_ptr<ReadOnlyMappedFile> mFile;
    Section mSections[MNN_NGRAM_KEY_MAX + 1];
    int mMaxKeyLen = 0;
};

static void mmapHashNgramSearch(const MmapHashNgramTable& table, int ngramKeyMin, int ngramKeyMax,
                                const std::vector<int>& tokens, std::vector<int>& drafts, int maxDraftSize) {
    MNN_ASSERT(drafts.size() == 1);
    while (drafts.size() < maxDraftSize) {
        int tokenDraft = -1;
        for (int ngramKeySize = ngramKeyMax; ngramKeySize >= ngramKeyMin; ngramKeySize--) {
            int ngramStartIdx = static_cast<int>(tokens.size()) - ngramKeySize + static_cast<int>(drafts.size());
            ngram_key ngram;
            int i = ngramStartIdx;
            if (i < 0) {
                i = 0;
            }
            for (; i < static_cast<int>(tokens.size()); i++) {
                ngram.keys[i - ngramStartIdx] = tokens[i];
            }
            for (; i < ngramStartIdx + ngramKeySize; i++) {
                ngram.keys[i - ngramStartIdx] = drafts[i - static_cast<int>(tokens.size())];
            }
            if (table.lookup(ngram, ngramKeySize, &tokenDraft)) {
                break;
            }
        }
        if (tokenDraft == -1) {
            break;
        }
        drafts.push_back(tokenDraft);
    }
    if (drafts.size() > 1) {
        for (int i = static_cast<int>(drafts.size()); i < maxDraftSize; i++) {
            drafts.push_back(0);
        }
    }
}

struct BinaryNgramEntry {
    ngram_key key;
    int keyLen = 0;
    int nextToken = 0;
    int count = 0;
};

static bool readBinaryNgramEntry(const uint8_t*& ptr, const uint8_t* end, int storedKeySlots,
                                  BinaryNgramEntry* entry) {
    uint32_t keyLen = 0;
    if (!readU32(ptr, end, &keyLen) || keyLen == 0 || keyLen > static_cast<uint32_t>(storedKeySlots) ||
        keyLen > MNN_NGRAM_KEY_MAX) {
        return false;
    }
    ngram_key parsedKey;
    for (int i = 0; i < storedKeySlots; i++) {
        int token = 0;
        if (!readI32(ptr, end, &token)) {
            return false;
        }
        if (i < static_cast<int>(keyLen)) {
            parsedKey.keys[i] = token;
        }
    }
    int nextToken = 0;
    int count = 0;
    if (!readI32(ptr, end, &nextToken) || !readI32(ptr, end, &count) || count <= 0) {
        return false;
    }
    entry->key = parsedKey;
    entry->keyLen = static_cast<int>(keyLen);
    entry->nextToken = nextToken;
    entry->count = count;
    return true;
}

template <typename Callback>
static int loadBinaryNgramTableImpl(const std::string& path, int maxKeyLen, const Callback& callback) {
    ReadOnlyMappedFile file(path);
    if (!file.valid()) {
        return 0;
    }
    const uint8_t* ptr = file.data();
    const uint8_t* end = file.data() + file.size();
    if (ptr + 24 > end || std::memcmp(ptr, kNgramBinaryMagic, sizeof(kNgramBinaryMagic)) != 0) {
        MNN_PRINT("Warning: invalid binary ngram table header: %s\n", path.c_str());
        return 0;
    }
    ptr += sizeof(kNgramBinaryMagic);
    uint32_t version = 0;
    uint32_t storedMaxKeyLen = 0;
    uint32_t reserved = 0;
    uint32_t entryCount = 0;
    if (!readU32(ptr, end, &version) || !readU32(ptr, end, &storedMaxKeyLen) || !readU32(ptr, end, &reserved) ||
        !readU32(ptr, end, &entryCount)) {
        return 0;
    }
    if (version != kNgramBinaryVersion) {
        MNN_PRINT("Warning: unsupported binary ngram table version %u: %s\n", version, path.c_str());
        return 0;
    }
    if (storedMaxKeyLen == 0 || storedMaxKeyLen > MNN_NGRAM_KEY_MAX) {
        MNN_PRINT("Warning: invalid binary ngram table max key len %u: %s\n", storedMaxKeyLen, path.c_str());
        return 0;
    }
    (void)reserved;
    int loaded = 0;
    int storedKeySlots = static_cast<int>(storedMaxKeyLen);
    for (uint32_t i = 0; i < entryCount; i++) {
        BinaryNgramEntry entry;
        if (!readBinaryNgramEntry(ptr, end, storedKeySlots, &entry)) {
            break;
        }
        if (entry.keyLen > maxKeyLen) {
            continue;
        }
        callback(entry);
        loaded++;
    }
    if (loaded > 0) {
        MNN_PRINT("Loaded %d binary prebuilt ngram table entries from %s\n", loaded, path.c_str());
    }
    return loaded;
}

static int loadBinaryNgramTable(ngram_cache<ngram_value>& data, const std::string& path, int maxKeyLen) {
    return loadBinaryNgramTableImpl(path, maxKeyLen, [&data](const BinaryNgramEntry& entry) {
        data[entry.key][entry.nextToken] += entry.count;
    });
}

static int loadBinaryNgramTable(ngram_cache<ngram_ordered_value>& data, const std::string& path, int maxKeyLen) {
    return loadBinaryNgramTableImpl(path, maxKeyLen, [&data](const BinaryNgramEntry& entry) {
        data[entry.key].push_back(entry.nextToken);
    });
}

static bool parseNgramTableLine(const std::string& line, int maxKeyLen, ngram_key* key, int* nextToken, int* count) {
    if (line.empty() || line[0] == '#') {
        return false;
    }
    if (line.compare(0, 2, "n\t") == 0) {
        return false;
    }
    size_t tab0 = line.find('\t');
    if (tab0 == std::string::npos) {
        return false;
    }
    size_t tab1 = line.find('\t', tab0 + 1);
    if (tab1 == std::string::npos) {
        return false;
    }
    size_t tab2 = line.find('\t', tab1 + 1);
    if (tab2 == std::string::npos) {
        return false;
    }

    int keyLen = 0;
    if (!parseInt(line.substr(0, tab0), &keyLen) || keyLen <= 0 || keyLen > maxKeyLen || keyLen > MNN_NGRAM_KEY_MAX) {
        return false;
    }
    if (!parseInt(line.substr(tab1 + 1, tab2 - tab1 - 1), nextToken) || !parseInt(line.substr(tab2 + 1), count) ||
        *count <= 0) {
        return false;
    }

    ngram_key parsedKey;
    size_t start = tab0 + 1;
    int parsedLen = 0;
    while (start <= tab1 && parsedLen < keyLen) {
        size_t comma = line.find(',', start);
        size_t end = (comma == std::string::npos || comma > tab1) ? tab1 : comma;
        int token = 0;
        if (!parseInt(line.substr(start, end - start), &token)) {
            return false;
        }
        parsedKey.keys[parsedLen++] = token;
        if (end == tab1) {
            break;
        }
        start = end + 1;
    }
    if (parsedLen != keyLen) {
        return false;
    }
    *key = parsedKey;
    return true;
}

static int loadNgramTable(ngram_cache<ngram_value>& data, const std::string& path, int maxKeyLen) {
    if (path.empty()) {
        return 0;
    }
    if (isBinaryNgramTable(path)) {
        return loadBinaryNgramTable(data, path, maxKeyLen);
    }
    if (hasSuffix(path, ".gz")) {
        MNN_PRINT("Warning: ngram_table_file only supports plain TSV or binary .bin/.mnnngram, not gzip: %s\n", path.c_str());
        return 0;
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }
    std::string line;
    int loaded = 0;
    while (std::getline(file, line)) {
        ngram_key key;
        int nextToken = 0;
        int count = 0;
        if (!parseNgramTableLine(line, maxKeyLen, &key, &nextToken, &count)) {
            continue;
        }
        data[key][nextToken] += count;
        loaded++;
    }
    if (loaded > 0) {
        MNN_PRINT("Loaded %d prebuilt ngram table entries from %s\n", loaded, path.c_str());
    }
    return loaded;
}

static int loadNgramTable(ngram_cache<ngram_ordered_value>& data, const std::string& path, int maxKeyLen) {
    if (path.empty()) {
        return 0;
    }
    if (isBinaryNgramTable(path)) {
        return loadBinaryNgramTable(data, path, maxKeyLen);
    }
    if (hasSuffix(path, ".gz")) {
        MNN_PRINT("Warning: ngram_table_file only supports plain TSV or binary .bin/.mnnngram, not gzip: %s\n", path.c_str());
        return 0;
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }
    std::string line;
    int loaded = 0;
    while (std::getline(file, line)) {
        ngram_key key;
        int nextToken = 0;
        int count = 0;
        if (!parseNgramTableLine(line, maxKeyLen, &key, &nextToken, &count)) {
            continue;
        }
        data[key].push_back(nextToken);
        loaded++;
    }
    if (loaded > 0) {
        MNN_PRINT("Loaded %d prebuilt ngram table entries from %s\n", loaded, path.c_str());
    }
    return loaded;
}

} // namespace


struct LookaheadNgramCache {
    ngram_cache<ngram_value> freqCache;
    ngram_cache<ngram_ordered_value> orderedCache;
    std::unique_ptr<MmapHashNgramTable> mmapHashCache;
};

LookaheadGeneration::LookaheadGeneration(Llm* llm, std::shared_ptr<LlmContext> context,
                                         std::shared_ptr<LlmConfig> config)
    : Generation(llm, context) {
    mNgramKeyMaxLen = config->ngram_match_maxlen();
    if (mNgramKeyMaxLen > 8) {
        MNN_PRINT("Warning: ngram match key length maybe too large!\n");
    }
    auto strictness = config->draft_match_strictness();
    mStrictLevel = MatchStrictLevel::LOW_LEVEL;
    if (strictness == "high") {
        mStrictLevel = MatchStrictLevel::HIGH_LEVEL;
    } else if (strictness == "medium") {
        mStrictLevel = MatchStrictLevel::MEDIUM_LEVEL;
    } else if (strictness == "low") {
        mStrictLevel = MatchStrictLevel::LOW_LEVEL;
    } else {
        MNN_PRINT("Warning: draft_match_strictness value set error!, use default param instead\n");
    }

    auto selectRule = config->draft_selection_rule();
    mSelectRule = NgramSelectRule::FreqxLen_RULE;
    if (selectRule == "fcfs") {
        mSelectRule = NgramSelectRule::FCFS_RULE;
    } else if (selectRule == "freqxlen") {
        mSelectRule = NgramSelectRule::FreqxLen_RULE;
    } else {
        MNN_PRINT("Warning: draft_selection_rule value set error!, use default param instead\n");
    }
    mUpdateNgram = config->ngram_update();

    auto ngramTableFile = config->ngram_table_file();
    mPrebuiltNgram.reset(new LookaheadNgramCache);
    if (isMmapHashNgramTable(ngramTableFile)) {
        if (mSelectRule == NgramSelectRule::FCFS_RULE) {
            mPrebuiltNgram->mmapHashCache.reset(new MmapHashNgramTable);
            mHasPrebuiltNgram = mPrebuiltNgram->mmapHashCache->load(ngramTableFile, mNgramKeyMaxLen);
        } else {
            MNN_PRINT("Warning: mmap hash ngram table only supports fcfs/top1 lookup: %s\n", ngramTableFile.c_str());
        }
    } else if (mSelectRule == NgramSelectRule::FreqxLen_RULE) {
        mHasPrebuiltNgram = loadNgramTable(mPrebuiltNgram->freqCache, ngramTableFile, mNgramKeyMaxLen) > 0;
    } else {
        mHasPrebuiltNgram = loadNgramTable(mPrebuiltNgram->orderedCache, ngramTableFile, mNgramKeyMaxLen) > 0;
    }
    if (!mHasPrebuiltNgram) {
        mPrebuiltNgram.reset();
    }
}

void LookaheadGeneration::generate(GenerationParams& param) {
    if (-1 == mContext->current_token) {
        mContext->current_token = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
    }
    int max_token = param.max_new_tokens;
    int len = 0;
    ngram_cache<ngram_value> prompt_ngram_cache;
    ngram_cache<ngram_ordered_value> prompt_ngram_ordered_cache;
    ngram_cache<ngram_value>* activeNgramCache = &prompt_ngram_cache;
    ngram_cache<ngram_ordered_value>* activeNgramOrderedCache = &prompt_ngram_ordered_cache;

    if (mHasPrebuiltNgram) {
        activeNgramCache = &mPrebuiltNgram->freqCache;
        activeNgramOrderedCache = &mPrebuiltNgram->orderedCache;
    } else if (mSelectRule == NgramSelectRule::FreqxLen_RULE) {
        ngram_cache_update(prompt_ngram_cache, 1, mNgramKeyMaxLen, mContext->history_tokens,
                           mContext->history_tokens.size());
    } else {
        ngram_cache_update(prompt_ngram_ordered_cache, 1, mNgramKeyMaxLen, mContext->history_tokens,
                           mContext->history_tokens.size());
    }

    // user provided text info to create ngram. Used as fallback when no prebuilt token-id table is loaded.
    if (!mHasPrebuiltNgram) {
        auto prior_prompt_file = mLlm->mConfig->lookup_file();
        std::ifstream file(prior_prompt_file);

        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string user_set_prompt = buffer.str();
            file.close();
            std::vector<int> user_ids = mLlm->tokenizer_encode(user_set_prompt);

            if (mSelectRule == NgramSelectRule::FreqxLen_RULE) {
                ngram_cache_update(prompt_ngram_cache, 1, mNgramKeyMaxLen, user_ids, user_ids.size());
            } else {
                ngram_cache_update(prompt_ngram_ordered_cache, 1, mNgramKeyMaxLen, user_ids, user_ids.size());
            }
        }
    }
    bool stop = false;
    // speculative total decode token numbers
    int spl_decode = 0;
    // speculative accept token numbers
    int spl_accept = 0;
    // autoregression number of times
    int arg_count = 0;
    // speculative number of times
    int spl_count = 0;
    int verify_len = mLlm->mDraftLength + 1;
    bool debugStats = mLlm->mConfig->lookahead_debug_stats();
    bool debugStatsDetail = mLlm->mConfig->lookahead_debug_stats_detail();
    int statSteps = 0;
    int statSpecSteps = 0;
    int statArSteps = 0;
    int statDraftTokens = 0;
    int statAcceptedDraftTokens = 0;
    int statAcceptedTokens = 0;
    int statFullAcceptSteps = 0;
    while (len < max_token) {
        if (mContext->status == LlmStatus::USER_CANCEL || mContext->status == LlmStatus::INTERNAL_ERROR) {
            break;
        }
        if (param.timeout_ms > 0 && (mContext->prefill_us + mContext->decode_us) / 1000 >= param.timeout_ms) {
            mContext->status = LlmStatus::TIMEOUT;
            break;
        }
        MNN::Timer _t;
        std::vector<int> drafts;
        drafts.push_back(mContext->current_token);

        auto decodeStr = mLlm->tokenizer_decode(mContext->current_token);
        mContext->generate_str += decodeStr;
        if (nullptr != mContext->os) {
            *mContext->os << decodeStr;
            *mContext->os << std::flush;
        }
        // mContext->current_token add to gen_seq_len
        mLlm->updateContext(0, 1);

        {
            // draft is "," or "." or ":" or "、", match maybe confusion
            bool confuse = decodeStr == "," || decodeStr == "." || decodeStr == ":" || decodeStr == "、" ||
                           decodeStr == ";" || decodeStr == "，" || decodeStr == "。" ||
                           decodeStr == "：" || // Chinese comma and semicolon
                           decodeStr == "、" || decodeStr == "；";
            MatchStrictLevel level = mStrictLevel;
            // for confuse key, set match_strictness to high
            if (confuse) {
                level = MatchStrictLevel::HIGH_LEVEL;
            }
            // generate draft tokens
            if (mHasPrebuiltNgram && mPrebuiltNgram->mmapHashCache) {
                mmapHashNgramSearch(*mPrebuiltNgram->mmapHashCache, 1, mNgramKeyMaxLen, mContext->history_tokens,
                                    drafts, verify_len);
            } else if (mSelectRule == NgramSelectRule::FreqxLen_RULE) {
                ngram_cache_search(*activeNgramCache, 1, mNgramKeyMaxLen, mContext->history_tokens, drafts, verify_len,
                                   level);
            } else {
                ngram_cache_search(*activeNgramOrderedCache, 1, mNgramKeyMaxLen, mContext->history_tokens, drafts,
                                   verify_len, level);
            }

            mLlm->mMeta->add = drafts.size();

            AUTOTIME;
            // do draft token parallel verify
            auto outputs = mLlm->forwardVec(drafts);
            if (outputs.empty()) {
                break;
            }
            auto logits = outputs[0];
            if (nullptr == logits.get()) {
                break;
            }
            if (logits->getInfo()->size == 0) {
                break;
            }

            // verify draft token whether be accepted
            int i_dft = draftVerify(logits, drafts, stop);

            // update ngram for each decoding
            if (!stop && mUpdateNgram) {
                if (mSelectRule == NgramSelectRule::FreqxLen_RULE) {
                    ngram_cache_update(*activeNgramCache, 1, mNgramKeyMaxLen, mContext->history_tokens, i_dft);
                } else {
                    ngram_cache_update(*activeNgramOrderedCache, 1, mNgramKeyMaxLen, mContext->history_tokens, i_dft);
                }
            }

            if (debugStats) {
                int draftTokenNum = static_cast<int>(drafts.size()) - 1;
                int acceptedDraftNum = i_dft - 1;
                statSteps++;
                statAcceptedTokens += i_dft;
                if (draftTokenNum > 0) {
                    statSpecSteps++;
                    statDraftTokens += draftTokenNum;
                    statAcceptedDraftTokens += acceptedDraftNum;
                    if (i_dft == static_cast<int>(drafts.size())) {
                        statFullAcceptSteps++;
                    }
                } else {
                    statArSteps++;
                }
                if (debugStatsDetail) {
                    std::fprintf(stderr,
                                 "[LOOKAHEAD_STATS] step=%d drafts=%d accepted=%d draft_tokens=%d "
                                 "accepted_draft_tokens=%d full_accept=%d current=%d\n",
                                 statSteps, static_cast<int>(drafts.size()), i_dft, draftTokenNum,
                                 acceptedDraftNum, i_dft == static_cast<int>(drafts.size()),
                                 mContext->current_token);
                }
            }

            // clear dirty kv-cache
            mLlm->mMeta->remove = drafts.size() - i_dft;
            len += i_dft;

            // update context state
            int seq_len = i_dft;
            int gen_len = i_dft - 1; // current_token has been added, add others
            mLlm->updateContext(seq_len, gen_len);

            // count time cost
            mContext->decode_us += _t.durationInUs();

            // add all accept tokens to string
            mContext->history_tokens.insert(mContext->history_tokens.end(), drafts.begin(), drafts.begin() + i_dft);
            mContext->output_tokens.insert(mContext->output_tokens.end(), drafts.begin(), drafts.begin() + i_dft);

#ifdef DUMP_PROFILE_INFO
            MNN_PRINT("\ndraft num:%d, adopt num:%d\n", drafts.size(), i_dft);
            if (drafts.size() > 1) {
                spl_decode += drafts.size();
                spl_accept += i_dft;
                spl_count++;
            } else {
                arg_count++;
            }
#endif
            if (stop) {
                mContext->history_tokens.push_back(mContext->current_token);
                mContext->output_tokens.push_back(mContext->current_token);
                mLlm->updateContext(0, 1);
                break;
            }
            if (mLlm->is_stop(mContext->current_token)) {
                mContext->history_tokens.push_back(mContext->current_token);
                mContext->output_tokens.push_back(mContext->current_token);
                mLlm->updateContext(0, 1);
                if (nullptr != mContext->os) {
                    *mContext->os << mContext->end_with << std::flush;
                }
                break;
            }
        }
    }
    if (len >= max_token) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
    if (debugStats) {
        float specStepRate = statSteps > 0 ? 100.0f * statSpecSteps / statSteps : 0.0f;
        float draftAcceptRate = statDraftTokens > 0 ? 100.0f * statAcceptedDraftTokens / statDraftTokens : 0.0f;
        float fullAcceptRate = statSpecSteps > 0 ? 100.0f * statFullAcceptSteps / statSpecSteps : 0.0f;
        float avgAccepted = statSteps > 0 ? 1.0f * statAcceptedTokens / statSteps : 0.0f;
        std::fprintf(stderr,
                     "[LOOKAHEAD_STATS] summary steps=%d spec_steps=%d ar_steps=%d "
                     "spec_step_rate=%.2f%% draft_tokens=%d accepted_draft_tokens=%d "
                     "draft_accept_rate=%.2f%% full_accept_steps=%d full_accept_rate=%.2f%% "
                     "accepted_tokens=%d avg_accepted_per_step=%.2f\n",
                     statSteps, statSpecSteps, statArSteps, specStepRate, statDraftTokens,
                     statAcceptedDraftTokens, draftAcceptRate, statFullAcceptSteps, fullAcceptRate,
                     statAcceptedTokens, avgAccepted);
    }
#ifdef DUMP_PROFILE_INFO
    // adopt speculative decoding rate
    float spl_rate = 100.0 * spl_count / (spl_count + arg_count);
    // draft accept rate if adopt speculative decoding
    float spl_accept_rate = 100.0 * spl_accept / spl_decode;

    MNN_PRINT("\n============== Speculative Decoding Statistics Start ===============\n");
    MNN_PRINT("Adopt speculative decode rate: %.2f%%\n", spl_rate);
    MNN_PRINT("Average speculative decode accept rate: %.2f%%\n", spl_accept_rate);

    // speculative decoding vs autoregressive decoding cost time rate
    // assume add 10% time with every additional token decoding
    float spl_cost_rate = 1.0 + (verify_len * 0.1);
    // original autoregressive decoding cost times
    float arg_time = spl_accept + arg_count;
    // speculative decoding cost times
    float spl_time = spl_count * spl_cost_rate + arg_count;
    float speed_up = 1.0 * arg_time / spl_time;
    MNN_PRINT("Verify length is: %d\n", verify_len);
    MNN_PRINT("Assume decode %d token is %.2f times consumption than single token decoding\n", verify_len,
              spl_cost_rate);
    MNN_PRINT("Total speed up is around: %.2fx\n", speed_up);
    MNN_PRINT("============== Speculative Decoding Statistics End =================\n");

#endif
    return;
}

} // namespace Transformer
} // namespace MNN
