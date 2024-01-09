#include <iostream>
#include <string>
#include <vector>
#include <map>

class Emitter {

std::ostream &out;
std::map<std::string, int> handlers;
std::map<std::string, int> handlerReplications;
std::map<std::string, std::vector<int>> externs;

static constexpr auto header = R"(
#ifndef _EP2_DEFS_
#define _EP2_DEFS_

#include "runtime.h"
)";

static constexpr auto footer = R"(
#endif // _EP2_DEFS_
)";

void emitDefine(std::string name, int value) {
    out << "#define " << name << " (" << value << ")\n";
}

void emitHandlers() {
    for (auto &[name,in] : handlers) {
        out << "void " << name << "(void * event);\n";
        out << "WORKER_FUNCTION(" << in << "," << name << ");\n";
    }

    out << "handler_worker_t handler_workers[NUM_HANDLERS] = {\n";
    for (auto &[name,in] : handlers)
        out << "    __thread" << name << ",\n";
    out << "};\n";
}

void emitHandlerInstances() {
    out << "int handler_replications[NUM_HANDLERS] = {";
    for (auto &[name,in] : handlerReplications)
        out << in << ",";
    out << "};\n";
}

int instances = 0;
void init() {
    for (auto &[name,in] : handlerReplications)
        instances += in;
}

void emitExtern() {

}

void emit() {
    init();
    out << header;

    emitDefine("NUM_QUEUES", 2);
    emitDefine("NUM_HANDLERS", handlers.size());
    emitDefine("NUM_INSTANCES", instances);

    emitHandlerInstances();

    emitHandlers();

    emitDefine("NUM_EXTERNS", externs.size());
    emitExtern();

    out << footer;
}

};