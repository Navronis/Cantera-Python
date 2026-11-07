#include <iostream>
#include <vector>
#include "cantera/base/AnyMap.h"

int main() {
    std::string jsonStr = "{\"sbd\": [[1.0, 2.0], [3.0, 4.0]]}";
    auto m = Cantera::AnyMap::fromYamlString(jsonStr);
    auto sbd = m["sbd"].asVector<std::vector<double>>();
    std::cout << "sbd size: " << sbd.size() << ", sbd[1][0]=" << sbd[1][0] << std::endl;
    return 0;
}
