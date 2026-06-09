// we pull these symbols into the StrModel namespace for ease of use
// (more liberally than core/using.h)

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <memory>
#include "core/global.h"

namespace StrModel {

using std::string;
using std::vector;
using std::set;
using std::map;
using std::move;
using std::unique_ptr;
using std::make_unique;
using std::shared_ptr;
using std::make_shared;
using std::size_t;
using std::min;
using std::max;
using std::pair;
using std::make_pair;
using std::tie;
using namespace std::string_literals;
using Global::strprintf;

};
