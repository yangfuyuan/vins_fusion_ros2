#ifndef VINS_LOGGER_H_
#define VINS_LOGGER_H_
#include "glog/logging.h"
#include "glog/raw_logging.h"
#define VINS_DEBUG VLOG(4) << "[DEBUG] "
#define VINS_INFO LOG(INFO)
#define VINS_WARN LOG(WARNING)
#define VINS_ERROR LOG(ERROR)
#define VINS_FATAL LOG(FATAL)

#endif  // VINS_LOGGER_H_
