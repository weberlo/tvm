# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_MICRO)
  message(STATUS "Build with Micro support")
  file(GLOB RUNTIME_MICRO_SRCS src/runtime/micro/*.cc)
  list(APPEND RUNTIME_MICRO_SRCS
      3rdparty/mbed-os/targets/TARGET_NORDIC/TARGET_NRF5x/TARGET_SDK_11/libraries/crc16/crc16.c
      src/runtime/crt/rpc_server/buffer.cc
      src/runtime/crt/rpc_server/framing.cc
      src/runtime/crt/rpc_server/session.cc
      src/runtime/crt/rpc_server/write_stream.cc)
  include_directories("src/runtime/crt/host")
  list(APPEND RUNTIME_SRCS ${RUNTIME_MICRO_SRCS})
endif(USE_MICRO)
