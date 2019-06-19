/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file TODO
 * \brief TODO
 */


// Socket stuff
#include <fcntl.h>
#include <netdb.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
// IO
#include<iostream>
#include<iomanip>
// Thread Sleeping
#include <unistd.h>

#include "tcl_socket.h"

namespace tvm {
namespace runtime {

// TODO(weberlo): Can we query the device for how much memory it has?

TclSocket::TclSocket() {
  tcp_socket_.Create();
  tcp_socket_.SetKeepAlive(true);
}

TclSocket::~TclSocket() {
  // tcp_socket_.Close();
}

void TclSocket::Connect(tvm::common::SockAddr addr) {
  CHECK(tcp_socket_.Connect(addr)) << "failed to connect";
}

std::string TclSocket::SendCommand(std::string cmd, bool verbose) {
  verbose = false;
  std::stringstream cmd_builder;
  cmd_builder << cmd;
  cmd_builder << kCommandTerminateToken;
  std::string full_cmd = cmd_builder.str();
  CHECK(tcp_socket_.Send(full_cmd.data(), full_cmd.length()) != -1) << "failed to send command";
  if (verbose) {
    std::cout << "SEND: " << full_cmd << std::endl;
  }

  // Receive reply.
  std::stringstream reply_builder;
  static char reply_buf[kReplyBufSize];
  do {
    ssize_t bytes_read;
    do {
      // Leave room at the end of `reply_buf` to tack on a null terminator.
      bytes_read = tcp_socket_.Recv(reply_buf, kReplyBufSize - 1);
      reply_buf[bytes_read] = '\0';
      for (int i = 0; i < bytes_read - 1; i++) {
        if (reply_buf[i] == kCommandTerminateToken) {
          CHECK(false) << "command terminator received in middle of reply";
        }
      }
      reply_builder << reply_buf;
    } while (bytes_read == kReplyBufSize - 1);
    // -1 signals an error from `socket`.
    CHECK(bytes_read != -1) << "failed to read command reply";
  } while (reply_builder.str()[reply_builder.str().length() - 1] != kCommandTerminateToken);
  std::string reply = reply_builder.str();

  if (verbose) {
    std::cout << "RECV";
    // Make sure the reply ends in a command terminator.
    if (reply[reply.length()-1] != kCommandTerminateToken) {
      std::cout << " (missing command terminator)";
    }
    std::cout << ": " << reply << std::endl;
  }

  return reply;
}

}  // namespace runtime
}  // namespace tvm
