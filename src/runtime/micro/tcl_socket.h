/*!
 *  Copyright (c) 2019 by Contributors
 * \file TODO
 * \brief TODO
 */
#ifndef TVM_RUNTIME_MICRO_TCL_SOCKET_H_
#define TVM_RUNTIME_MICRO_TCL_SOCKET_H_

#include "../../common/socket.h"

namespace tvm {
namespace runtime {

/*!
 * \brief TODO
 */
class TclSocket {
 public:
  /*!
   * \brief TODO
   */
  explicit TclSocket();

  /*!
   * \brief TODO
   */
  ~TclSocket();

  /*!
   * \brief TODO
   * \param addr TODO
   */
  void Connect(tvm::common::SockAddr addr);

  /*
   * brief TODO
   *
   * returns the reply
   */
  std::string SendCommand(std::string cmd, bool verbose=false);

 private:
  tvm::common::TCPSocket tcp_socket_;

  static const constexpr char kCommandTerminateToken = '\x1a';
  static const constexpr size_t kSendBufSize = 4096;
  static const constexpr size_t kReplyBufSize = 4096;
};

}  // namespace runtime
}  // namespace tvm
#endif  //TVM_RUNTIME_MICRO_TCL_SOCKET_H_
