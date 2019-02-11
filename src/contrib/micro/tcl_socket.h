/*!
 *  Copyright (c) 2017 by Contributors
 * \file socket.h
 * \brief this file aims to provide a wrapper of sockets
 * \author Tianqi Chen
 */
#ifndef TVM_RUNTIME_UTVM_TCL_SOCKET_H_
#define TVM_RUNTIME_UTVM_TCL_SOCKET_H_

#include "../../common/socket.h"


namespace tvm {
namespace common {
/*!
 * \brief a wrapper of Tcl socket that hopefully be cross platform
 */
class TclSocket : public tvm::common::Socket {
 public:
  TclSocket() : Socket(INVALID_SOCKET) {
  }
  /*!
   * \brief construct a Tcl socket from existing descriptor
   * \param sockfd The descriptor
   */
  explicit TclSocket(SockType sockfd) : Socket(sockfd) {
  }
  /*!
   * \brief enable/disable Tcl keepalive
   * \param keepalive whether to set the keep alive option on
   */
  void SetKeepAlive(bool keepalive) {
    int opt = static_cast<int>(keepalive);
    if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE,
                   reinterpret_cast<char*>(&opt), sizeof(opt)) < 0) {
      Socket::Error("SetKeepAlive");
    }
  }
  /*!
   * \brief create the socket, call this before using socket
   * \param af domain
   */
  void Create(int af = PF_INET) {
    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd == INVALID_SOCKET) {
      Socket::Error("Create");
    }
  }
  /*!
   * \brief perform listen of the socket
   * \param backlog backlog parameter
   */
  void Listen(int backlog = 16) {
    listen(sockfd, backlog);
  }
  /*!
   * \brief get a new connection
   * \return The accepted socket connection.
   */
  TclSocket Accept() {
    SockType newfd = accept(sockfd, NULL, NULL);
    if (newfd == INVALID_SOCKET) {
      Socket::Error("Accept");
    }
    return TclSocket(newfd);
  }
  /*!
   * \brief decide whether the socket is at OOB mark
   * \return 1 if at mark, 0 if not, -1 if an error occured
   */
  int AtMark() const {
    int atmark;
    if (ioctl(sockfd, SIOCATMARK, &atmark) == -1) return -1;
    return static_cast<int>(atmark);
  }
  /*!
   * \brief connect to an address
   * \param addr the address to connect to
   * \return whether connect is successful
   */
  bool Connect(const SockAddr &addr) {
    return connect(sockfd, reinterpret_cast<const sockaddr*>(&addr.addr),
                   sizeof(addr.addr)) == 0;
  }
  /*!
   * \brief send data using the socket
   * \param buf_ the pointer to the buffer
   * \param len the size of the buffer
   * \param flag extra flags
   * \return size of data actually sent
   *         return -1 if error occurs
   */
  ssize_t Send(const void *buf_, size_t len, int flag = 0) {
    const char *buf = reinterpret_cast<const char*>(buf_);
    return send(sockfd, buf, static_cast<sock_size_t>(len), flag);
  }
  /*!
   * \brief receive data using the socket
   * \param buf_ the pointer to the buffer
   * \param len the size of the buffer
   * \param flags extra flags
   * \return size of data actually received
   *         return -1 if error occurs
   */
  ssize_t Recv(void *buf_, size_t len, int flags = 0) {
    char *buf = reinterpret_cast<char*>(buf_);
    return recv(sockfd, buf, static_cast<sock_size_t>(len), flags);
  }
  /*!
   * \brief peform block write that will attempt to send all data out
   *    can still return smaller than request when error occurs
   * \param buf_ the pointer to the buffer
   * \param len the size of the buffer
   * \return size of data actually sent
   */
  size_t SendAll(const void *buf_, size_t len) {
    const char *buf = reinterpret_cast<const char*>(buf_);
    size_t ndone = 0;
    while (ndone <  len) {
      ssize_t ret = send(sockfd, buf, static_cast<ssize_t>(len - ndone), 0);
      if (ret == -1) {
        if (LastErrorWouldBlock()) return ndone;
        Socket::Error("SendAll");
      }
      buf += ret;
      ndone += ret;
    }
    return ndone;
  }
  /*!
   * \brief peform block read that will attempt to read all data
   *    can still return smaller than request when error occurs
   * \param buf_ the buffer pointer
   * \param len length of data to recv
   * \return size of data actually sent
   */
  size_t RecvAll(void *buf_, size_t len) {
    char *buf = reinterpret_cast<char*>(buf_);
    size_t ndone = 0;
    while (ndone <  len) {
      ssize_t ret = recv(sockfd, buf,
                         static_cast<sock_size_t>(len - ndone), MSG_WAITALL);
      if (ret == -1) {
        if (LastErrorWouldBlock())  {
          LOG(FATAL) << "would block";
          return ndone;
        }
        Socket::Error("RecvAll");
      }
      if (ret == 0) return ndone;
      buf += ret;
      ndone += ret;
    }
    return ndone;
  }
};
}  // namespace common
}  // namespace tvm
#endif // TVM_RUNTIME_UTVM_TCL_SOCKET_H_
