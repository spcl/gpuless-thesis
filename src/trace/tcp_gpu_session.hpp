#ifndef GPULESS_TCP_GPU_SESSION_H
#define GPULESS_TCP_GPU_SESSION_H

#include "../TcpClient.hpp"
#include <netinet/in.h>

namespace gpuless {
class TcpGpuSession {
  private:
    TcpClient m_socked;
    int32_t m_session_id = -1;

  public:
    TcpGpuSession(const sockaddr_in &adrr, int32_t session_id);
    TcpGpuSession(const char *ip, const short port, int32_t session_id);
    ~TcpGpuSession();

    int32_t getSessionId();

    void send(const uint8_t *buf, size_t len);
    std::vector<uint8_t> recv();
};

} // namespace gpuless

#endif // GPULESS_TCP_GPU_SESSION_H