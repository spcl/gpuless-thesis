#include "tcp_gpu_session.hpp"
#include <netinet/in.h>

namespace gpuless {

TcpGpuSession::TcpGpuSession(const sockaddr_in &addr, int32_t session_id)
    : m_socked(addr), m_session_id(session_id) {}

TcpGpuSession::TcpGpuSession(const char *ip, const short port,
                             int32_t session_id)
    : m_socked(ip, port), m_session_id(session_id) {}

TcpGpuSession::~TcpGpuSession() {}

int32_t TcpGpuSession::getSessionId() { return m_session_id; }

void TcpGpuSession::send(const uint8_t *buf, size_t len) {
    m_socked.send(buf, len);
}

std::vector<uint8_t> TcpGpuSession::recv() { return m_socked.recv(); }

} // namespace gpuless