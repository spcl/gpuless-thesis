#include "TcpSocked.hpp"
#include "utils.hpp"
#include <netinet/in.h>
#include <spdlog/spdlog.h>

TcpSocked::TcpSocked(sockaddr_in addr) {
    if ((m_sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        SPDLOG_ERROR("failed to open socket");
        throw "failed to open socket";
    }

    if (connect(m_sockfd, (sockaddr *)&addr, sizeof(addr)) < 0) {
        SPDLOG_ERROR("failed to connect");
        throw "failed to connect";
    }
}

TcpSocked::~TcpSocked() { close(m_sockfd); }

void TcpSocked::send(const uint8_t *buf, size_t len) {
    send_buffer(m_sockfd, buf, len);
}

std::vector<uint8_t> TcpSocked::recv() { return recv_buffer(m_sockfd); }