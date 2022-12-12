#ifndef TCPSOCKED_HPP
#define TCPSOCKED_HPP

#include <netinet/in.h>
#include <vector>

class TcpSocked {

  public:
    explicit TcpSocked(const sockaddr_in &addr);
    TcpSocked(const char *ip, const short port);
    TcpSocked(const TcpSocked &) = delete;
    ~TcpSocked();
    TcpSocked &operator=(const TcpSocked &) = delete;

    void send(const uint8_t *buf, size_t len);
    std::vector<uint8_t> recv();

  private:
    int m_sockfd;
};

#endif // TCPSOCKED_HPP