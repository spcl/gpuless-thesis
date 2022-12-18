#ifndef TCPClient_HPP
#define TCPClient_HPP

#include <netinet/in.h>
#include <vector>

class TcpClient {

  public:
    explicit TcpClient(const sockaddr_in &addr);
    TcpClient(const char *ip, const short port);
    TcpClient(const TcpClient &) = delete;
    ~TcpClient();
    TcpClient &operator=(const TcpClient &) = delete;

    void send(const uint8_t *buf, size_t len);
    std::vector<uint8_t> recv();

  private:
    int m_sockfd;
};

#endif // TCPClient_HPP