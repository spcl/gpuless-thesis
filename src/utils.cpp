#include "utils.hpp"

void __checkCudaErrors(CUresult r, const char *file, const int line) {
    if (r != CUDA_SUCCESS) {
        const char *msg;
        cuGetErrorName(r, &msg);
        std::cout << "cuda error in " << file << "(" << line << "):"
            << std::endl << msg << std::endl;
    }
}

std::vector<uint8_t> recv_buffer(int socket_fd) {
    size_t msg_len;
    recv(socket_fd, &msg_len, sizeof(msg_len), 0);
    std::vector<uint8_t> buf(msg_len);
    size_t bytes_read = 0;
    do {
        void *dst = buf.data() + bytes_read;
        size_t n_read = std::min(65536UL, msg_len - bytes_read);
        bytes_read += recv(socket_fd, dst, n_read, 0);
    } while (bytes_read < msg_len);
    return buf;
}

void send_buffer(int socket_fd, const uint8_t *buf, size_t len) {
    send(socket_fd, &len, sizeof(len), 0);
    size_t bytes_sent = 0;
    do {
        void *src = (void *) (buf + bytes_sent);
        size_t n_send = std::min(65536UL, len - bytes_sent);
        bytes_sent += send(socket_fd, src, n_send, 0);
    } while (bytes_sent < len);
}
