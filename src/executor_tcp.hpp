#ifndef __EXECUTOR_TCP_HPP__
#define __EXECUTOR_TCP_HPP__

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cerrno>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.hpp"

namespace gpuless {

const uint8_t REQ_TYPE_ALLOCATE   = 0;
const uint8_t REQ_TYPE_DEALLOCATE = 1;
const uint8_t REQ_TYPE_EXECUTE    = 2;
const uint8_t REQ_ANS_OK          = 3;
const uint8_t REQ_ANS_FAIL        = 4;

const int32_t KERNEL_ARG_POINTER        = 1 << 0;
const int32_t KERNEL_ARG_VALUE          = 1 << 1;
const int32_t KERNEL_ARG_COPY_TO_DEVICE = 1 << 2;
const int32_t KERNEL_ARG_COPY_TO_HOST   = 1 << 3;

struct buffer_tcp {
    size_t size;
    void *data = nullptr;

    buffer_tcp(size_t size) : size(size) {
        data = malloc(size);
    }

    ~buffer_tcp() {
        free(data);
    }
};

struct kernel_arg {
    int32_t flags;
    size_t  length;
    void    *data;
};

// read a length-prefixed value from a buffer
template<typename T>
static T read_value(size_t **next_object) {
    size_t l = **next_object;
    T v = *((T *) (*next_object + 1));
    *next_object = (size_t *) ((uint8_t *) *next_object + sizeof(size_t) + l);
    return v;
}

// append a value to a byte buffer
template<typename T>
static void append_buffer(std::vector<uint8_t> &buf, T *data) {
    size_t len = sizeof(*data);
    uint8_t *v = (uint8_t *) data;
    for (size_t i = 0; i < len; i++) {
        buf.push_back(v[i]);
    }
}

// append a value to a byte buffer, with length prefix
template<typename T>
static void append_buffer_value(std::vector<uint8_t> &buf, T *data) {
    size_t len = sizeof(*data);
    append_buffer(buf, &len);
    append_buffer(buf, data);
}

// append a string to a byte buffer, with length prefix
static void append_buffer_string(std::vector<uint8_t> &buf, const char *str) {
    size_t len = strlen(str) + 1; // include null byte
    append_buffer(buf, &len);
    for (size_t i = 0; i < len; i++) {
        buf.push_back(str[i]);
    }
}

// append data to buffer, with length prefix
static void append_buffer_data(std::vector<uint8_t> &buf, std::vector<uint8_t> &data) {
    size_t len = data.size();
    append_buffer(buf, &len);
    buf.insert(buf.end(), data.begin(), data.end());
}

// append raw data to buffer, with length prefix
static void append_buffer_raw(std::vector<uint8_t> &buf, void *data, size_t len) {
    append_buffer(buf, &len);
    auto s = buf.size();
    buf.resize(s + len);
    memcpy(buf.data() + s, data, len);
}

class executor_tcp {
private:
    std::vector<uint8_t> cuda_bin;
    sockaddr_in server_addr;

    bool load_cuda_bin(const char *fname) {
        std::ifstream input(fname, std::ios::binary);
        this->cuda_bin = std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
        std::cout << "CUDA binary size: " << this->cuda_bin.size() << std::endl; // dbg
        return true;
    }

    bool send_buffer(std::vector<uint8_t> &input, std::vector<uint8_t> &output) {
        int sock;
        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            std::cerr << "failed to open socket" << std::endl;
            return false;
        }

        if (connect(sock, (sockaddr *) &this->server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "failed to connect" << std::endl;
            return false;
        }

        // send length of data
        size_t len = input.size();
        send(sock, &len, sizeof(len), 0);

        // send data
        send(sock, input.data(), input.size(), 0);

        // read answer length
        size_t ans_len;
        recv(sock, &ans_len, sizeof(len), 0);
        output.resize(ans_len);

        // read answer data
        recv(sock, output.data(), ans_len, 0);

        close(sock);
        return true;
    }

public:
    executor_tcp() {}
    ~executor_tcp() {}

    static kernel_arg pointer_argument(const buffer_tcp *buffer, int32_t flags) {
        return kernel_arg {
            flags | KERNEL_ARG_POINTER,
            buffer->size,
            buffer->data,
        };
    }

    template<typename T>
    static kernel_arg value_argument(T* data, int32_t flags) {
        return kernel_arg {
            flags | KERNEL_ARG_VALUE,
            sizeof(*data),
            (void *) data,
        };
    }


    bool allocate(const char *ip,
                  const short port,
                  const char *cuda_bin_fname,
                  const std::vector<buffer_tcp*> &buffers) {
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        if (inet_pton(AF_INET, ip, &server_addr.sin_addr) < 0) {
            std::cerr << "invalid ip address" << std::endl;
            return false;
        }

        // load kernel coda from file
        if (!load_cuda_bin(cuda_bin_fname)) {
            std::cerr << "failed to load cuda binary" << std::endl;
            return false;
        }

        // remote buffers
        auto n_buffers = buffers.size();

        std::vector<uint8_t> buf;
        append_buffer_value(buf, &REQ_TYPE_ALLOCATE);
        append_buffer_value(buf, &n_buffers);
        for (const auto &b: buffers) {
            append_buffer_value(buf, &(b->size));
        }

        std::vector<uint8_t> answer;
        if (!this->send_buffer(buf, answer)) {
            return false;
        }

        return true;
    }

    bool execute(const char *kernel,
                 dim3 dimGrid,
                 dim3 dimBlock,
                 std::vector<kernel_arg> &args) {

        std::vector<uint8_t> buf;
        append_buffer_value(buf, &REQ_TYPE_EXECUTE);
        append_buffer_value(buf, &dimGrid);
        append_buffer_value(buf, &dimBlock);
        append_buffer_string(buf, kernel);
        append_buffer_data(buf, this->cuda_bin);

        auto n_args = args.size();
        append_buffer_value(buf, &n_args);
        for (const auto &a : args) {
            append_buffer_value(buf, &a.flags);
            append_buffer_raw(buf, a.data, a.length);
        }

        std::vector<uint8_t> answer;
        if (!this->send_buffer(buf, answer)) {
            return false;
        }

        size_t *next_object = (size_t *) answer.data();
        read_value<uint8_t>(&next_object); // type

        size_t n_return_buffers = read_value<size_t>(&next_object);

        size_t args_ctr = 0;
        for (size_t i = 0; i < n_return_buffers; i++) {
            size_t data_len = *next_object;

            // find the right kernel argument to write back data
            for (; args_ctr < n_args; args_ctr++) {
                if (args[args_ctr].flags & gpuless::KERNEL_ARG_COPY_TO_HOST) {
                    memcpy(args[args_ctr].data, next_object + 1, data_len);
                    break;
                }
            }

            next_object = (size_t *) ((uint8_t *) next_object + sizeof(size_t) + data_len);
        }

        return true;
    }

    bool deallocate() {
        std::vector<uint8_t> buf;
        append_buffer_value(buf, &REQ_TYPE_DEALLOCATE);

        std::vector<uint8_t> answer;
        if (!this->send_buffer(buf, answer)) {
            return false;
        }

        return true;
    }
};

} // namespace gpuless

#endif // __EXECUTOR_TCP_HPP__

