#include <csignal>
#include <iostream>
#include <pthread.h>
#include <sys/wait.h>

#include "../utils.hpp"
#include "manager.hpp"
#include "manager_device.hpp"

#include "../schemas/allocation_protocol_generated.h"

using namespace gpuless::manager;

extern const bool DEBUG = true;
extern const int BACKLOG = 5;

// nvidia (mig) devices: [device, profile id, session assignment]
static std::mutex lock_devices;
static std::vector<std::tuple<int, int, int>> devices = {
    {0, NO_MIG, NO_SESSION_ASSIGNED},
};

static std::atomic<int> next_session_id(1);
static std::vector<pid_t> child_processes;

static void deallocate_session_devices(int32_t session_id) {
    lock_devices.lock();
    for (auto &d : devices) {
        if (std::get<2>(d) == session_id) {
            std::get<2>(d) = NO_SESSION_ASSIGNED;
            break;
        }
    }
    lock_devices.unlock();
}

static std::vector<int32_t> available_profiles() {
    std::vector<int32_t> r;
    lock_devices.lock();
    for (const auto &d : devices) {
        if (std::get<2>(d) == NO_SESSION_ASSIGNED) {
            r.push_back(std::get<1>(d));
        }
    }
    lock_devices.unlock();
    return r;
}

static int32_t assign_device(int32_t profile, int32_t session_id) {
    lock_devices.lock();
    int assigned_device = -1;
    for (size_t i = 0; i < devices.size(); i++) {
        if (std::get<2>(devices[i]) == NO_SESSION_ASSIGNED &&
            profile == std::get<1>(devices[i])) {
            std::get<2>(devices[i]) = session_id;
            assigned_device = i;
            break;
        }
    }
    lock_devices.unlock();
    return assigned_device;
}

void handle_allocate_request(int socket_fd, const ProtocolMessage *msg) {
    int session_id = next_session_id++;
    if (DEBUG) {
        printf("allocation request for %d (session_id=%d)\n",
               msg->message_as_AllocateRequest()->profile(), session_id);
    }

    flatbuffers::FlatBufferBuilder builder;

    // offer a set of instances (all available instances)
    auto profiles = available_profiles();
    auto allocate_offer_msg = CreateProtocolMessage(
        builder, Message_AllocateOffer,
        CreateAllocateOffer(builder, Status_OK, session_id,
                            builder.CreateVector(profiles))
            .Union());
    builder.Finish(allocate_offer_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    // client selects a profile
    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto allocate_select_msg = GetProtocolMessage(buffer.data());
    int32_t selected_profile =
        allocate_select_msg->message_as_AllocateSelect()->profile();

    // send confirmation
    int32_t assigned_device = assign_device(selected_profile, session_id);
    auto status = assigned_device < 0 ? Status_FAILURE : Status_OK;
    // TODO: set correct ip, port
    uint32_t ip;
    inet_pton(AF_INET, "127.0.0.1", &ip);
    builder.Reset();
    auto allocate_confirm_msg = CreateProtocolMessage(
        builder, Message_AllocateConfirm,
        CreateAllocateConfirm(builder, status, session_id, ip, MANAGER_PORT + 1)
            .Union());
    builder.Finish(allocate_confirm_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
}

void handle_deallocate_request(int socket_fd, const ProtocolMessage *msg) {
    int32_t session_id = msg->message_as_DeallocateRequest()->session_id();
    if (DEBUG) {
        printf("deallocation request for session_id=%d\n", session_id);
    }

    deallocate_session_devices(session_id);
    flatbuffers::FlatBufferBuilder builder;
    auto deallocate_confirm_msg = CreateProtocolMessage(
        builder, Message_DeallocateConfirm,
        CreateDeallocateConfirm(builder, Status_OK, session_id).Union());
    builder.Finish(deallocate_confirm_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
}

void *handle_request(void *arg) {
    int socket_fd = *((int *)arg);
    delete ((int *)arg);

    // generic initial request from client
    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto msg = GetProtocolMessage(buffer.data());
    if (msg->message_type() == Message_AllocateRequest) {
        handle_allocate_request(socket_fd, msg);
    } else if (msg->message_type() == Message_DeallocateRequest) {
        handle_deallocate_request(socket_fd, msg);
    } else {
        std::cerr << "invalid request" << std::endl;
    }

    close(socket_fd);
    return nullptr;
}

void sigint_handler(int signum) {
    for (const auto p : child_processes) {
        kill(p, SIGTERM);
        wait(nullptr);
    }
    exit(signum);
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    signal(SIGINT, sigint_handler);

    // start device management processes
    short next_port = MANAGER_PORT + 1;
    for (const auto &t : devices) {
        int device = std::get<0>(t);
        int profile = std::get<1>(t);

        pid_t pid = fork();
        if (pid == 0) {
            manage_device(device, next_port);
        } else {
            child_processes.push_back(pid);
        }

        if (DEBUG) {
            printf("managing device: %d (profile=%d,pid=%d,port=%d)\n", device,
                   profile, pid, next_port);
        }
        next_port++;
    }

    // run server
    int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        std::cerr << "failed to open socket" << std::endl;
        exit(EXIT_FAILURE);
    }

    int opt = 1;
    setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, (void *)&opt, sizeof(opt));

    sockaddr_in sa;
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port = htons(MANAGER_PORT);

    if (bind(socket_fd, (sockaddr *)&sa, sizeof(sa)) < 0) {
        std::cerr << "failed to bind socket" << std::endl;
        close(socket_fd);
        exit(EXIT_FAILURE);
    }

    if (listen(socket_fd, BACKLOG) < 0) {
        std::cerr << "failed to listen on socket" << std::endl;
        close(socket_fd);
        exit(EXIT_FAILURE);
    }

    std::cout << "manager running on port " << MANAGER_PORT << std::endl;

    int s_new;
    sockaddr remote_addr;
    socklen_t remote_addrlen = sizeof(remote_addr);
    while ((s_new = accept(socket_fd, &remote_addr, &remote_addrlen))) {
        if (DEBUG) {
            const char *ip = inet_ntoa(((sockaddr_in *)&remote_addr)->sin_addr);
            std::cout << "manager: connection from " << ip << std::endl;
        }

        // handle connection in new thread
        int *s_new_alloc = new int;
        *s_new_alloc = s_new;
        pthread_t t;
        pthread_create(&t, nullptr, &handle_request, s_new_alloc);
    }

    close(socket_fd);
    return EXIT_SUCCESS;
}
