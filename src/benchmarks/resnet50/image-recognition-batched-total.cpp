#include <torch/script.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>

#define kIMAGE_SIZE 224
#define kCHANNELS 3
#define kTOP_K 3

const char *file_name = "./800px-Porsche_991_silver_IAA.jpg";

bool load_image(cv::Mat &image, const char *fname) {
    image = cv::imread(fname);  // CV_8UC3
    if (image.empty() || !image.data) {
        return false;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int w = image.size().width, h = image.size().height;
    cv::Size scale((int)256*((float)w)/h, 256);
    cv::resize(image, image, scale);
    w = image.size().width, h = image.size().height;
    image = image(cv::Range(16,240), cv::Range(80, 304));
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

    return true;
}

int main(int argc, const char **argv) {
    if (argc < 2) {
        std::cerr << "please provide batch size as argument" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int n_runs = std::atoi(argv[1]);
    // std::cout << "n_runs=" << n_runs << std::endl;

    std::vector<std::string> images_str = {
        "./800px-Porsche_991_silver_IAA.jpg",
        "./800px-Sardinian_Warbler.jpg",
    };

    std::vector<cv::Mat> images(2);
    // load_image(images[0], images_str[0].c_str());
    // load_image(images[1], images_str[1].c_str());

    auto s = std::chrono::high_resolution_clock::now();

    static torch::jit::script::Module mod;
    try {
        mod = torch::jit::load("./resnet50.pt");
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model " << e.msg() << '\n';
        return -1;
    }

    torch::Device device(torch::kCUDA);
    mod.to(device);

    for (int i = 0; i < n_runs; i++) {
        load_image(images[i%2], images_str[i%2].c_str());
        auto input_tensor = torch::from_blob(
                images[i%2].data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
        input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
        input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

        input_tensor = input_tensor.to(device);

        torch::Tensor out_tensor = mod.forward({input_tensor}).toTensor();
        auto results = out_tensor.sort(-1, true);
        auto softmaxs = std::get<0>(results)[0].softmax(0);
        auto indexs = std::get<1>(results)[0];

        int idx = indexs[0].item<int>();
        double softmax = softmaxs[0].item<double>();
        // printf("%d %f\n", idx, softmax);
    }

    auto e = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000000.0;
    printf("%.10f\n", d);

    return 0;
}
