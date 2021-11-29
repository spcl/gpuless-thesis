#include <torch/script.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>

#define kIMAGE_SIZE 224
#define kCHANNELS 3
#define kTOP_K 3

const char *file_name = "./800px-Porsche_991_silver_IAA.jpg";

bool load_image(cv::Mat &image) {
    image = cv::imread(file_name);  // CV_8UC3
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

int recognition(cv::Mat &image) {
    static torch::jit::script::Module mod;
    static bool init = false;
    if (!init) {
        try {
            // reuse model in warm functions
            mod = torch::jit::load("./resnet50.pt");
            init = true;
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model " << e.msg() << '\n';
            return -1;
        }
    }

    if (load_image(image)) {
        torch::Device device(torch::kCUDA);
        // torch::Device device(torch::kCPU);

        mod.to(device);

        auto input_tensor = torch::from_blob(
                image.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
        input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
        input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

        input_tensor = input_tensor.to(device);

        torch::Tensor out_tensor = mod.forward({input_tensor}).toTensor();
        auto results = out_tensor.sort(-1, true);
        auto softmaxs = std::get<0>(results)[0].softmax(0);
        auto indexs = std::get<1>(results)[0];

        std::cout << indexs[0].item<int>() << " " << softmaxs[0].item<double>() << std::endl;
        return indexs[0].item<int>();
    }

    return -1;
}

int main() {
    auto s = std::chrono::high_resolution_clock::now();

    cv::Mat image;
    if (recognition(image) < 0) {
        std::cout << "image recognition failed" << std::endl;
    }

    auto e = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000000.0;
    printf("%.8f\n", d);

    return 0;
}
