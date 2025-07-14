#include <torch/script.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    const int num_steps = 40000;
    std::vector<float> data(num_steps);
    for (int i = 0; i < num_steps; ++i) {
        data[i] = i * 0.001f;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input_tensor = torch::from_blob(data.data(), {num_steps, 1}, options).clone(); // clone to ensure safety

    at::Tensor output = module.forward({input_tensor}).toTensor();

    // Save to CSV
    std::ofstream outfile("output.csv");
    if (!outfile.is_open()) {
        std::cerr << "Failed to create the file\n";
        return -1;
    }

    auto sizes = output.sizes();
    int64_t num_rows = sizes[0];
    int64_t num_cols = sizes[1];

    std::vector<float> output_col1(num_rows), output_col2(num_rows);
    float y_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest();

    for (int64_t i = 0; i < num_rows; ++i) {
        for (int64_t j = 0; j < num_cols; ++j) {
            float val = output[i][j].item<float>();
            outfile << val;
            if (j != num_cols - 1)
                outfile << ",";
            if (j == 0) output_col1[i] = val;
            if (j == 1) output_col2[i] = val;
            y_min = std::min(y_min, val);
            y_max = std::max(y_max, val);
        }
        outfile << "\n";
    }
    outfile.close();
    std::cout << "CSV saved as output.csv\n";

    // PLOTTING using OpenCV
    const int width = 1024;
    const int height = 768;
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    auto map_x = [=](float x) {
        return static_cast<int>((x / 40.0f) * width);
    };
    auto map_y = [=](float y) {
        return static_cast<int>(height - ((y - y_min) / (y_max - y_min)) * height);
    };

    // Plot red line (col1)
    for (int i = 1; i < num_rows; ++i) {
        cv::line(
            plot,
            cv::Point(map_x(data[i - 1]), map_y(output_col1[i - 1])),
            cv::Point(map_x(data[i]), map_y(output_col1[i])),
            cv::Scalar(0, 0, 255), 1
        );
    }

    // Plot blue line (col2)
    for (int i = 1; i < num_rows; ++i) {
        cv::line(
            plot,
            cv::Point(map_x(data[i - 1]), map_y(output_col2[i - 1])),
            cv::Point(map_x(data[i]), map_y(output_col2[i])),
            cv::Scalar(255, 0, 0), 1
        );
    }

    // Save the plot
    if (!cv::imwrite("output_plot.png", plot)) {
        std::cerr << "Failed to write plot image.\n";
        return -1;
    }

    std::cout << "Plot saved as output_plot.png\n";
    std::cout << "ok\n";
    return 0;
}
