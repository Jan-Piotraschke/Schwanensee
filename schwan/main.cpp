#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "schwan");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::string model_path = "./model.onnx";
    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_ptr.get();

    std::vector<const char*> input_names{input_name};
    std::vector<const char*> output_names{output_name};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const int num_steps = 180;
    std::vector<float> t_values(num_steps);
    std::vector<float> y_values(num_steps);

    float x = 1.0f;
    float y = 0.1f;

    for (int i = 0; i < num_steps; ++i) {
        t_values[i] = i * 0.1f;

        std::vector<float> input_tensor_values = {t_values[i], x, y};
        std::vector<int64_t> input_shape = {1, 3};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &input_tensor,
            input_names.size(),
            output_names.data(),
            output_names.size()
        );

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        x = output_data[0];
        y = output_data[1];
        float dx_dt = output_data[2];
        float dy_dt = output_data[3];

        y_values[i] = y; // store for plotting
    }

    // PLOT using OpenCV
    const int width = 1024;
    const int height = 768;
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    float y_min = *std::min_element(y_values.begin(), y_values.end());
    float y_max = *std::max_element(y_values.begin(), y_values.end());

    auto map_x = [&](float t) { return static_cast<int>((t / t_values.back()) * width); };
    auto map_y = [&](float yv) { return static_cast<int>(height - ((yv - y_min) / (y_max - y_min)) * height); };

    for (int i = 1; i < num_steps; ++i) {
        cv::line(
            plot,
            cv::Point(map_x(t_values[i - 1]), map_y(y_values[i - 1])),
            cv::Point(map_x(t_values[i]), map_y(y_values[i])),
            cv::Scalar(0, 0, 255), 1
        );
    }

    // Save the plot
    if (!cv::imwrite("y_trajectory.png", plot)) {
        std::cerr << "Failed to write plot image.\n";
        return -1;
    }
    std::cout << "Plot saved as y_trajectory.png\n";

    return 0;
}
