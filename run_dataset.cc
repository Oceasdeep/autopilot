#include <fstream>
#include <vector>

//#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/graph/graph_constructor.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::int32;
using tensorflow::string;
using tensorflow::DT_FLOAT;
using tensorflow::DT_STRING;

// Reads a frozen graph protocol buffer from disk.
Status LoadGraph(string graph_file_name, tensorflow::GraphDef* graph_def) {
//                 std::unique_ptr<tensorflow::Session>* session) {
//  tensorflow::GraphDef graph_def;

  // Try reading initially in binary form and if that fails then in text form.
  Status read_binary_proto_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, graph_def);
  if (!read_binary_proto_status.ok()) {
    Status read_text_proto_status =
        ReadTextProto(tensorflow::Env::Default(), graph_file_name, graph_def);
    if (!read_text_proto_status.ok()) {
      return tensorflow::errors::NotFound("Failed to load graph file: ",
                                          graph_file_name);
    }
  }
  return Status::OK();
}
/*
  // Create a new session
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));

  // Create a new graph based on the graph_def loaded from disk
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}
*/

// Given an image folder, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ImageReader(const int input_height,
                   const int input_width,
                   const int cropped_height,
                   const int height,
                   const int width,
                   string input_name,
                   string output_name,
                   tensorflow::GraphDef* graph_def) {

  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  // Create root scope
  auto root = tensorflow::Scope::NewRootScope();

  // File path Placeholder
  auto file_path_placeholder = Placeholder(root.WithOpName(input_name),
                                           DT_STRING);

  // File reader
  auto file_reader = tensorflow::ops::ReadFile(root.WithOpName("file_reader"),
                                               file_path_placeholder);

  // Now decode the JPEG file
  const int wanted_channels = 3;
  auto image_reader =
      DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                 DecodeJpeg::Channels(wanted_channels));


  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, image_reader, 0);

  // Crop image to wanted dimensions
  auto cropped = Slice(root.WithOpName("crop"), dims_expander,
                       {0, input_height-cropped_height, 0, 0},
                       {1, cropped_height, input_width, wanted_channels});

  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, cropped,
      Const(root.WithOpName("size"), {height, width}));

  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), resized,
           tensorflow::DT_FLOAT);

  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), float_caster, {255.0f});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  TF_RETURN_IF_ERROR(root.ToGraphDef(graph_def));
  return Status::OK();
/*
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  */
}

int main(int argc, char* argv[]) {
  using namespace tensorflow;
  using namespace tensorflow::ops;

  // Initialize paths
  string log_dir = "logs";
  string graph_dir = "save";
  string graph_file = "frozen_graph.pb";
  string log_file = "runlog.cc.csv";
  string data_dir = "driving_dataset";
  string root_dir = "";

  // Placeholder names
  string input_layer_label = "x";
  string keep_prob_label = "keep_prob";
  string output_layer_label = "y";
  string image_file_label = "file_path";
  string loaded_image_label = "normalized";

  // Image properties
  int32 input_width = 455;
  int32 input_height = 256;
  int32 cropped_height = 150;
  int32 width = 200;
  int32 height = 66;

  // Input and output layer names
  string input_name = "x";
  string output_name = "y";


  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  //std::vector<Tensor> resized_tensors;
  tensorflow::GraphDef reader_def;
  Status img_reader_status =
      ImageReader(input_height, input_width,
                  cropped_height, height, width,
                  image_file_label, loaded_image_label,
                  &reader_def);
  if (!img_reader_status.ok()) {
    LOG(ERROR) << img_reader_status;
    return -1;
  }

  // Create image reader session
  std::unique_ptr<tensorflow::Session> reader_session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  Status reader_session_status = reader_session->Create(reader_def);
  if (!reader_session_status.ok()) {
    LOG(ERROR) << reader_session_status;
    return -1;
  }


  // Load the frozen graph from file
  string graph_path = tensorflow::io::JoinPath(root_dir, graph_dir, graph_file);
  tensorflow::GraphDef inference_def;
  Status load_graph_status = LoadGraph(graph_path, &inference_def);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // Create inference session
  std::unique_ptr<tensorflow::Session> inference_session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  Status inference_session_status = inference_session->Create(inference_def);
  if (!inference_session_status.ok()) {
    LOG(ERROR) << inference_session_status;
    return -1;
  }


/*
  GraphConstructorOptions opts;

  // Create root scope
  auto root = tensorflow::Scope::NewRootScope();

  // Create inference subscope
  auto inference_scope = root.NewSubScope("inference");

  // Add inference graphdef to the subscope
  Status inference_conversion_status = ConvertGraphDefToGraph(
      opts, inference_def, inference_scope.graph());
  if (!inference_conversion_status.ok()) {
    LOG(ERROR) << inference_conversion_status;
    return -1;
  }

  // Create reader subscope
  auto reader_scope = root.NewSubScope("reader");

  // Add reader graphdef to the subscope
  Status reader_conversion_status = ConvertGraphDefToGraph(
      opts, reader_def, reader_scope.graph());
  if (!reader_conversion_status.ok()) {
    LOG(ERROR) << reader_conversion_status;
    return -1;
  }

  ClientSession session(root);

  //const Tensor& resized_tensor = resized_tensors[0];
*/
  Tensor image_file(DT_STRING, tensorflow::TensorShape());
  image_file.scalar<string>()() = "driving_dataset/0.jpg";

  // Prepare inputs to be fed for the run
  std::vector<std::pair<string, tensorflow::Tensor>> load_inputs = {
    {image_file_label, image_file}
  };


  // Load image file
  std::vector<Tensor> image_outputs;
  Status load_status = reader_session->Run(load_inputs, {loaded_image_label},
                                           {}, &image_outputs);
  if (!load_status.ok()) {
    LOG(ERROR) << "Loading image file failed: " << load_status;
    return -1;
  }
  const Tensor& image_tensor = image_outputs[0];

  // Set dropout keep propability to 1.0 to turn off drouput during inference
  Tensor keep_prob(DT_FLOAT, tensorflow::TensorShape());
  keep_prob.scalar<float>()() = 1.0;

  // Prepare inputs to be fed for the run
  std::vector<std::pair<string, tensorflow::Tensor>> inference_inputs = {
    {input_layer_label, image_tensor},
    {keep_prob_label, keep_prob}
  };

  // Perform inference
  std::vector<Tensor> inference_outputs;
  Status inference_status = inference_session->Run(
                                        inference_inputs, {output_layer_label},
                                        {}, &inference_outputs);
  if (!inference_status.ok()) {
    LOG(ERROR) << "Inference failed: " << inference_status;
    return -1;
  }

  LOG(INFO) << inference_outputs[0].scalar<float>();


/*
  // Actually run the image through the model.
  std::vector<Tensor> outputs;

  Status run_status = session->Run(inputs,
                                   {output_layer_label}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  LOG(INFO) << outputs[0].scalar<float>();

*/
  return 0;
}
