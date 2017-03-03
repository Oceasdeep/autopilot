#include <fstream>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
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

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::int32;
using tensorflow::string;

// Reads a frozen graph protocol buffer from disk.
Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;

  // Try reading initially in binary form and if that fails then in text form.
  Status read_binary_proto_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!read_binary_proto_status.ok()) {
    Status read_text_proto_status =
        ReadTextProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!read_text_proto_status.ok()) {
      return tensorflow::errors::NotFound("Failed to load graph file: ",
                                          graph_file_name);
    }
  }

  // Create a new session
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));

  // Create a new graph based on the graph_def loaded from disk
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

int main(int argc, char* argv[]) {

  // Initialize paths
  string log_dir = "logs";
  string graph_dir = "save";
  string graph_file = "frozen_graph.pb";
  string log_file = "runlog.cc.csv";
  string data_dir = "driving_dataset";
  string root_dir = "/home/tomi/tensorflow/tensorflow/autopilot";

  // Image properties
  int32 input_width = 455;
  int32 input_height = 256;
  int32 cropped_input_height = 150;
  int32 width = 200;
  int32 height = 66;

  // Input and output layer names
  string input_name = "x";
  string output_name = "y";

  // Load the frozen graph from file
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph_dir, graph_file);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  return 0;
}
