#define _USE_MATH_DEFINES

#include <ostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <malloc.h>
#include <pthread.h>
#include <limits.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/prctl.h>

#include <jpeglib.h>
#include <setjmp.h>

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


// Compute the time difference in nanosecods between two time events
int64_t timediff_nanoseconds(timespec end, timespec start)
{
  const int64_t second = 1000000000;
  timespec temp;
  temp.tv_sec = end.tv_sec-start.tv_sec;
  temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_nsec += second;
    temp.tv_sec -= 1;
  }
	return static_cast<int64_t>(temp.tv_sec)*second
         + static_cast<int64_t>(temp.tv_nsec);
}


// Computes the time difference in seconds between two time events
double timediff(timespec end, timespec start)
{
  const double second = 1000000000;
  int64_t diff =  timediff_nanoseconds(end, start);
  return static_cast<double>(diff) / second;
}


// Set process (real-time) priority and scheduler
static void setprio(int prio, int sched)
{
 struct sched_param param;
 // Set realtime priority for this thread
 param.sched_priority = prio;
 if (sched_setscheduler(0, sched, &param) < 0)
   perror("sched_setscheduler");
}


// Display memory pagefault count
void show_new_pagefault_count(const char* logtext,
           const char* allowed_maj,
           const char* allowed_min)
{
 static int last_majflt = 0, last_minflt = 0;
 struct rusage usage;

 getrusage(RUSAGE_SELF, &usage);

 printf("%-30.30s: Pagefaults, Major:%ld (Allowed %s), " \
        "Minor:%ld (Allowed %s)\n", logtext,
        usage.ru_majflt - last_majflt, allowed_maj,
        usage.ru_minflt - last_minflt, allowed_min);

 last_majflt = usage.ru_majflt;
 last_minflt = usage.ru_minflt;
}

// Prepare memory allocations for real-time usage
static void configure_malloc_behavior(void)
{
 /* Now lock all current and future pages
    from preventing of being paged */
 if (mlockall(MCL_CURRENT | MCL_FUTURE))
   perror("mlockall failed:");

 /* Turn off malloc trimming.*/
 mallopt(M_TRIM_THRESHOLD, -1);

 /* Turn off mmap usage. */
 mallopt(M_MMAP_MAX, 0);
}


// Allocate memory and touch it to force it to RAM
static void reserve_process_memory(int size)
{
 int i;
 char *buffer;

 buffer = (char*) malloc(size);

 /* Touch each page in this piece of memory to get it mapped into RAM */
 for (i = 0; i < size; i += sysconf(_SC_PAGESIZE)) {
   /* Each write to this buffer will generate a pagefault.
      Once the pagefault is handled a page will be locked in
      memory and never given back to the system. */
   buffer[i] = 0;
 }

 /* buffer will now be released. As Glibc is configured such that it
    never gives back memory to the kernel, the memory allocated above is
    locked for this process. All malloc() and new() calls come from
    the memory pool reserved and locked above. Issuing free() and
    delete() does NOT make this locking undone. So, with this locking
    mechanism we can build C++ applications that will never run into
    a major/minor pagefault, even with swapping enabled. */
 free(buffer);
}



// Error handling for JPEG decoding.
void CatchError(j_common_ptr cinfo) {
  (*cinfo->err->output_message)(cinfo);
  jmp_buf *jpeg_jmpbuf = reinterpret_cast<jmp_buf *>(cinfo->client_data);
  jpeg_destroy(cinfo);
  longjmp(*jpeg_jmpbuf, 1);
}

// Decompresses a JPEG file from disk into a uin8 vector
Status LoadJpegFile(string file_name, std::vector<tensorflow::uint8>* data,
		    int* width, int* height, int* channels) {
  struct jpeg_decompress_struct cinfo;
  FILE * infile;
  JSAMPARRAY buffer;
  int row_stride;

  if ((infile = fopen(file_name.c_str(), "rb")) == NULL) {
    LOG(ERROR) << "Can't open " << file_name;
    return tensorflow::errors::NotFound("JPEG file ", file_name,
					" not found");
  }

  struct jpeg_error_mgr jerr;
  jmp_buf jpeg_jmpbuf;  // recovery point in case of error
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = CatchError;
  if (setjmp(jpeg_jmpbuf)) {
    fclose(infile);
    return tensorflow::errors::Unknown("JPEG decoding failed");
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);
  *width = cinfo.output_width;
  *height = cinfo.output_height;
  *channels = cinfo.output_components;
  data->resize((*height) * (*width) * (*channels));

  row_stride = cinfo.output_width * cinfo.output_components;
  buffer = (*cinfo.mem->alloc_sarray)
    ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
  while (cinfo.output_scanline < cinfo.output_height) {
    tensorflow::uint8* row_address = &((*data)[cinfo.output_scanline * row_stride]);
    jpeg_read_scanlines(&cinfo, buffer, 1);
    memcpy(row_address, buffer[0], row_stride);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(infile);
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(string file_name, const int wanted_height,
                               const int wanted_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  std::vector<tensorflow::uint8> image_data;
  int image_width;
  int image_height;
  int image_channels;
  TF_RETURN_IF_ERROR(LoadJpegFile(file_name, &image_data, &image_width,
				  &image_height, &image_channels));
  const int wanted_channels = 3;
  if (image_channels < wanted_channels) {
    return tensorflow::errors::FailedPrecondition("Image needs to have at least ",
						  wanted_channels, " but only has ",
						  image_channels);
  }
  // In these loops, we convert the eight-bit data in the image into float, resize
  // it using bilinear filtering, and scale it numerically to the float range that
  // the model expects (given by input_mean and input_std).
  tensorflow::Tensor image_tensor(
      tensorflow::DT_FLOAT, tensorflow::TensorShape(
      {1, wanted_height, wanted_width, wanted_channels}));
  auto image_tensor_mapped = image_tensor.tensor<float, 4>();
  tensorflow::uint8* in = image_data.data();
  float *out = image_tensor_mapped.data();
  const size_t image_rowlen = image_width * image_channels;
  const float width_scale = static_cast<float>(image_width) / wanted_width;
  const float height_scale = static_cast<float>(image_height) / wanted_height;
  for (int y = 0; y < wanted_height; ++y) {
    const float in_y = y * height_scale;
    const int top_y_index = static_cast<int>(floorf(in_y));
    const int bottom_y_index =
      std::min(static_cast<int>(ceilf(in_y)), (image_height - 1));
    const float y_lerp = in_y - top_y_index;
    tensorflow::uint8* in_top_row = in + (top_y_index * image_rowlen);
    tensorflow::uint8* in_bottom_row = in + (bottom_y_index * image_rowlen);
    float *out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const float in_x = x * width_scale;
      const int left_x_index = static_cast<int>(floorf(in_x));
      const int right_x_index =
	std::min(static_cast<int>(ceilf(in_x)), (image_width - 1));
      tensorflow::uint8* in_top_left_pixel =
	in_top_row + (left_x_index * wanted_channels);
      tensorflow::uint8* in_top_right_pixel =
	in_top_row + (right_x_index * wanted_channels);
      tensorflow::uint8* in_bottom_left_pixel =
	in_bottom_row + (left_x_index * wanted_channels);
      tensorflow::uint8* in_bottom_right_pixel =
	in_bottom_row + (right_x_index * wanted_channels);
      const float x_lerp = in_x - left_x_index;
      float *out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
	const float top_left((in_top_left_pixel[c] - input_mean) / input_std);
	const float top_right((in_top_right_pixel[c] - input_mean) / input_std);
	const float bottom_left((in_bottom_left_pixel[c] - input_mean) / input_std);
	const float bottom_right((in_bottom_right_pixel[c] - input_mean) / input_std);
	const float top = top_left + (top_right - top_left) * x_lerp;
	const float bottom =
	  bottom_left + (bottom_right - bottom_left) * x_lerp;
	out_pixel[c] = top + (bottom - top) * y_lerp;
      }
    }
  }
  out_tensors->push_back(image_tensor);
  return Status::OK();
}

// Reads a frozen graph protocol buffer from disk.
Status LoadGraph(string graph_file_name, tensorflow::GraphDef* graph_def) {
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

// Main function
int main(int argc, char* argv[]) {
  using namespace tensorflow;
  using namespace tensorflow::ops;

  // Initialize path variables
  string log_dir = "results";
  string graph_dir = "save";
  string graph_file = "frozen_graph.pb";
  string log_file = "run.rt.csv";
  string data_dir = tensorflow::io::JoinPath("driving_dataset","scaled");
  string root_dir = "";

  // Tensorflow graph placeholder names
  string input_layer_label = "x";
  string keep_prob_label = "keep_prob";
  string output_layer_label = "y";
  string image_file_label = "file_path";
  string loaded_image_label = "normalized";

  // Image properties
  const int input_height = 66;
  const int input_width = 200;
  const float input_mean = 0.0;
  const float input_std = 255.0;
  const int one_past_last_image = 45567;

  // Real-Time properties
  int priority = 40; // Must be be
  int scheduler = SCHED_FIFO;
  int clk = CLOCK_MONOTONIC_RAW;
  size_t heap_preallocation_size = 8589934592; // 8 GB

  // Show page faults
 	show_new_pagefault_count("Initial count", ">=0", ">=0");

  // Configure memory allocator behavior
  configure_malloc_behavior();

  // Show page faults
  show_new_pagefault_count("mlockall() generated", ">=0", ">=0");

  // Reserve and activate heap
  reserve_process_memory(heap_preallocation_size);

  // Set process priority
  setprio(priority, scheduler);

  // Load the frozen graph we are going to use for inference from file
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

  // Get the image from disk as a float array of numbers, resized and
  // normalized to the specifications the graph input placeholder expects.
  std::vector<Tensor> resized_tensors;
  std::stringstream ss;
  ss << 0 << ".jpg";
  string image_file_name = ss.str();
  string image_path = tensorflow::io::JoinPath(root_dir, data_dir,
                                               image_file_name);
  Status read_tensor_status =
     ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                             input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << "Failed to read image file.";
    return -1;
  }

  // Extract the image from the returned tensor vector
  const Tensor& image_tensor = resized_tensors[0];

  // Set dropout keep propability to 1.0 to turn off drouput during inference
  Tensor keep_prob(DT_FLOAT, tensorflow::TensorShape());
  keep_prob.scalar<float>()() = 1.0;

  // Create runlog stream that we use for logging during inference
  std::ostringstream runlog;

  // Write header to the runlog stream
  runlog << "Index,Time,Time_Diff,Output" << std::endl;

  // Initialize current time
  timespec t;
  clock_gettime(clk, &t);

  // Initialize star time
  timespec t0 = t;

  // Initialize image index
  long image_index = -1;
  
  // Initialize output variable
  float output = 0.0;

  // Initialize time for previous iteration
  timespec t_prev;
  t_prev = t;

  // Time variables in seconds for logging
  double dt;
  double dt0;

  // Write initial entry to runlog
  dt0 = timediff(t,t0);
  dt = timediff(t,t_prev);
  runlog << -1 << "," << dt0 << "," << dt
        << "," << output << std::endl;

  // Show page faults
  show_new_pagefault_count("Before inference", ">=0", ">=0");

  // Inference loop, loop through defined number of images
  while(true){

    // Increment image index
    image_index++;

    // Get the image from disk as a float array of numbers, resized and
    // normalized to the specifications the graph input placeholder expects.
    std::vector<Tensor> resized_tensors;
    std::stringstream ss;
    ss << image_index << ".jpg";
    string image_file_name = ss.str();
    string image_path = tensorflow::io::JoinPath(root_dir, data_dir,
                                                 image_file_name);
    Status read_tensor_status =
       ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                               input_std, &resized_tensors);
    if (!read_tensor_status.ok()) {
      // Last image, exit loop
      break;
    }

    // Extract the image from the returned tensor vector
    const Tensor& image_tensor = resized_tensors[0];

    // Prepare inputs to be fed for the inference step
    std::vector<std::pair<string, tensorflow::Tensor>> inference_inputs = {
      {input_layer_label, image_tensor},
      {keep_prob_label, keep_prob}
    };

    // Perform the inference step
    std::vector<Tensor> inference_outputs;
    Status inference_status = inference_session->Run(
                                          inference_inputs, {output_layer_label},
                                          {}, &inference_outputs);
    if (!inference_status.ok()) {
      LOG(ERROR) << "Inference failed: " << inference_status;
      return -1;
    }

    // Extract the steering angle from the inference outputs
    output = inference_outputs[0].scalar<float>()(0);


    // Time the inference duration
    clock_gettime(clk, &t);
    dt0 = timediff(t,t0);
    dt = timediff(t,t_prev);

    // Write log entry
    runlog << image_index << "," << dt0 << "," << dt
          << "," << output << std::endl;

    // Set t_prev to current time
    t_prev = t;

  }

  // Show page faults
  show_new_pagefault_count("final count", ">=0", ">=0");

  // Open runlog file and add a header line
  string runlog_path = tensorflow::io::JoinPath(root_dir, log_dir,
                                                log_file);
  std::ofstream runlog_file;
  runlog_file.open(runlog_path);
  runlog_file << runlog.str();
  runlog_file.close();

  return 0;
}
