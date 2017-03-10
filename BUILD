# Description:
#   TensorFlow C++ inference example for self driving cars

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_binary(
    name = "autopilot",
    srcs = [
        "run.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow"
    ],
)

cc_binary(
    name = "autopilot-rt",
    srcs = [
        "run_rt.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow"
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
            "**/*.ipynb",
            "**/*.ipynb_checkpoints"
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
