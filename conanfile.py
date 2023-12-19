import os
from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.files import copy


class filterRecipe(ConanFile):
    name = "filter"
    version = "0.1"

    # Optional metadata
    license = "MIT"
    author = "Felix Lubbe"
    url = "https://github.com/flubbe/filter"
    description = "Filters for C++."
    topics = "Kalman filter"

    # Binary configuration
    settings = "os", "arch", "compiler", "build_type"

    # Configure header-only library.
    exports_sources = "include/*", "test/*"
    no_copy_source = True

    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("fmt/[>=10.0.0 <11.0]")
        self.requires("eigen/[>=3.4.0 <4.0]")
        self.test_requires("gtest/[>=1.14.0 <2.0]")

    def build_requirements(self):
        self.tool_requires("cmake/[>=3.22.6 <4.0]")

    def validate(self):
        check_min_cppstd(self, 17)

    def layout(self):
        cmake_layout(self)

    def build(self):
        if not self.conf.get("tools.build:skip_test", default=False):
            cmake = CMake(self)
            cmake.configure(build_script_folder="test")
            cmake.build()
            self.run(os.path.join(self.cpp.build.bindir, "ukf"))

    def package(self):
        # This will also copy the "include" folder
        copy(self, "*.h", self.source_folder, self.package_folder)

    def package_info(self):
        # For header-only packages, libdirs and bindirs are not used
        # so it's necessary to set those as empty.
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []

    def package_id(self):
        self.info.clear()
