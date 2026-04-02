cmake_minimum_required(VERSION 3.5)
project(vikit_py)

if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "/usr/bin/python3")
  set(PYTHON_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Python interpreter for ROS builds" FORCE)
  set(Python3_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "Python interpreter for ROS builds" FORCE)
  list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  list(APPEND CMAKE_IGNORE_PATH
    "$ENV{CONDA_PREFIX}"
    "$ENV{CONDA_PREFIX}/bin"
    "$ENV{CONDA_PREFIX}/include"
    "$ENV{CONDA_PREFIX}/lib")
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# set dependencies
set(dependencies
rclpy
)

#install python modules
# ament_python_install_package(${PROJECT_NAME})

#install python executables
install(PROGRAMS
  src/vikit_py/align_trajectory.py
  src/vikit_py/cpu_info.py
  src/vikit_py/depthmap_utils.py
  src/vikit_py/math_utils.py
  src/vikit_py/ros_node.py
  src/vikit_py/transformations.py
  DESTINATION lib/${PROJECT_NAME}
)

#--------------------------------------------------------------------
# export dependencies
ament_export_dependencies(${dependencies})
ament_package()
