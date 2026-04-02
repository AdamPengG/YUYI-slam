# judge which cmake codes to use 
if(ROS_EDITION STREQUAL "ROS1")

  # Copyright(c) 2019 livoxtech limited.

  cmake_minimum_required(VERSION 3.0)


  #---------------------------------------------------------------------------------------
  # Start livox_ros_driver2 project
  #---------------------------------------------------------------------------------------
  include(cmake/version.cmake)
  project(livox_ros_driver2 VERSION ${LIVOX_ROS_DRIVER2_VERSION} LANGUAGES CXX)
  message(STATUS "livox_ros_driver2 version: ${LIVOX_ROS_DRIVER2_VERSION}")

  #---------------------------------------------------------------------------------------
  # Add ROS Version MACRO
  #---------------------------------------------------------------------------------------
  add_definitions(-DBUILDING_ROS1)

  #---------------------------------------------------------------------------------------
  # find package and the dependecy
  #---------------------------------------------------------------------------------------
  find_package(Boost 1.54 REQUIRED COMPONENTS
    system
    thread
    chrono
  )

  ## Find catkin macros and libraries
  ## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
  ## is used, also find other catkin packages
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    sensor_msgs
    std_msgs
    message_generation
    rosbag
    pcl_ros
  )

  ## Find pcl lib
  find_package(PCL REQUIRED)

  ## Generate messages in the 'msg' folder
  add_message_files(FILES
    CustomPoint.msg
    CustomMsg.msg
  #   Message2.msg
  )

  ## Generate added messages and services with any dependencies listed here
  generate_messages(DEPENDENCIES
    std_msgs
  )

  find_package(PkgConfig)
  pkg_check_modules(APR apr-1)
  if (APR_FOUND)
    message(${APR_INCLUDE_DIRS})
    message(${APR_LIBRARIES})
  endif (APR_FOUND)

  ###################################
  ## catkin specific configuration ##
  ###################################
  ## The catkin_package macro generates cmake config files for your package
  ## Declare things to be passed to dependent projects
  ## INCLUDE_DIRS: uncomment this if your package contains header files
  ## LIBRARIES: libraries you create in this project that dependent projects als    o need
  ## CATKIN_DEPENDS: catkin_packages dependent projects also need
  ## DEPENDS: system dependencies of this project that dependent projects also n    eed
  catkin_package(CATKIN_DEPENDS
    roscpp rospy std_msgs message_runtime
    pcl_ros
  )

  #---------------------------------------------------------------------------------------
  # Set default build to release
  #---------------------------------------------------------------------------------------
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
  endif()

  #---------------------------------------------------------------------------------------
  # Compiler config
  #---------------------------------------------------------------------------------------
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  ## make sure the livox_lidar_sdk_static library is installed
  find_library(LIVOX_LIDAR_SDK_LIBRARY  liblivox_lidar_sdk_static.a    /usr/local/lib)

  ## PCL library
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})

  #---------------------------------------------------------------------------------------
  # generate excutable and add libraries
  #---------------------------------------------------------------------------------------
  add_executable(${PROJECT_NAME}_node
    ""
  )

  #---------------------------------------------------------------------------------------
  # precompile macro and compile option
  #---------------------------------------------------------------------------------------
  target_compile_options(${PROJECT_NAME}_node
    PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall>
  )

  #---------------------------------------------------------------------------------------
  # add projects that depend on
  #---------------------------------------------------------------------------------------
  add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_generate_messages_cpp)

  #---------------------------------------------------------------------------------------
  # source file
  #---------------------------------------------------------------------------------------
  target_sources(${PROJECT_NAME}_node
    PRIVATE
    src/driver_node.cpp
    src/lds.cpp
    src/lds_lidar.cpp
    src/lddc.cpp
    src/livox_ros_driver2.cpp

    src/comm/comm.cpp
    src/comm/ldq.cpp
    src/comm/semaphore.cpp
    src/comm/lidar_imu_data_queue.cpp
    src/comm/cache_index.cpp
    src/comm/pub_handler.cpp

    src/parse_cfg_file/parse_cfg_file.cpp
    src/parse_cfg_file/parse_livox_lidar_cfg.cpp

    src/call_back/lidar_common_callback.cpp
    src/call_back/livox_lidar_callback.cpp
  )

  #---------------------------------------------------------------------------------------
  # include file
  #---------------------------------------------------------------------------------------
  target_include_directories(${PROJECT_NAME}_node
    PUBLIC
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${APR_INCLUDE_DIRS}
    3rdparty
    src
  )

  #---------------------------------------------------------------------------------------
  # link libraries
  #---------------------------------------------------------------------------------------
  target_link_libraries(${PROJECT_NAME}_node
    ${LIVOX_LIDAR_SDK_LIBRARY}
    ${Boost_LIBRARY}
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${APR_LIBRARIES}
  )


  #---------------------------------------------------------------------------------------
  # Install
  #---------------------------------------------------------------------------------------

  install(TARGETS ${PROJECT_NAME}_node
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  install(DIRECTORY launch_ROS1/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch_ROS1
  )

  #---------------------------------------------------------------------------------------
  # end of CMakeList.txt
  #---------------------------------------------------------------------------------------


else(ROS_EDITION STREQUAL "ROS2")

  # Copyright(c) 2020 livoxtech limited.

  cmake_minimum_required(VERSION 3.14)
  project(livox_ros_driver2)

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

  # Default to C99
  if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
  endif()

  # Default to C++14
  if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
  endif()

  list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake/modules")

  if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
  endif()

  # Printf version info
  include(cmake/version.cmake)
  project(${PROJECT_NAME} VERSION ${LIVOX_ROS_DRIVER2_VERSION} LANGUAGES CXX)
  message(STATUS "${PROJECT_NAME} version: ${LIVOX_ROS_DRIVER2_VERSION}")

  #---------------------------------------------------------------------------------------
  # Add ROS Version MACRO
  #---------------------------------------------------------------------------------------
  add_definitions(-DBUILDING_ROS2)

  # The message definitions are enough for FAST-LIVO2 builds. The driver node
  # itself is only compiled when the optional Livox-SDK2 dependency is present.
  find_package(ament_cmake REQUIRED)
  find_package(builtin_interfaces REQUIRED)
  find_package(rosidl_default_generators REQUIRED)
  find_package(std_msgs REQUIRED)

  # generate custom msg headers
  set(LIVOX_INTERFACES livox_interfaces2)
  rosidl_generate_interfaces(${LIVOX_INTERFACES}
    "msg/CustomPoint.msg"
    "msg/CustomMsg.msg"
    DEPENDENCIES builtin_interfaces std_msgs
    LIBRARY_NAME ${PROJECT_NAME}
  )

  option(BUILD_LIVOX_DRIVER "Build the Livox driver node when Livox-SDK2 is available" ON)

  ament_export_dependencies(rosidl_default_runtime)

  if(BUILD_LIVOX_DRIVER)
    find_package(PCL QUIET)
    find_package(pcl_conversions QUIET)
    find_package(rcl_interfaces QUIET)
    find_package(rclcpp QUIET)
    find_package(rclcpp_components QUIET)
    find_package(rcutils QUIET)
    find_package(sensor_msgs QUIET)

    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(APR QUIET apr-1)
    endif()

    find_library(LIVOX_LIDAR_SDK_LIBRARY
      NAMES livox_lidar_sdk_shared liblivox_lidar_sdk_shared.so
      HINTS /usr/local/lib
    )
    find_path(LIVOX_LIDAR_SDK_INCLUDE_DIR
      NAMES livox_lidar_api.h livox_lidar_def.h
      HINTS /usr/local/include
    )

    if(PCL_FOUND
      AND pcl_conversions_FOUND
      AND rcl_interfaces_FOUND
      AND rclcpp_FOUND
      AND rclcpp_components_FOUND
      AND rcutils_FOUND
      AND sensor_msgs_FOUND
      AND LIVOX_LIDAR_SDK_LIBRARY
      AND LIVOX_LIDAR_SDK_INCLUDE_DIR)

      link_directories(${PCL_LIBRARY_DIRS})
      add_definitions(${PCL_DEFINITIONS})

      add_library(${PROJECT_NAME} SHARED
        src/livox_ros_driver2.cpp
        src/lddc.cpp
        src/driver_node.cpp
        src/lds.cpp
        src/lds_lidar.cpp

        src/comm/comm.cpp
        src/comm/ldq.cpp
        src/comm/semaphore.cpp
        src/comm/lidar_imu_data_queue.cpp
        src/comm/cache_index.cpp
        src/comm/pub_handler.cpp

        src/parse_cfg_file/parse_cfg_file.cpp
        src/parse_cfg_file/parse_livox_lidar_cfg.cpp

        src/call_back/lidar_common_callback.cpp
        src/call_back/livox_lidar_callback.cpp
      )

      rosidl_get_typesupport_target(cpp_typesupport_target
        ${LIVOX_INTERFACES} "rosidl_typesupport_cpp")

      target_include_directories(${PROJECT_NAME} PUBLIC
        ${APR_INCLUDE_DIRS}
        ${LIVOX_LIDAR_SDK_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        3rdparty
        src
      )

      ament_target_dependencies(${PROJECT_NAME}
        pcl_conversions
        rcl_interfaces
        rclcpp
        rclcpp_components
        rcutils
        sensor_msgs
        std_msgs
      )

      target_link_libraries(${PROJECT_NAME}
        "${cpp_typesupport_target}"
        ${APR_LIBRARIES}
        ${LIVOX_LIDAR_SDK_LIBRARY}
        ${PCL_LIBRARIES}
      )

      rclcpp_components_register_node(${PROJECT_NAME}
        PLUGIN "livox_ros::DriverNode"
        EXECUTABLE ${PROJECT_NAME}_node
      )

      install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
      )
    else()
      message(WARNING
        "Livox-SDK2 or driver build dependencies were not found. "
        "Building a message-only livox_ros_driver2 package.")
    endif()
  else()
    message(STATUS "BUILD_LIVOX_DRIVER=OFF, building message-only livox_ros_driver2 package.")
  endif()

  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # uncomment the line when a copyright and license is not present in all source files
    #set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # uncomment the line when this package is not in a git repo
    #set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
  endif()

  install(DIRECTORY
    config
    launch_ROS2
    DESTINATION share/${PROJECT_NAME}
  )

  ament_package()

endif()
