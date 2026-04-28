cmake_minimum_required(VERSION 3.8)
project(curobo_ik_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Generate service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SolveIK.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

# Install Python package manually (avoid ament_python_install_package collision
# with rosidl_generate_interfaces which creates targets with the same name)
execute_process(
  COMMAND python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')"
  OUTPUT_VARIABLE PYTHON_VERSION_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(DIRECTORY curobo_ik_ros/
  DESTINATION lib/${PYTHON_VERSION_DIR}/site-packages/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install executable scripts
install(PROGRAMS
  curobo_ik_ros/ik_service_node.py
  curobo_ik_ros/ik_client.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
