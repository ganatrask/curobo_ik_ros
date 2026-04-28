cmake_minimum_required(VERSION 3.8)
project(curobo_ik_ros)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Generate service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SolveIK.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

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
