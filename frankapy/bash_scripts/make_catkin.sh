cd catkin_ws
catkin build # replace in conda env to avoid warnings --> catkin build -DCMAKE_CXX_FLAGS="-DBOOST_BIND_GLOBAL_PLACEHOLDERS" 
cd ..