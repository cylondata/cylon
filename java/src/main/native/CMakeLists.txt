
set(SOURCE_FILES
        "src/AllToAll.cpp"
        "src/AllToAll.h")
add_library(twisterxjni SHARED ${SOURCE_FILES})

if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    target_link_libraries(twisterxjni ${MPI_LIBRARIES})
    target_link_libraries(twisterxjni glog::glog)
endif()