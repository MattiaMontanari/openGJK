# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                                    #####        # #    #                #
#        ####  #####  ###### #    # #     #       # #   #                 #
#       #    # #    # #      ##   # #             # #  #                  #
#       #    # #    # #####  # #  # #  ####       # ###                   #
#       #    # #####  #      #  # # #     # #     # #  #                  #
#       #    # #      #      #   ## #     # #     # #   #                 #
#        ####  #      ###### #    #  #####   #####  #    #                #
#                                                                         #
#           Mattia Montanari    |   University of Oxford 2018             #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


project(openGJKdemo)

message( "[${CMAKE_PROJECT_NAME}] Compiling the executable ..")

# Set source file
set(SOURCE_FILES main.c )

# Create the executable
add_executable(demo ${SOURCE_FILES})

# Link to openGJK
target_link_libraries(demo openGJKlib )

# Report
message( ".. executable DONE!")