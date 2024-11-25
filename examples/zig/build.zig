const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the library
    const lib = b.addStaticLibrary(.{ .name = "opengjk", .target = target, .optimize = optimize });

    // Add C source files for the library
    lib.addCSourceFile(.{
        .file = b.path("../../openGJK.c"),
    });

    lib.addIncludePath(b.path("../../include/"));

    lib.linkLibC();

    // Install the library
    b.installArtifact(lib);

    // Create the example executable
    const example = b.addExecutable(.{
        .name = "example_lib_opengjk_ce",
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Add C source file for the example
    example.addCSourceFile(.{
        .file = b.path("../c/main.c"),
    });

    example.addIncludePath(b.path("../../include/"));
    // Link the library to the example
    example.linkLibrary(lib);

    // Add math library if needed
    example.linkLibC();
    example.linkSystemLibrary("m");

    // Install the example
    b.installArtifact(example);

    // Create run step for the example
    const run_cmd = b.addRunArtifact(example);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the example");
    run_step.dependOn(&run_cmd.step);
}
