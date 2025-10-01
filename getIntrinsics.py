import pyrealsense2 as rs

try:
    # 1. Create a pipeline objectpipe = rs.pipeline()
    pipeline = rs.pipeline()
    config = rs.config()

    # 2. Configure the streams you want to enableconfig = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # 3. Start the pipelineprofile = pipe.start(config)
    profile = pipeline.start(config)

    # 4. Get the active stream profilesdepth_profile = profile.get_stream(rs.stream.depth)
    color_profile = profile.get_stream(rs.stream.color)
    depth_profile = profile.get_stream(rs.stream.depth)

    # 5. Downcast to video_stream_profile to get the intrinsicsdepth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

    # 6. Print the intrinsic parameters
    '''
    print("Depth Camera Intrinsics:")
    print(f"  Width: {depth_intrinsics.width}")
    print(f"  Height: {depth_intrinsics.height}")
    print(f"  Principal Point (ppx, ppy): ({depth_intrinsics.ppx}, {depth_intrinsics.ppy})")
    print(f"  Focal Length (fx, fy): ({depth_intrinsics.fx}, {depth_intrinsics.fy})")
    print(f"  Distortion Model: {depth_intrinsics.model}")
    print(f"  Distortion Coefficients: {depth_intrinsics.coeffs}")

    print("\nColor Camera Intrinsics:")
    print(f"  Width: {color_intrinsics.width}")
    print(f"  Height: {color_intrinsics.height}")
    print(f"  Principal Point (ppx, ppy): ({color_intrinsics.ppx}, {color_intrinsics.ppy})")
    print(f"  Focal Length (fx, fy): ({color_intrinsics.fx}, {color_intrinsics.fy})")
    print(f"  Distortion Model: {color_intrinsics.model}")
    print(f"  Distortion Coefficients: {color_intrinsics.coeffs}")
    '''
except:
    print("failed lol")

loc_color = "intrinsics/color.txt"
loc_depth = "intrinsics/depth.txt"

with open(loc_color, "w") as f:
    f.write("{} {} {}\n".format(color_intrinsics.fx, 0, color_intrinsics.ppx))
    f.write("{} {} {}\n".format(0, color_intrinsics.fy, color_intrinsics.ppy))
    f.write("{} {} {}".format(0, 0, 1))

with open(loc_depth, "w") as f:
    f.write("{} {} {}\n".format(depth_intrinsics.fx, 0, depth_intrinsics.ppx))
    f.write("{} {} {}\n".format(0, depth_intrinsics.fy, depth_intrinsics.ppy))
    f.write("{} {} {}".format(0, 0, 1))