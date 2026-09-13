# Taken directly from movi_def_worker.py

import logging

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np


# --- Some configuration values
# the region in which to place objects [(min), (max)]
STATIC_SPAWN_REGION = [(-7, -7, 0), (7, 7, 10)]
DYNAMIC_SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
VELOCITY_RANGE = [(-8., -8., 0.), (8., 8., 0.)]

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"],
                    default="train")
# Configuration for the objects of the scene
parser.add_argument("--min_num_static_objects", type=int, default=10,
                    help="minimum number of static (distractor) objects")
parser.add_argument("--max_num_static_objects", type=int, default=20,
                    help="maximum number of static (distractor) objects")
parser.add_argument("--min_num_dynamic_objects", type=int, default=1,
                    help="minimum number of dynamic (tossed) objects")
parser.add_argument("--max_num_dynamic_objects", type=int, default=3,
                    help="maximum number of dynamic (tossed) objects")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")

parser.add_argument("--camera", choices=["fixed_random", "linear_movement", "linear_movement_linear_lookat", "orbit"],
                    default="orbit")
parser.add_argument("--min_camera_movement", type=float, default=6.0)
parser.add_argument("--max_camera_movement", type=float, default=12.0)
# Fit the main subject region instead of enclosing every moving object.
parser.add_argument("--orbit_radius", type=float, default=6.0,
                    help="minimum reference camera distance in scene units")
parser.add_argument("--camera_motion", choices=["mixed", "orbit", "truck", "dolly", "orbit_dolly"],
                    default="mixed")
parser.add_argument("--subject_fraction", type=float, default=0.7,
                    help="fraction of central static objects used for framing")
parser.add_argument("--min_subject_fill", type=float, default=0.7)
parser.add_argument("--max_subject_fill", type=float, default=0.85)
parser.add_argument("--camera_max_attempts", type=int, default=32)
parser.add_argument("--min_orbit_speed", type=float, default=20.0,
                    help="minimum orbit angular speed in degrees per second")
parser.add_argument("--max_orbit_speed", type=float, default=35.0,
                    help="maximum orbit angular speed in degrees per second")
parser.add_argument("--orbit_elevation", type=float, default=None,
                    help="fixed elevation override; otherwise sample 20 to 40 degrees")
parser.add_argument("--camera_frame_margin", type=float, default=0.04,
                    help="fraction of image reserved on each edge")
parser.add_argument("--camera_clearance", type=float, default=1.5,
                    help="minimum distance to every foreground world-axis bounding box")
parser.add_argument("--lookat_inner_radius", type=float, default=1.0)
parser.add_argument("--lookat_outer_radius", type=float, default=4.0)
parser.add_argument("--lookat_continuation_max", type=float, default=1.0)
# parser.add_argument("--min_motion_blur", type=float, default=0.0)
# parser.add_argument("--max_motion_blur", type=float, default=0.0)
parser.add_argument("--min_motion_blur", type=float, default=8.0)
parser.add_argument("--max_motion_blur", type=float, default=10.0)
parser.add_argument("--min_blur_exposure", type=float, default=0.0,
                    help="minimum image-domain exposure applied only to blurred RGB, in stops")
parser.add_argument("--max_blur_exposure", type=float, default=0.0,
                    help="maximum image-domain exposure applied only to blurred RGB, in stops")


# Configuration for the source of the assets
# parser.add_argument("--kubasic_assets", type=str,
#                     default="gs://kubric-public/assets/KuBasic/KuBasic.json")
# parser.add_argument("--hdri_assets", type=str,
#                     default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
# parser.add_argument("--gso_assets", type=str,
#                     default="gs://kubric-public/assets/GSO/GSO.json")
parser.add_argument("--kubasic_assets", type=str,
                    default="/remote-home/share/cjk/datasets/kubric_assets/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="/remote-home/share/cjk/datasets/kubric_assets/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="/remote-home/share/cjk/datasets/kubric_assets/GSO.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=96, frame_rate=48,
                    resolution=512)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)

if FLAGS.min_motion_blur > FLAGS.max_motion_blur:
  raise ValueError(f"min_motion_blur ({FLAGS.min_motion_blur}) must be <= max_motion_blur ({FLAGS.max_motion_blur}).")
if FLAGS.min_blur_exposure > FLAGS.max_blur_exposure:
  raise ValueError(f"min_blur_exposure ({FLAGS.min_blur_exposure}) must be <= max_blur_exposure ({FLAGS.max_blur_exposure}).")
if FLAGS.lookat_inner_radius > FLAGS.lookat_outer_radius:
  raise ValueError(f"lookat_inner_radius ({FLAGS.lookat_inner_radius}) must be <= lookat_outer_radius ({FLAGS.lookat_outer_radius}).")
if FLAGS.lookat_continuation_max < 0.0:
  raise ValueError(f"lookat_continuation_max ({FLAGS.lookat_continuation_max}) must be >= 0.")

if FLAGS.camera == "orbit":
  orbit_values = [FLAGS.orbit_radius, FLAGS.min_orbit_speed,
                  FLAGS.max_orbit_speed, FLAGS.camera_frame_margin,
                  FLAGS.camera_clearance, FLAGS.subject_fraction,
                  FLAGS.min_subject_fill, FLAGS.max_subject_fill]
  if FLAGS.orbit_elevation is not None:
    orbit_values.append(FLAGS.orbit_elevation)
  if not np.all(np.isfinite(orbit_values)):
    raise ValueError("Orbit parameters must be finite.")
  if FLAGS.orbit_radius <= 0 or FLAGS.camera_clearance <= 0:
    raise ValueError("Orbit radius and camera clearance must be positive.")
  if not 0 <= FLAGS.min_orbit_speed <= FLAGS.max_orbit_speed:
    raise ValueError("Require 0 <= min_orbit_speed <= max_orbit_speed.")
  if FLAGS.frame_rate <= 0 or FLAGS.max_orbit_speed / FLAGS.frame_rate > 1.5:
    raise ValueError("Orbit motion must not exceed 1.5 degrees per frame.")
  if FLAGS.orbit_elevation is not None and not 5 <= FLAGS.orbit_elevation <= 75:
    raise ValueError("Orbit elevation must be between 5 and 75 degrees.")
  if not 0 <= FLAGS.camera_frame_margin < 0.5:
    raise ValueError("Camera frame margin must be in [0, 0.5).")

  if not 0 < FLAGS.subject_fraction <= 1:
    raise ValueError("Subject fraction must be in (0, 1].")
  if not 0 < FLAGS.min_subject_fill <= FLAGS.max_subject_fill < 1:
    raise ValueError("Require 0 < min_subject_fill <= max_subject_fill < 1.")
  if FLAGS.min_subject_fill >= 1 - 2 * FLAGS.camera_frame_margin:
    raise ValueError("Subject fill must leave room for the frame margin.")
  if FLAGS.camera_max_attempts < 1 or FLAGS.frame_end <= FLAGS.frame_start:
    raise ValueError("Require positive camera attempts and at least two frames.")

motion_blur = rng.uniform(FLAGS.min_motion_blur, FLAGS.max_motion_blur)
if motion_blur > 0.0:
  logging.info(f"Using motion blur strength {motion_blur}")
blur_exposure = rng.uniform(FLAGS.min_blur_exposure, FLAGS.max_blur_exposure)
logging.info(f"Using blur-only exposure {blur_exposure:.3f} stops")

simulator = PyBullet(scene, scratch_dir)
# Keep clear render in `rgba`; render blur in a second pass with the same renderer.
renderer = Blender(scene, scratch_dir, use_denoising=True, samples_per_pixel=64,
                   motion_blur=motion_blur)
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)


# --- Populate the scene
# background HDRI
train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
if FLAGS.backgrounds_split == "train":
  logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
  hdri_id = rng.choice(train_backgrounds)
else:
  logging.info("Choosing one of the %d held-out backgrounds...", len(test_backgrounds))
  hdri_id = rng.choice(test_backgrounds)
background_hdri = hdri_source.create(asset_id=hdri_id)
#assert isinstance(background_hdri, kb.Texture)
logging.info("Using background %s", hdri_id)
scene.metadata["background"] = hdri_id
scene.metadata["motion_blur"] = motion_blur
scene.metadata["blur_exposure"] = blur_exposure
renderer._set_ambient_light_hdri(background_hdri.filename)

# Dome
dome = kubasic.create(asset_id="dome", name="dome",
                      friction=1.0,
                      restitution=0.0,
                      static=True, background=True)
assert isinstance(dome, kb.FileBasedObject)
scene += dome
dome_blender = dome.linked_objects[renderer]
texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
texture_node.image = bpy.data.images.load(background_hdri.filename)



def get_linear_camera_motion_start_end(
    movement_speed: float,
    inner_radius: float = 8.,
    outer_radius: float = 12.,
    z_offset: float = 0.1,
):
  """Sample a linear path which starts and ends within a half-sphere shell."""
  while True:
    camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius,
                                                                 outer_radius,
                                                                 z_offset))
    direction = rng.rand(3) - 0.5
    movement = direction / np.linalg.norm(direction) * movement_speed
    camera_end = camera_start + movement
    if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
        camera_end[2] > z_offset):
      return camera_start, camera_end

def get_linear_lookat_motion_start_end(
    inner_radius: float = 1.0,
    outer_radius: float = 6.0,
    continuation_max: float = 1.0,
):
  """Sample a linear path which goes through the workspace center."""
  while True:
    # Sample a point near the workspace center that the path travels through
    camera_through = np.array(
        kb.sample_point_in_half_sphere_shell(0.0, inner_radius, 0.0)
    )
    while True:
      # Sample one endpoint of the trajectory
      camera_start = np.array(
          kb.sample_point_in_half_sphere_shell(0.0, outer_radius, 0.0)
      )
      if camera_start[-1] < inner_radius:
        break

    # Continue the trajectory beyond the point in the workspace center, so the
    # final path passes through that point.
    continuation = rng.rand(1) * continuation_max
    camera_end = camera_through + continuation * (camera_through - camera_start)

    # Second point will probably be closer to the workspace center than the
    # first point.  Get extra augmentation by randomly swapping first and last.
    if rng.rand(1)[0] < 0.5:
      tmp = camera_start
      camera_start = camera_end
      camera_end = tmp
    return camera_start, camera_end


# Camera
logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
if FLAGS.camera == "fixed_random":
  scene.camera.position = kb.sample_point_in_half_sphere_shell(
      inner_radius=7., outer_radius=9., offset=0.1)
  scene.camera.look_at((0, 0, 0))
elif (
    FLAGS.camera == "linear_movement"
    or FLAGS.camera == "linear_movement_linear_lookat"
):

  is_panning = FLAGS.camera == "linear_movement_linear_lookat"
  camera_inner_radius = 6.0 if is_panning else 8.0
  camera_start, camera_end = get_linear_camera_motion_start_end(
      movement_speed=rng.uniform(low=FLAGS.min_camera_movement, high=FLAGS.max_camera_movement),
      inner_radius=camera_inner_radius,
  )
  if is_panning:
    lookat_start, lookat_end = get_linear_lookat_motion_start_end(
        inner_radius=FLAGS.lookat_inner_radius,
        outer_radius=FLAGS.lookat_outer_radius,
        continuation_max=FLAGS.lookat_continuation_max,
    )

  # linearly interpolate the camera position between these two points
  # while keeping it focused on the center of the scene
  # we start one frame early and end one frame late to ensure that
  # forward and backward flow are still consistent for the last and first frames
  for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
    interp = ((frame - FLAGS.frame_start + 1) /
              (FLAGS.frame_end - FLAGS.frame_start + 3))
    scene.camera.position = (interp * np.array(camera_start) +
                             (1 - interp) * np.array(camera_end))
    if is_panning:
      scene.camera.look_at(
          interp * np.array(lookat_start)
          + (1 - interp) * np.array(lookat_end)
      )
    else:
      scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)


# ---- Object placement ----
train_split, test_split = gso.get_test_split(fraction=0.1)
if FLAGS.objects_split == "train":
  logging.info("Choosing one of the %d training objects...", len(train_split))
  active_split = train_split
else:
  logging.info("Choosing one of the %d held-out objects...", len(test_split))
  active_split = test_split



# add STATIC objects
num_static_objects = rng.randint(FLAGS.min_num_static_objects,
                                 FLAGS.max_num_static_objects+1)
logging.info("Randomly placing %d static objects:", num_static_objects)
for i in range(num_static_objects):
  obj = gso.create(asset_id=rng.choice(active_split))
  assert isinstance(obj, kb.FileBasedObject)
  scale = rng.uniform(0.75, 3.0)
  obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
  obj.metadata["scale"] = scale
  scene += obj
  kb.move_until_no_overlap(obj, simulator, spawn_region=STATIC_SPAWN_REGION,
                           rng=rng)
  obj.friction = 1.0
  obj.restitution = 0.0
  obj.metadata["is_dynamic"] = False
  logging.info("    Added %s at %s", obj.asset_id, obj.position)


logging.info("Running 100 frames of simulation to let static objects settle ...")
_, _ = simulator.run(frame_start=-100, frame_end=0)


# stop any objects that are still moving and reset friction / restitution
for obj in scene.foreground_assets:
  if hasattr(obj, "velocity"):
    obj.velocity = (0., 0., 0.)
    obj.friction = 0.5
    obj.restitution = 0.5


dome.friction = FLAGS.floor_friction
dome.restitution = FLAGS.floor_restitution



# Add DYNAMIC objects
num_dynamic_objects = rng.randint(FLAGS.min_num_dynamic_objects,
                                  FLAGS.max_num_dynamic_objects+1)
logging.info("Randomly placing %d dynamic objects:", num_dynamic_objects)
for i in range(num_dynamic_objects):
  obj = gso.create(asset_id=rng.choice(active_split))
  assert isinstance(obj, kb.FileBasedObject)
  scale = rng.uniform(0.75, 3.0)
  obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
  obj.metadata["scale"] = scale
  scene += obj
  kb.move_until_no_overlap(obj, simulator, spawn_region=DYNAMIC_SPAWN_REGION,
                           rng=rng)
  obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
                  [obj.position[0], obj.position[1], 0])
  obj.metadata["is_dynamic"] = True
  logging.info("    Added %s at %s", obj.asset_id, obj.position)



if FLAGS.save_state:
  logging.info("Saving the simulator state to '%s' prior to the simulation.",
               output_dir / "scene.bullet")
  simulator.save_state(output_dir / "scene.bullet")

# Run dynamic objects simulation
logging.info("Running the simulation ...")
animation, collisions = simulator.run(frame_start=0,
                                      frame_end=scene.frame_end+1)

def set_orbit_camera():
  """Choose a complete smooth trajectory using subject framing and box clearance."""
  objects = list(scene.foreground_assets)
  if not objects:
    raise ValueError("Camera framing requires foreground objects.")
  padding = max(1, int(np.ceil(motion_blur / 2)) + 1)
  frames = np.arange(FLAGS.frame_start - padding, FLAGS.frame_end + padding + 1)
  # Check half-frames too; outside the simulation, object poses are held constant.
  check_frames = np.arange(frames[0], frames[-1] + 0.5, 0.5)
  bounds = []
  for frame in check_frames:
    frame_bounds = []
    for obj in objects:
      with obj.at_frame(float(np.clip(frame, 0, scene.frame_end + 1))):
        frame_bounds.append(obj.bbox_3d)
    bounds.append(frame_bounds)
  bounds = np.asarray(bounds)
  if not np.all(np.isfinite(bounds)):
    raise ValueError("Camera framing requires finite object bounds.")
  output = (check_frames >= FLAGS.frame_start) & (check_frames <= FLAGS.frame_end)
  candidates = [i for i, obj in enumerate(objects) if not obj.metadata.get("is_dynamic", False)]
  if not candidates:
    candidates = list(range(len(objects)))
  centers = bounds[output][0].mean(axis=1)
  middle = np.median(centers[candidates], axis=0)
  order = np.argsort(np.linalg.norm(centers[candidates] - middle, axis=1))
  count = max(1, int(np.ceil(len(candidates) * FLAGS.subject_fraction)))
  subjects = np.asarray(candidates)[order[:count]]
  subject_bounds = bounds[:, subjects].reshape(len(check_frames), -1, 3)
  subject_points = subject_bounds[output].reshape(-1, 3)
  center = (subject_points.min(axis=0) + subject_points.max(axis=0)) / 2
  lower, upper = bounds.min(axis=2), bounds.max(axis=2)
  sensor = np.array([scene.camera.sensor_width, scene.camera.sensor_height])
  focal = scene.camera.focal_length / sensor
  duration = (FLAGS.frame_end - FLAGS.frame_start) / FLAGS.frame_rate
  times = (check_frames - FLAGS.frame_start) / FLAGS.frame_rate
  progress = times / duration - 0.5
  target_fill = rng.uniform(FLAGS.min_subject_fill, FLAGS.max_subject_fill)
  modes = ["orbit", "truck", "dolly", "orbit_dolly"]
  # Pick once so difficult modes are not silently replaced by easier ones.
  mode = str(rng.choice(modes)) if FLAGS.camera_motion == "mixed" else FLAGS.camera_motion
  best = None
  for attempt in range(FLAGS.camera_max_attempts):
    elevation_deg = (rng.uniform(20, 40) if FLAGS.orbit_elevation is None
                     else FLAGS.orbit_elevation)
    elevation = np.deg2rad(elevation_deg)
    start_angle = rng.uniform(0, 2 * np.pi)
    speed = (rng.uniform(FLAGS.min_orbit_speed, FLAGS.max_orbit_speed)
             * rng.choice([-1, 1]) if mode in ("orbit", "orbit_dolly") else 0.0)
    dolly = (rng.uniform(0.10, 0.15 if mode == "orbit_dolly" else 0.20)
             * rng.choice([-1, 1]) if mode in ("dolly", "orbit_dolly") else 0.0)
    truck = rng.uniform(0.15, 0.30) * rng.choice([-1, 1]) if mode == "truck" else 0.0
    # Reserve half the attempts for the smallest truck displacement, since
    # close framing may leave too little room for the larger sampled sweeps.
    if mode == "truck" and attempt >= FLAGS.camera_max_attempts // 2:
      truck = float(np.sign(truck) * 0.15)
    angles = start_angle + np.deg2rad(speed) * (times - duration / 2)
    radial = np.column_stack([np.cos(elevation) * np.cos(angles),
                              np.cos(elevation) * np.sin(angles),
                              np.full(len(angles), np.sin(elevation))])
    tangent = np.array([-np.sin(start_angle), np.cos(start_angle), 0.0])
    # Orientation depends on direction, not the distance being fitted.
    quaternions, rotations = [], []
    for direction in radial:
      scene.camera.position = direction
      # Truck keeps a fixed orientation instead of tracking the center.
      scene.camera.look_at((0, 0, 0))
      quaternion = np.asarray(scene.camera.quaternion)
      if quaternions and np.dot(quaternions[-1], quaternion) < 0:
        quaternion = -quaternion
      quaternions.append(quaternion)
      rotations.append(scene.camera.rotation_matrix)
    quaternions = np.asarray(quaternions)
    # Search actual projected framing, with a finite distance and attempt budget.
    for radius in np.geomspace(FLAGS.orbit_radius, max(60.0, FLAGS.orbit_radius), 72):
      distances = radius * (1 + dolly * progress)
      if np.any(distances <= 0):
        continue
      positions = center + distances[:, None] * radial
      positions += radius * truck * progress[:, None] * tangent
      separation = np.maximum(np.maximum(lower - positions[:, None],
                                         positions[:, None] - upper), 0)
      if (np.linalg.norm(separation, axis=2).min() < FLAGS.camera_clearance
          or positions[:, 2].min() < FLAGS.camera_clearance):
        continue
      view_rotations = np.asarray(rotations)
      view_quaternions = quaternions
      local = np.einsum("fpi,fij->fpj", subject_bounds - positions[:, None], view_rotations)
      depth = -local[..., 2]
      if depth.min() <= scene.camera.min_render_distance:
        continue
      projected = local[..., :2] / depth[..., None] * focal
      if mode == "truck":
        # Center the whole truck shot once; its orientation then stays fixed.
        # Perspective can otherwise offset a close subject even at its 3D center.
        offset = (projected.min(axis=(0, 1)) + projected.max(axis=(0, 1))) / 2
        direction = view_rotations[0] @ np.array([*(offset / focal), -1.0])
        scene.camera.position = (0, 0, 0)
        scene.camera.look_at(direction)
        view_rotations = np.broadcast_to(scene.camera.rotation_matrix, (len(positions), 3, 3))
        view_quaternions = np.tile(scene.camera.quaternion, (len(positions), 1))
        local = np.einsum("fpi,fij->fpj", subject_bounds - positions[:, None], view_rotations)
        depth = -local[..., 2]
      if (depth.min() <= scene.camera.min_render_distance
          or depth.max() >= scene.camera.max_render_distance):
        continue
      projected = local[..., :2] / depth[..., None] * focal
      if np.abs(projected).max() > 0.5 - FLAGS.camera_frame_margin:
        continue
      fill = np.ptp(projected[output], axis=1).max(axis=1).mean()
      if not FLAGS.min_subject_fill <= fill <= FLAGS.max_subject_fill:
        continue
      # Limits are per output frame (checks are spaced half a frame apart).
      turns = np.rad2deg(2 * np.arccos(np.clip(
          np.abs(np.sum(view_quaternions[1:] * view_quaternions[:-1], axis=1)), 0, 1))) * 2
      steps = np.diff(positions, axis=0) * 2
      acceleration = np.diff(positions, n=2, axis=0) * 4
      image_steps = np.diff(projected, axis=0) * 2
      image_acceleration = np.diff(projected, n=2, axis=0) * 4
      if (turns.max() > 1.5 + 1e-5
          or np.linalg.norm(steps, axis=1).max() / distances.min() > 0.03
          or np.linalg.norm(acceleration, axis=1).max() / distances.min() > 0.005
          or np.abs(image_steps).max() > 0.08
          or np.abs(image_acceleration).max() > 0.02):
        continue
      score = abs(fill - target_fill)
      if best is None or score < best[0]:
        best = (score, positions.copy(), view_quaternions.copy(), {
            "motion": mode, "center": center.tolist(), "radius": float(radius),
            "angular_speed_deg_s": float(speed), "elevation_deg": float(elevation_deg),
            "start_angle_rad": float(start_angle), "dolly_fraction": float(dolly),
            "truck_fraction": float(truck), "subject_indices": subjects.tolist(),
            "subject_names": [objects[i].name for i in subjects],
            "mean_subject_fill": float(fill), "target_subject_fill": float(target_fill),
            "attempts": attempt + 1,
        })
    if best is not None:
      break
  if best is None:
    raise ValueError(f"No valid {mode} camera trajectory after {FLAGS.camera_max_attempts} attempts; "
                     "reduce subject fill, motion, or clearance constraints.")
  _, positions, quaternions, metadata = best
  # Only commit after the entire trajectory passes; no frame-local corrections.
  for frame, position, quaternion in zip(frames, positions[::2], quaternions[::2]):
    scene.camera.position = position
    scene.camera.quaternion = quaternion
    scene.camera.keyframe_insert("position", int(frame))
    scene.camera.keyframe_insert("quaternion", int(frame))
  scene.metadata["camera_orbit"] = metadata
  logging.info("Camera %s: reference distance %.2f, subject fill %.2f",
               mode, metadata["radius"], metadata["mean_subject_fill"])


if FLAGS.camera == "orbit":
  set_orbit_camera()

# --- Rendering
if FLAGS.save_state:
  logging.info("Saving the renderer state to '%s' ",
               output_dir / "scene.blend")
  renderer.save_state(output_dir / "scene.blend")


logging.info("Rendering the scene ...")
data_stack = renderer.render(
    return_layers=("rgba_sharp", "rgba_blur", "depth", "normal", "object_coordinates", "segmentation")
)
# Keep naming convention: clear image stays under `rgba`.
data_stack["rgba"] = data_stack.pop("rgba_sharp")
# Apply exposure only to the blurred RGB branch. The sharp rgba remains the
# unmodified source used later by Vid2E/ESIM.
if blur_exposure != 0.0:
  blur_rgba = data_stack["rgba_blur"]
  channel_max = np.iinfo(blur_rgba.dtype).max
  adjusted_blur = blur_rgba.astype(np.float32)
  adjusted_blur[..., :3] *= 2.0 ** float(blur_exposure)
  data_stack["rgba_blur"] = np.clip(
      adjusted_blur, 0.0, channel_max).astype(blur_rgba.dtype)


# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
visible_foreground_assets = [asset for asset in scene.foreground_assets
                             if np.max(asset.metadata["visibility"]) > 0]
visible_foreground_assets = sorted(  # sort assets by their visibility
    visible_foreground_assets,
    key=lambda asset: np.sum(asset.metadata["visibility"]),
    reverse=True)

data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    visible_foreground_assets)
scene.metadata["num_instances"] = len(visible_foreground_assets)

# Save to image files
kb.write_image_dict(data_stack, output_dir)
kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                  visible_foreground_assets)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.write_json(filename=output_dir / "metadata.json", data={
    "flags": vars(FLAGS),
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, visible_foreground_assets),
})
kb.write_json(filename=output_dir / "events.json", data={
    "collisions":  kb.process_collisions(
        collisions, scene, assets_subset=visible_foreground_assets),
})

kb.done()