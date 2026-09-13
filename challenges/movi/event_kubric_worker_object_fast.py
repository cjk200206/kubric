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
# DYNAMIC_TARGET_REGION = [(-1.5, -1.5, 0.), (1.5, 1.5, 0.)]
DYNAMIC_TARGET_REGION = [(-2.5, -2.5, 0.), (2.5, 2.5, 0.)]

# Object-focused dataset profile. Adjust these defaults directly when needed.
OBJECT_FOCUSED_CAMERA_MODE = "linear_movement"
MIN_CAMERA_MOVEMENT = 0.0
MAX_CAMERA_MOVEMENT = 4.0
MIN_OBJECT_SPEED = 12.0
MAX_OBJECT_SPEED = 16.0
MIN_MOTION_BLUR = 2.0
MAX_MOTION_BLUR = 5.0

# underexpose
# MIN_BLUR_EXPOSURE = -3.5
# MAX_BLUR_EXPOSURE = -1.5

# overexpose
# MIN_BLUR_EXPOSURE = 1.0
# MAX_BLUR_EXPOSURE = 2.5

# normal exposure
MIN_BLUR_EXPOSURE = 0
MAX_BLUR_EXPOSURE = 0

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"],
                    default="train")
# Configuration for the objects of the scene
parser.add_argument("--min_num_static_objects", type=int, default=4,
                    help="minimum number of static (distractor) objects")
parser.add_argument("--max_num_static_objects", type=int, default=8,
                    help="maximum number of static (distractor) objects")
parser.add_argument("--min_num_dynamic_objects", type=int, default=5,
                    help="minimum number of dynamic (tossed) objects")
parser.add_argument("--max_num_dynamic_objects", type=int, default=5,
                    help="maximum number of dynamic (tossed) objects")
parser.add_argument("--min_object_speed", type=float, default=MIN_OBJECT_SPEED,
                    help="minimum initial XY speed for dynamic objects in world units per second")
parser.add_argument("--max_object_speed", type=float, default=MAX_OBJECT_SPEED,
                    help="maximum initial XY speed for dynamic objects in world units per second")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")

parser.add_argument("--camera", choices=["fixed_random", "linear_movement", "linear_movement_linear_lookat"],
                    default="fixed_random")
parser.add_argument("--min_camera_movement", type=float, default=MIN_CAMERA_MOVEMENT)
parser.add_argument("--max_camera_movement", type=float, default=MAX_CAMERA_MOVEMENT)
parser.add_argument("--lookat_inner_radius", type=float, default=1.0)
parser.add_argument("--lookat_outer_radius", type=float, default=4.0)
parser.add_argument("--lookat_continuation_max", type=float, default=1.0)
# parser.add_argument("--min_motion_blur", type=float, default=0.0)
# parser.add_argument("--max_motion_blur", type=float, default=0.0)
parser.add_argument("--min_motion_blur", type=float, default=MIN_MOTION_BLUR)
parser.add_argument("--max_motion_blur", type=float, default=MAX_MOTION_BLUR)
parser.add_argument("--min_blur_exposure", type=float, default=MIN_BLUR_EXPOSURE,
                    help="minimum image-domain exposure applied only to blurred RGB, in stops")
parser.add_argument("--max_blur_exposure", type=float, default=MAX_BLUR_EXPOSURE,
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
# sample.py always forwards a camera mode; this worker deliberately owns the effective mode.
FLAGS.camera = OBJECT_FOCUSED_CAMERA_MODE

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)

if FLAGS.min_motion_blur > FLAGS.max_motion_blur:
  raise ValueError(f"min_motion_blur ({FLAGS.min_motion_blur}) must be <= max_motion_blur ({FLAGS.max_motion_blur}).")
if FLAGS.min_object_speed < 0.0 or FLAGS.min_object_speed > FLAGS.max_object_speed:
  raise ValueError(f"Expected 0 <= min_object_speed <= max_object_speed, got {FLAGS.min_object_speed} and {FLAGS.max_object_speed}.")
if FLAGS.min_blur_exposure > FLAGS.max_blur_exposure:
  raise ValueError(f"min_blur_exposure ({FLAGS.min_blur_exposure}) must be <= max_blur_exposure ({FLAGS.max_blur_exposure}).")
if FLAGS.lookat_inner_radius > FLAGS.lookat_outer_radius:
  raise ValueError(f"lookat_inner_radius ({FLAGS.lookat_inner_radius}) must be <= lookat_outer_radius ({FLAGS.lookat_outer_radius}).")
if FLAGS.lookat_continuation_max < 0.0:
  raise ValueError(f"lookat_continuation_max ({FLAGS.lookat_continuation_max}) must be >= 0.")

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
  scene.metadata["camera_movement"] = 0.0
  scene.camera.position = kb.sample_point_in_half_sphere_shell(
      inner_radius=7., outer_radius=9., offset=0.1)
  scene.camera.look_at((0, 0, 0))
elif (
    FLAGS.camera == "linear_movement"
    or FLAGS.camera == "linear_movement_linear_lookat"
):

  is_panning = FLAGS.camera == "linear_movement_linear_lookat"
  camera_inner_radius = 6.0 if is_panning else 8.0
  camera_movement = rng.uniform(low=FLAGS.min_camera_movement,
                                high=FLAGS.max_camera_movement)
  scene.metadata["camera_movement"] = camera_movement
  camera_start, camera_end = get_linear_camera_motion_start_end(
      movement_speed=camera_movement,
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
  scale = rng.uniform(1.25, 2.5)
  obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
  obj.metadata["scale"] = scale
  scene += obj
  kb.move_until_no_overlap(obj, simulator, spawn_region=DYNAMIC_SPAWN_REGION,
                           rng=rng)
  motion_target = rng.uniform(*DYNAMIC_TARGET_REGION)
  motion_direction = motion_target - [obj.position[0], obj.position[1], 0]
  motion_direction[2] = 0.0
  motion_direction /= np.linalg.norm(motion_direction)
  initial_speed = rng.uniform(FLAGS.min_object_speed, FLAGS.max_object_speed)
  obj.velocity = motion_direction * initial_speed
  obj.metadata["is_dynamic"] = True
  obj.metadata["motion_target"] = motion_target
  obj.metadata["initial_speed"] = initial_speed
  obj.metadata["initial_velocity"] = obj.velocity
  logging.info("    Added %s at %s", obj.asset_id, obj.position)



if FLAGS.save_state:
  logging.info("Saving the simulator state to '%s' prior to the simulation.",
               output_dir / "scene.bullet")
  simulator.save_state(output_dir / "scene.bullet")

# Run dynamic objects simulation
logging.info("Running the simulation ...")
animation, collisions = simulator.run(frame_start=0,
                                      frame_end=scene.frame_end+1)

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

# Apply adverse exposure only to the blurred RGB branch. The sharp `rgba`
# remains untouched and is the source used later by Vid2E/ESIM.
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