import os
import subprocess
import enum
import re
import threading
import time
import argparse
import re
import contextlib
import tensorflow as tf

alpha_config = {
  "quota": {
    "tpu": 50,
    "pod": 2,
  },
}

tpu_configs = {
  "tpu": {
    "accelerator_type": "v3-8",
    "version": "v2-alpha",
    "name": "alpha-{i + 1}",
    "worker_count": 1,
  },
  "pod": {
    "accelerator_type": "v3-32",
    "version": "v2-alpha-pod",
    "name": "pod-{i + 1}",
    "worker_count": 4,
  },
}

deployment_configs = {
  "selfplay": {
    "roles": {
      "train": {
        "count": 1,
        "command": "docker run --rm --privileged --network host --mount type=bind,source=/usr/share/tpu,target=/usr/share/tpu --mount type=bind,source=/lib/libtpu.so,target=/lib/libtpu.so eu.gcr.io/chesscoach/chesscoach-train:selfplay11a_v26",
        "on_error": "dmesg",
      },
      "play": {
        "count": 47,
        "command": "docker run --rm --privileged --network host --mount type=bind,source=/usr/share/tpu,target=/usr/share/tpu --mount type=bind,source=/lib/libtpu.so,target=/lib/libtpu.so eu.gcr.io/chesscoach/chesscoach-play:selfplay11a_v26",
        "on_error": "dmesg",
      },
    },
  },
  "selfplay_pods": {
    "roles": {
      "play": {
        "count": 8,
        "command": "docker run --rm --privileged --network host --mount type=bind,source=/usr/share/tpu,target=/usr/share/tpu --mount type=bind,source=/lib/libtpu.so,target=/lib/libtpu.so eu.gcr.io/chesscoach/chesscoach-play:selfplay11a_v26",
        "on_error": "dmesg",
      },
    },
  },
}

IMAGE_PREFIX = "eu.gcr.io/chesscoach/"
KEY_PATH = "gs://chesscoach-eu/key.json"
KEY_FILENAME = os.path.basename(KEY_PATH)

# Assume that TPU VMs will never be stopped.
class State(enum.Enum):
  DELETED = 0,
  CREATED = 1,
  INITIALIZED = 2,
  WORKING = 3,
  BROKEN = 4,

class Assignment(enum.Enum):
  UNASSIGNED = 0,
  ASSIGNED = 1,

class Tpu:

  def __init__(self, tpu_type, tpu_type_worker_count, name, instance_index, worker_name, worker_index, log_path, state=State.DELETED, assignment=Assignment.UNASSIGNED):
    assert " " not in name
    self.tpu_type = tpu_type
    self.tpu_type_worker_count = tpu_type_worker_count
    self.name = name
    self.instance_index = instance_index
    self.worker_name = worker_name
    self.worker_index = worker_index
    self.state = state
    self.assignment = assignment
    self.log_path = log_path
    self.log_file = None
    self.coordinator = None
    self.coordinator_ssh_lock = None

  def update_state(self, state, reason):
    print(f"[{self.worker_name}] {self.state.name} -> {state.name}: {reason}")
    self.state = state

  def update_assignment(self, assignment, reason):
    print(f"[{self.worker_name}] {self.assignment.name} -> {assignment.name}: {reason}")
    self.assignment = assignment

  def log(self, content):
    if not self.log_file:
      self.log_file = open(self.log_path, "w")
    self.log_file.write(content)
    self.log_file.flush()

  def logline(self, content):
    self.log(content + "\n")

  def sequence_numer(self):
    return ((self.tpu_type_worker_count * self.instance_index) + self.worker_index)

class Role:

  def __init__(self, name, command, on_error):
    self.name = name
    self.command = command
    self.on_error = on_error
    self.tpu = None
    self.process = None
    self.process_reader = None

class Deployment:

  def __init__(self, name):
    self.name = name
    self.roles = []

# Until GKE is supported for alpha TPU VMs, use this utility to manage 10-100 v3-8s.
class AlphaManager:

  def __init__(self, config, tpu_ranges=[(None, None)], deployment_names=None, whatif_mode=True):
    self.config = config
    self.tpu_ranges = tpu_ranges
    self.deployment_names = deployment_names
    self.whatif_mode = whatif_mode
    self.assign_lock = threading.Lock()
    self.wait_seconds = config.training["wait_milliseconds"] / 1000.0
    self.log_prefix = config.join(config.misc["paths"]["alpha_manager"], time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(self.log_prefix)

  def run(self, command, run_async=False):
    process = subprocess.Popen(command, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if run_async:
      return process
    else:
      stdout, _ = process.communicate()
      return process.returncode, stdout

  # Take care: will retry 10 times total if the return code is non-zero.
  def run_ssh(self, tpu, ssh_command, run_async=False):
    # Google Cloud doesn't like initiating SSH connections to multiple workers within a pod simultaneously,
    # returning "unable to queue the operation", so lock around the synchronous portion, but still let multiple
    # async commands run in parallel.
    with tpu.coordinator_ssh_lock or contextlib.nullcontext():
      ssh_command = ssh_command.replace("\"", "\\\"")
      worker_option = f" --worker {tpu.worker_index}" if tpu.coordinator else ""
      command = f"gcloud alpha compute tpus tpu-vm ssh {tpu.name}{worker_option} --quiet --command \"{ssh_command}\""
      tpu.logline("$ " + ssh_command)
      return self.run(command, run_async)

  # Take care: will retry 10 times total if the return code is non-zero.
  def run_ssh_and_log_sync(self, tpu, ssh_command):
    return_code, stdout = self.run_ssh(tpu, ssh_command, run_async=False)
    tpu.log(stdout) # Already includes newlines.
    return return_code, stdout

  # Take care: will retry 10 times total if the return code is non-zero.
  def run_ssh_and_log_async(self, tpu, ssh_command):
    process = self.run_ssh(tpu, ssh_command, run_async=True)
    def read():
      while True:
        line = process.stdout.readline()
        if not line:
          break
        tpu.log(line) # Already includes trailing newline.
      process.stdout.close()
    process_reader = threading.Thread(target=read)
    process_reader.start()
    return process, process_reader

  def check(self, process):
    return_code, _ = process
    return return_code == 0

  def ignore(self, process):
    return True

  def wait_for_state(self, tpu, states):
    while True:
      coordinator_state = tpu.coordinator.state
      if coordinator_state in states:
        return coordinator_state
      time.sleep(1.0)

  def create(self, tpu):
    if tpu.worker_index != 0:
      # Workers 1+ need to rely on their coordinator to create.
      # The coordinator may also have jumped past CREATED, or reached a BROKEN state.
      state = self.wait_for_state(tpu, [State.CREATED, State.INITIALIZED, State.WORKING, State.BROKEN])
      return state != State.BROKEN
    tpu_config = tpu_configs[tpu.tpu_type]
    return self.check(self.run(f"gcloud alpha compute tpus tpu-vm create {tpu.name} --accelerator-type {tpu_config['accelerator_type']} --version {tpu_config['version']}  --quiet"))

  def initialize(self, tpu):
    return (
      # Allow the SSH user to launch containers as non-root via the docker daemon (requires a logout).
      self.check(self.run_ssh_and_log_sync(tpu, "sudo usermod -a -G docker ${USER}")) and
      # Use a key file to authenticate to eu.gcr.io and bridge the credentials to docker.
      self.check(self.run_ssh_and_log_sync(tpu, f"gsutil cp {KEY_PATH} . && gcloud auth activate-service-account --key-file={KEY_FILENAME} && gcloud auth configure-docker --quiet")) and
      # Kill any of our containers already running (they currently outlive SSH sessions on alpha TPU VMs).
      # Return code is 1 if none running, so just don't check.
      # Also use || : until SSH stops trying 10 times.
      self.ignore(self.run_ssh_and_log_sync(tpu, f"docker rm -f $(docker ps | grep \"{IMAGE_PREFIX}\" | cut -d ' ' -f1) || :"))
      )

  # Returns True if not found or successfully deleted; False otherwise.
  def delete(self, tpu):
    if tpu.worker_index != 0:
      # Workers 1+ need to rely on their coordinator to create.
      # The coordinator may also have reached a BROKEN state.
      state = self.wait_for_state(tpu, [State.DELETED, State.BROKEN])
      return state != State.BROKEN
    return_code, stdout = self.run(f"gcloud alpha compute tpus tpu-vm delete {tpu.name} --quiet")
    return (return_code == 0) or ("NOT_FOUND" in stdout)

  def exists(self, tpu):
    return self.check(self.run(f"gcloud alpha compute tpus tpu-vm describe {tpu.name} --quiet"))

  def slice_tpu_ranges(self, all_tpus, tpu_ranges):
    union = set()
    for tpu_range in tpu_ranges:
      range_start, range_finish = tpu_range
      if range_start is not None:
        range_start -= 1
      union.update(all_tpus[slice(range_start, range_finish)])
    return [tpu for tpu in all_tpus if tpu in union]

  def describe_tpus(self, tpus):
    segments = []
    current_segment = []
    def push_segment():
      segments.append(current_segment.copy())
      current_segment.clear()
    for tpu in tpus:
      if current_segment and (
          (tpu.tpu_type != current_segment[-1].tpu_type) or
          (tpu.sequence_numer() != current_segment[-1].sequence_numer() + 1)):
        push_segment()
      current_segment.append(tpu)
    if current_segment:
      push_segment()
    return ", ".join(f"{segment[0].worker_name} to {segment[-1].worker_name}" for segment in segments) + f" ({len(tpus)})"

  def set_up_tpus(self):
    all_tpus = []
    for tpu_type, quota in alpha_config["quota"].items():
      for i in range(quota): # pylint: disable=unused-variable
        name = eval(f'f\"{tpu_configs[tpu_type]["name"]}\"')
        worker_count = tpu_configs[tpu_type]["worker_count"]
        coordinator = None
        coordinator_ssh_lock = None
        for worker_index in range(worker_count):
          worker_name = (name if (worker_count == 1) else f"{name}-{worker_index + 1}")
          log_path = self.config.join(self.log_prefix, worker_name + ".log")
          tpu = Tpu(tpu_type, worker_count, name, i, worker_name, worker_index, log_path)
          all_tpus.append(tpu)
          # For pods, workers 1+ need to rely on worker 0, the coordinator, to create/delete.
          # The coordinator will also see itself as the coordinator.
          if (worker_count > 1) and (worker_index == 0):
            coordinator = tpu
            coordinator_ssh_lock = threading.Lock()
          tpu.coordinator = coordinator
          tpu.coordinator_ssh_lock = coordinator_ssh_lock
    print(f"All TPUs: {self.describe_tpus(all_tpus)}")
    self.tpus = self.slice_tpu_ranges(all_tpus, self.tpu_ranges)
    print(f"Operating on: {self.describe_tpus(self.tpus)}")
    listing_error, listing = self.run("gcloud alpha compute tpus tpu-vm list --quiet")
    if listing_error == 0:
      for tpu in self.tpus:
        if re.search(f"^{tpu.name}\\s", listing, re.MULTILINE):
          tpu.update_state(State.CREATED, "TPU listing")

  def set_up_deployments(self):
    print(f"All deployments: {', '.join(deployment_configs.keys())} ({len(deployment_configs)})")
    if self.deployment_names:
      for name in self.deployment_names:
        if not name in deployment_configs:
          raise ValueError(f"Deployment not in config: {name}")
    managing = list(self.deployment_names if self.deployment_names else deployment_configs.keys())
    print(f"Managing deployments: {', '.join(managing)} ({len(managing)})")
    self.deployments = []
    for name, deployment_config in deployment_configs.items():
      if name not in managing:
        continue
      print(f"Deployment: {name}")
      deployment = Deployment(name)
      self.deployments.append(deployment)
      for role_name, role_config in deployment_config["roles"].items():
        count = role_config["count"]
        command = role_config["command"]
        on_error = role_config.get("on_error", None)
        print(f"  Role: {role_name} x{count}")
        print(f"    command: {command}")
        print(f"    on_error: {on_error}")
        for _ in range(count):
          deployment.roles.append(Role(role_name, command, on_error))

  def assign(self, deployment, role):
    with self.assign_lock:
      tpu = next((t for t in self.tpus if t.state != State.BROKEN and t.assignment == Assignment.UNASSIGNED), None)
      if not tpu:
        raise Exception("Failed to find TPU to assign to deployment")
      tpu.update_assignment(Assignment.ASSIGNED, f"{deployment.name}.{role.name}")
      return tpu

  def discard(self, tpu, reason):
    # Make a final attempt to delete.
    self.delete(tpu)
    with self.assign_lock:
      tpu.update_assignment(Assignment.UNASSIGNED, reason)
      tpu.update_state(State.BROKEN, reason)

  def ensure_working(self, deployment, role):
    while True:
      if role.tpu is None:
        role.tpu = self.assign(deployment, role)
      if role.tpu.state == State.DELETED:
        if self.exists(role.tpu):
          if not self.delete(role.tpu):
            self.discard(role.tpu, "Failed to delete")
            role.tpu = None
            continue
        if not self.create(role.tpu):
          self.discard(role.tpu, "Failed to create")
          role.tpu = None
          continue
        role.tpu.update_state(State.CREATED, "Created")
      if role.tpu.state == State.CREATED:
        if not self.initialize(role.tpu):
          self.discard(role.tpu, "Failed to initialize")
          role.tpu = None
          continue
        role.tpu.update_state(State.INITIALIZED, "Initialized")
      if role.tpu.state == State.INITIALIZED:
        role.process, role.process_reader = self.run_ssh_and_log_async(role.tpu, role.command)
        role.tpu.update_state(State.WORKING, "Started work command")
      if role.tpu.state == State.WORKING:
        return_code = role.process.poll()
        if return_code is not None:
          role.process_reader.join()
          if role.on_error:
            self.run_ssh_and_log_sync(role.tpu, role.on_error)
          if return_code == 0:
            raise NotImplementedError("Unable to handle deployments finishing successfully")
          # Treat the TPU as being in a bad but not broken state - e.g. /dev/accel0 problem - and recreate it.
          if not self.delete(role.tpu):
            self.discard(role.tpu, "Failed to delete")
            role.tpu = None
            continue
          role.tpu.update_state(State.DELETED, "Work command failed")
        else:
          return

  def tick(self):
    threads = []
    for deployment in self.deployments:
      for role in deployment.roles:
        thread = threading.Thread(target=self.ensure_working, args=(deployment, role))
        thread.start()
        threads.append(thread)
    for thread in threads:
      thread.join()

  def command_manage(self):
    self.set_up_tpus()
    self.set_up_deployments()
    if self.whatif_mode:
      print("Pass --confirm to actually run \"manage\" command")
      return
    print("Command: manage")
    while True:
      self.tick()
      time.sleep(self.wait_seconds)

  def command_up(self):
    self.set_up_tpus()
    if self.whatif_mode:
      print("Pass --confirm to actually run \"up\" command")
      return
    print("Command: up")
    threads = []
    def command_up_impl(tpu):
      if not self.exists(tpu) and not self.create(tpu):
        print(f"[{tpu.worker_name}] Failed to create")
        return
      tpu.update_state(State.CREATED, "Exists or created during \"up\" command")
      if not self.initialize(tpu):
        print(f"[{tpu.worker_name}] Failed to initialize")
        return
      tpu.update_state(State.INITIALIZED, "Initialized during \"up\" command")
    for tpu in self.tpus:
      thread = threading.Thread(target=command_up_impl, args=(tpu,))
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()

  def command_down(self):
    self.set_up_tpus()
    if self.whatif_mode:
      print("Pass --confirm to actually run \"down\" command")
      return
    print("Command: down")
    threads = []
    def command_down_impl(tpu):
      if not self.delete(tpu):
        print(f"[{tpu.worker_name}] Failed to delete")
        return
      tpu.update_state(State.DELETED, "Deleting during \"down\" command")
    for tpu in self.tpus:
      thread = threading.Thread(target=command_down_impl, args=(tpu,))
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()

def parse_range(arg):
  match = re.match("^(\\d*)-(\\d*)$", arg)
  if match:
    try:
      start = int(match.group(1))
    except:
      start = None
    try:
      finish = int(match.group(2))
    except:
      finish = None
    if (start is not None and start < 1) or (finish is not None and finish < 1):
      raise ValueError("TPU numbers are base-1")
    if start is not None and finish is not None and finish < start:
      raise ValueError("TPU numbers must be non-empty")
    return (start, finish)
  elif arg == "*":
    return (None, None)
  else:
    match = re.match("^(\\d+)$", arg)
    if match:
      try:
        single = int(match.group(1))
        return (single, single)
      except:
        pass
  raise ValueError("Expected range like 5-10, 5-, -10, *")
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Manage Google Cloud alpha TPU VMs")
  parser.add_argument("-n", "--numbers", help="TPU numbers to operate on, base-1, e.g. 5 5-10 5- -10 *", nargs="+", default=[parse_range("*")], type=parse_range)
  parser.add_argument("-d", "--deployments", help="Deployments to manage", nargs="+")
  parser.add_argument("--confirm", help="Actually run (inverse of what-if mode)", action="store_true")
  parser.add_argument("command", help="Top-level command to run", choices=["manage", "up", "down"])
  args = parser.parse_args()

  tpu_ranges = args.numbers

  deployment_names = args.deployments or None
  if deployment_names and (list(deployment_names) == ["*"]):
    deployment_names = None

  whatif_mode = not args.confirm

  import network
  alpha_manager = AlphaManager(network.config, tpu_ranges, deployment_names, whatif_mode)

  if args.command == "manage":
    alpha_manager.command_manage()
  elif args.command == "up":
    alpha_manager.command_up()
  elif args.command == "down":
    alpha_manager.command_down()