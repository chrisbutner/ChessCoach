import os
import subprocess
import enum
import re
import threading
import time
import tensorflow as tf

# TODO: Handle pods

alpha_config = {
  "quota": {
    "tpu": 25,
    # "pod": 2,
  },
}

tpu_configs = {
  "tpu": {
    "accelerator_type": "v3-8",
    "version": "v2-alpha",
    "name": "alpha-{i+1}",
  },
  # "pod": {
  #   "accelerator_type": "v3-32",
  #   "version": "v2-alpha-pod",
  #   "name": "pod-{i+1}",
  # },
}

deployment_configs = {
  "main": {
    "roles": {
      "train": {
        "count": 1,
        "command": "docker run --rm --privileged --mount type=bind,source=/usr/share/tpu,target=/usr/share/tpu --mount type=bind,source=/lib/libtpu.so,target=/lib/libtpu.so gcr.io/chesscoach/chesscoach-train:selfplay4_v29",
        "on_error": "dmesg",
      },
      "play": {
        "count": 19,
        "command": "docker run --rm --privileged --mount type=bind,source=/usr/share/tpu,target=/usr/share/tpu --mount type=bind,source=/lib/libtpu.so,target=/lib/libtpu.so gcr.io/chesscoach/chesscoach-play:selfplay4_v29",
        "on_error": "dmesg",
      },
    }
  }
}

IMAGE_PREFIX = "gcr.io/chesscoach/"
KEY_PATH = "gs://chesscoach-eu/key.json"
KEY_FILENAME = os.path.basename(os.path.normpath(KEY_PATH))

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

  def __init__(self, name, log_path, state=State.DELETED, assignment=Assignment.UNASSIGNED):
    assert " " not in name
    self.name = name
    self.state = state
    self.assignment = assignment
    self.log_path = log_path
    self.log_file = None

  def update_state(self, state, reason):
    print(f"[{self.name}] {self.state.name} -> {state.name}: {reason}")
    self.state = state

  def update_assignment(self, assignment, reason):
    print(f"[{self.name}] {self.assignment.name} -> {assignment.name}: {reason}")
    self.assignment = assignment

  def log(self, content):
    if not self.log_file:
      self.log_file = open(self.log_path, "w")
    self.log_file.write(content)
    self.log_file.flush()

  def logline(self, content):
    self.log(content + "\n")

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

  def __init__(self, config, multithread=True):
    self.config = config
    self.multithread = multithread
    self.assign_lock = threading.Lock()
    self.wait_seconds = config.training["wait_milliseconds"] / 1000.0
    self.log_prefix = config.join(config.misc["paths"]["alpha_manager"], time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(self.log_prefix)

  def run(self, command, run_async=False):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if run_async:
      return process
    else:
      stdout, _ = process.communicate()
      return process.returncode, stdout

  # Take care: will retry 10 times total if the return code is non-zero.
  def run_ssh(self, tpu, ssh_command, run_async=False):
    ssh_command = ssh_command.replace("\"", "\\\"")
    command = f"gcloud alpha compute tpus tpu-vm ssh {tpu.name} --quiet --command \"{ssh_command}\""
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

  def create(self, tpu_config, name):
    return self.check(self.run(f"gcloud alpha compute tpus tpu-vm create {name} --accelerator-type {tpu_config['accelerator_type']} --version {tpu_config['version']}  --quiet"))

  def initialize(self, tpu):
    return (
      # Allow the SSH user to launch containers as non-root via the docker daemon (requires a logout).
      self.check(self.run_ssh_and_log_sync(tpu, "sudo usermod -a -G docker ${USER}")) and
      # Use a key file to authenticate to gcr.io and bridge the credentials to docker.
      self.check(self.run_ssh_and_log_sync(tpu, f"gsutil cp {KEY_PATH} . && gcloud auth activate-service-account --key-file={KEY_FILENAME} && gcloud auth configure-docker --quiet")) and
      # Kill any of our containers already running (they currently outlive SSH sessions on alpha TPU VMs).
      # Return code is 1 if none running, so just don't check.
      # Also use || : until SSH stops trying 10 times.
      self.ignore(self.run_ssh_and_log_sync(tpu, f"docker rm -f $(docker ps | grep \"{IMAGE_PREFIX}\" | cut -d ' ' -f1) || :"))
      )

  # Returns True if not found or successfully deleted; False otherwise.
  def delete(self, name):
    return_code, stdout = self.run(f"gcloud alpha compute tpus tpu-vm delete {name} --quiet")
    return (return_code == 0) or ("NOT_FOUND" in stdout)

  def exists(self, name):
    return self.check(self.run(f"gcloud alpha compute tpus tpu-vm describe {name} --quiet"))

  def set_up_tpus(self):
    self.tpus = []
    for i in range(alpha_config["quota"]["tpu"]): # pylint: disable=unused-variable
      name = eval(f'f\"{tpu_configs["tpu"]["name"]}\"')
      log_path = self.config.join(self.log_prefix, name + ".log")
      self.tpus.append(Tpu(name, log_path))
    print(f"TPUs: {self.tpus[0].name} to {self.tpus[-1].name}")
    listing_error, listing = self.run("gcloud alpha compute tpus tpu-vm list --quiet")
    if listing_error == 0:
      for tpu in self.tpus:
        if re.search(f"^{tpu.name}", listing, re.MULTILINE):
          tpu.update_state(State.CREATED, "TPU listing")

  def set_up_deployments(self):
    self.deployments = []
    for name, deployment_config in deployment_configs.items():
      print(f"Deployment: {name}")
      deployment = Deployment(name)
      self.deployments.append(deployment)
      for role_name, role_config in deployment_config["roles"].items():
        count = role_config["count"]
        print(f"Role: {role_name} x{count}")
        for _ in range(count):
          deployment.roles.append(Role(role_name, role_config["command"], role_config.get("on_error", None)))

  def assign(self, deployment, role):
    with self.assign_lock:
      tpu = next((t for t in self.tpus if t.state != State.BROKEN and t.assignment == Assignment.UNASSIGNED), None)
      if not tpu:
        raise Exception("Failed to find TPU to assign to deployment")
      tpu.update_assignment(Assignment.ASSIGNED, f"{deployment.name}.{role.name}")
      return tpu

  def discard(self, tpu, reason):
    # Make a final attempt to delete.
    self.delete(tpu.name)
    with self.assign_lock:
      tpu.update_assignment(Assignment.UNASSIGNED, reason)
      tpu.update_state(State.BROKEN, reason)

  def ensure_working(self, deployment, role):
    while True:
      if role.tpu is None:
        role.tpu = self.assign(deployment, role)
      if role.tpu.state == State.DELETED:
        if self.exists(role.tpu.name):
          if not self.delete(role.tpu.name):
            self.discard(role.tpu, "Failed to delete")
            role.tpu = None
            continue
        if not self.create(tpu_configs["tpu"], role.tpu.name):
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
          if not self.delete(role.tpu.name):
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
        if self.multithread:
          thread = threading.Thread(target=self.ensure_working, args=(deployment, role))
          thread.start()
          threads.append(thread)
        else:
          self.ensure_working(deployment, role)
    for thread in threads:
      thread.join()

  def manage(self):
    self.set_up_tpus()
    self.set_up_deployments()
    while True:
      self.tick()
      time.sleep(self.wait_seconds)

  def down(self):
    self.set_up_tpus()
    threads = []
    for tpu in self.tpus:
      thread = threading.Thread(target=self.delete, args=(tpu.name,))
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()
    
        
