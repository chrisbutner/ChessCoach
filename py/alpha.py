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
      },
      "play": {
        "count": 19,
        "command": "docker run --rm --privileged --mount type=bind,source=/usr/share/tpu,target=/usr/share/tpu --mount type=bind,source=/lib/libtpu.so,target=/lib/libtpu.so gcr.io/chesscoach/chesscoach-play:selfplay4_v29",
      },
    }
  }
}

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

  def __init__(self, name, state=State.DELETED, assignment=Assignment.UNASSIGNED):
    assert " " not in name
    self.name = name
    self.state = state
    self.assignment = assignment

  def update_state(self, state, reason):
    print(f"[{self.name}] {self.state.name} -> {state.name}: {reason}")
    self.state = state

  def update_assignment(self, assignment, reason):
    print(f"[{self.name}] {self.assignment.name} -> {assignment.name}: {reason}")
    self.assignment = assignment

class Role:

  def __init__(self, name, command):
    self.name = name
    self.command = command
    self.tpu = None
    self.process = None
    self.process_output = None
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

  def run(self, command, run_async=False):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if run_async:
      return process
    else:
      stdout, _ = process.communicate()
      return process.returncode, stdout

  # Take care: will retry 10 times total if the return code is non-zero.
  def run_ssh(self, name, ssh_command, run_async=False):
    ssh_command = ssh_command.replace("\"", "\\\"")
    command = f"gcloud alpha compute tpus tpu-vm ssh {name} --quiet --command \"{ssh_command}\""
    print(f"[{name}] SSH: {ssh_command}")
    return self.run(command, run_async)

  def check(self, process):
    return_code, _ = process
    return return_code == 0

  def create(self, tpu_config, name):
    return self.check(self.run(f"gcloud alpha compute tpus tpu-vm create {name} --accelerator-type {tpu_config['accelerator_type']} --version {tpu_config['version']}  --quiet"))

  def initialize(self, name):
    return (self.check(self.run_ssh(name, "sudo usermod -a -G docker ${USER}")) and
      self.check(self.run_ssh(name, f"gsutil cp {KEY_PATH} . && gcloud auth activate-service-account --key-file={KEY_FILENAME} && gcloud auth configure-docker --quiet")))

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
      self.tpus.append(Tpu(name))
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
          deployment.roles.append(Role(role_name, role_config["command"]))

  def assign(self, deployment, role):
    with self.assign_lock:
      tpu = next((t for t in self.tpus if t.state != State.BROKEN and t.assignment == Assignment.UNASSIGNED), None)
      if not tpu:
        raise Exception("Failed to find TPU to assign to deployment")
      tpu.update_assignment(Assignment.ASSIGNED, f"{deployment.name}.{role.name}")
      return tpu

  def discard(self, tpu, reason):
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
        if not self.initialize(role.tpu.name):
          self.discard(role.tpu, "Failed to initialize")
          role.tpu = None
          continue
        role.tpu.update_state(State.INITIALIZED, "Initialized")
      if role.tpu.state == State.INITIALIZED:
        role.process = self.run_ssh(role.tpu.name, role.command, run_async=True)
        def read():
          role.process_output = role.process.stdout.readlines()
          role.process.stdout.close()
        role.process_reader = threading.Thread(target=read)
        role.process_reader.start()
        role.tpu.update_state(State.WORKING, "Started work command")
      if role.tpu.state == State.WORKING:
        while role.process.poll() is None:
          pass
        return_code = role.process.poll()
        if return_code is not None:
          role.process_reader.join()
          print(f"[{role.tpu.name}] Work command exited: {return_code}")
          print(*role.process_output[-10:], sep="\n")
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
    
        
