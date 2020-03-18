import time

class ProfileNode:

  def __init__(self, parent, name):
    self.parent = parent
    self.name = name
    self.time = 0.0
    self.count = 0
    self.children = {}
    self.recent_child = None

  def get_child(self, name):
    if ((self.recent_child is not None) and self.recent_child.name == name):
      return self.recent_child
    child = self.children.get(name)
    if (child is None):
      child = ProfileNode(self, name)
      self.children[name] = child
    self.recent_child = child
    return child

  def remove_child(self, name):
    self.children.pop(name)
    self.recent_child = None

class Profiler:

  global_print = True
  active = ProfileNode(None, None)

  def __init__(self, name, *, threshold_time=float("inf")):
    self.name = name
    self.threshold_time = threshold_time
    
  def __enter__(self):
    self.start = time.process_time()
    self.node = Profiler.active.get_child(self.name)
    Profiler.active = self.node

  def __exit__(self, type, value, traceback):
    assert (self.node == Profiler.active)
    self.node.time += (time.process_time() - self.start)
    self.node.count += 1
    
    Profiler.active = Profiler.active.parent

    if ((Profiler.active.parent is None) or (self.node.time >= self.threshold_time)):
      self.consume(self.node)

  def consume(self, node):
    if (Profiler.global_print):
      indent = 0
      counter = node
      while (counter.parent.parent is not None):
        counter = counter.parent
        indent += 1
      self.print(node, indent)
    node.parent.remove_child(node.name)

  def print(self, node, indent):
      indent_string = "--" * indent
      preamble = f"{indent_string}{node.name}:"
      time_part = f"{node.time:.4f}".rjust(60 - len(preamble))
      time_per_part = f"{(node.time / node.count):.4f}".rjust(20)
      count_part = f"({node.count})".rjust(20)
      print(f"{preamble}{time_part}{time_per_part}{count_part}")
      for child in node.children.values():
        self.print(child, indent + 1)

