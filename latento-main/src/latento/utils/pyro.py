from contextlib import contextmanager
import torch
import pyro

import emoji
import random

from scpyro import logger

def create_scoped(id, scoped_function):
    def sample(site_id, fn, *args, **kwargs):
        if site_id == "":
            return scoped_function(id, fn, *args, **kwargs)
        return scoped_function(id + "/" + site_id, fn, *args, **kwargs)
    return sample

class ScopedPyroModule(pyro.nn.PyroModule):
    def __init__(self, _internal_name=None):
        super().__init__(_internal_name)
        self._pyro_distribution_ids = set()

    def init_scoping(self, _internal_name = None):
        if _internal_name is None:
            _internal_name = uuid()
        self._internal_name = self._pyro_name = _internal_name

        self.sample = create_scoped(self._internal_name, pyro.sample)
        self.deterministic = create_scoped(self._internal_name, self.deterministic_)
        self.plate = create_scoped(self._internal_name, pyro.plate)
        self.param = create_scoped(self._internal_name, pyro.param)

    def forward(self, *args):
        raise NotImplementedError

    def scope(self, value):
        if value == "":
            return self._internal_name
        return self._internal_name + "/" + value

    def deterministic_(self, _internal_name, value, event_dim=None, **kwargs):
        """
        Taken from pyro.deterministic, but with support for **kwargs (i.e. for marking sites as auxillary)
        """
        event_dim = value.ndim if event_dim is None else event_dim
        return pyro.sample(_internal_name, pyro.distributions.Delta(value, event_dim=event_dim).mask(False),
                    obs=value, infer={"_deterministic": True, **kwargs})

    def __setattr__(self, _internal_name, value):
        # register as buffer if it is a tensor
        if torch.is_tensor(value):
            self.register_buffer(_internal_name, value)
            return

        if isinstance(value, pyro.distributions.distribution.Distribution):
            self._pyro_distribution_ids.add(_internal_name)
            super().__setattr__(_internal_name, value)
            return

        super().__setattr__(_internal_name, value)

    def to(self, device, **kwargs):
        for distribution_id in self._pyro_distribution_ids:
            # check if independent distribution, in that case we change the base_dist
            if isinstance(self.__getattribute__(distribution_id), pyro.distributions.torch.Independent):
                self.__getattribute__(distribution_id).base_dist = self.__getattribute__(distribution_id).base_dist.to(device)
            else:
                self.__setattr__(distribution_id, self.__getattribute__(distribution_id).to(device))

        # for module in self.modules():
        #     if module != self:
        #         module.to(device)
        super().to(device, **kwargs)

    def descope(self, site_id):
        return site_id.replace(self._internal_name + "/", "")

    @property
    def internal_name(self):
        return self._internal_name

def scope_autoguide(base, _internal_name, *args, **kwargs):
    """
    Autoguides in pyro don't allow you to set their _internal_name, which can cause name conflicts if you have multiple autoguides
    The _internal_name is automatically generated from the class's type
    To fix this, we create a subclass with the desired name. Then once the class is initialized, we again revert the class back to the base autoguide class
    """

    AutoguideSubclass = type(_internal_name, (base,), {})
    guide = AutoguideSubclass(*args, **kwargs)
    guide.__class__ = guide.__class__.__bases__[0]

    return guide
        

def free_tensor(x):
    """
    Brings the tensor to the cpu, detaches the gradients, and converts it to a numpy array
    """
    return x.cpu().detach().numpy()

@contextmanager
def run_on_cpu(tensor_type, modules = None):
    if modules is None:
        modules = []

    if torch.tensor(0).is_cuda: #pylint: disable=not-callable
        old_tensor_type = torch.cuda.FloatTensor
        old_device = torch.device("cuda")
    else:
        old_tensor_type = torch.FloatTensor
        old_device = torch.device("cpu")

    cpu = torch.device("cpu")
        
    torch.set_default_tensor_type(tensor_type)

    new_modules = [module.to(cpu) for module in modules]
    for key in pyro.get_param_store():
        pyro.get_param_store()[key] = pyro.get_param_store()[key].to(cpu)

    try:
        yield new_modules
    finally:
        torch.set_default_tensor_type(old_tensor_type)
        for module in modules:
            module.to(old_device)

        for key in pyro.get_param_store():
            pyro.get_param_store()[key] = pyro.get_param_store()[key].to(old_device)

def clear_param_store(match):
    for key in list(pyro.get_param_store().keys()):
        if match in key:
            logger.info("Deleting parameter %s", key)
            del pyro.get_param_store()[key]

def uuid():
    for _ in range(100):
        name = random.choice(list(emoji.unicode_codes.UNICODE_EMOJI.keys()))

        matched = False
        for key in list(pyro.get_param_store().keys()):
            if name in key:
                matched = True
                break
        if not matched:
            return name

def fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__