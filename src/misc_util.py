import gym
import numpy as np
import os
import pickle
import random
import tempfile
import time
import zipfile
import errno
from PIL import Image

def get_cur_dir():
    return os.getcwd()

def getint(name):
    file_num = name.split('.')
    return int(file_num[0])

def read_dataset(dir_name):
    img_dir = dir_name
    img_rgb_dir = os.path.join(img_dir, "rgb")
    img_lidar_dir = os.path.join(img_dir, "lidar")
    rgb_files = [f for f in os.listdir(img_rgb_dir) if os.path.isfile(os.path.join(img_rgb_dir, f))]
    lidar_files = [f for f in os.listdir(img_lidar_dir) if os.path.isfile(os.path.join(img_lidar_dir, f))]  
    rgb_files.sort(key=getint)
    lidar_files.sort(key=getint)

    if set(rgb_files) != set(lidar_files):
        print "Training data checked: Error"
    else:
        print "Training data checked: OK"
        file_name = rgb_files
        return file_name
    



def load_image(dir_name, img_names):
    img_dir = dir_name
    rgb_images = []
    lidar_images = []
    img_rgb_dir = os.path.join(img_dir, "rgb")
    img_lidar_dir = os.path.join(img_dir, "lidar")

    for name in img_names:
        # Processing 2D images
        img_rgb_file_path = os.path.join(img_rgb_dir, name)
        im_rgb = Image.open(img_rgb_file_path)
        im_rgb_arr = np.array(im_rgb)
        if im_rgb_arr.size == 210*160*3:
            im_rgb_arr = np.reshape(im_rgb_arr, [160, 210, 3]).astype(np.float32)
            im_rgb_arr = im_rgb_arr[:, :, 0] * 0.299 + im_rgb_arr[:, :, 1] * 0.587 + im_rgb_arr[:, :, 2] * 0.114
            im_rgb_arr = np.expand_dims(im_rgb_arr, axis = 2)
            rgb_images.append(im_rgb_arr)
        else:
            print header("Error-1")
        # Processing 3D images, preprocessed, projected into plane pane
        img_lidar_file_path = os.path.join(img_lidar_dir, name)
        im_lidar = Image.open(img_lidar_file_path)
        im_lidar_arr = np.array(im_lidar)
        if im_lidar_arr.size == 210*160*3:
            im_lidar_arr = np.reshape(im_lidar_arr, [160, 210, 3]).astype(np.float32)
            im_lidar_arr = im_lidar_arr[:, :, 0] * 0.299 + im_lidar_arr[:, :, 1] * 0.587 + im_lidar_arr[:, :, 2] * 0.114
            im_lidar_arr = np.expand_dims(im_lidar_arr, axis = 2)
            lidar_images.append(im_lidar_arr)
        else:
            print header("Error-2")
    return rgb_images, lidar_images
    


def next_batch(arr, arr2, index, slice_size, debug=False):
    has_reset = False
    index *= batch_size
    updated_index = index % len(arr)
    if updated_index + slice_size > len(arr):
        updated_index = 0
        has_reset = True
    beg = updated_index
    end = updated_index + slice_size
    if debug:
        print(beg, end)
    return arr[beg:end], arr2[beg:end], has_reset

class Img_Saver(object):
    def __init__(self, Img_dir):
        self.img_dir = Img_dir
        mkdir_p(self.img_dir)
    def save(self, pil_img, file_name, sub_dir):
        img_dir = self.img_dir
        if sub_dir is not None:
            img_dir = os.path.join(self.img_dir, sub_dir)
            mkdir_p(img_dir)
        file_path = os.path.join(img_dir, file_name)
        if pil_img != 'RGB':
            pil_img = pil_img.convert('RGB')
        pil_img.save(file_path)


class Colors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def header(s): print Colors.HEADER + s + Colors.ENDC
def warn(s): print Colors.WARNING + s + Colors.ENDC
def failure(s): print Colors.FAIL + s + Colors.ENDC
def filesave(s) : print Colors.OKGREEN + s + Colors.ENDC

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)


def unpack(seq, sizes):
    """
    Unpack 'seq' into a sequence of lists, with lengths specified by 'sizes'.
    None = just one bare element, not a list

    Example:
    unpack([1,2,3,4,5,6], [3,None,2]) -> ([1,2,3], 4, [5,6])
    """
    seq = list(seq)
    it = iter(seq)
    assert sum(1 if s is None else s for s in sizes) == len(seq), "Trying to unpack %s into %s" % (seq, sizes)
    for size in sizes:
        if size is None:
            yield it.__next__()
        else:
            li = []
            for _ in range(size):
                li.append(it.__next__())
            yield li


class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.

    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...

    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.

    This is generally needed only for environments which wrap C/C++ code, such as MuJoCo
    and Atari.
    """

    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {"_ezpickle_args": self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def pretty_eta(seconds_left):
    """Print the number of seconds in human readable format.

    Examples:
    2 days
    2 hours and 37 minutes
    less than a minute

    Paramters
    ---------
    seconds_left: int
        Number of seconds to be converted to the ETA
    Returns
    -------
    eta: str
        String representing the pretty ETA.
    """
    minutes_left = seconds_left // 60
    seconds_left %= 60
    hours_left = minutes_left // 60
    minutes_left %= 60
    days_left = hours_left // 24
    hours_left %= 24

    def helper(cnt, name):
        return "{} {}{}".format(str(cnt), name, ('s' if cnt > 1 else ''))

    if days_left > 0:
        msg = helper(days_left, 'day')
        if hours_left > 0:
            msg += ' and ' + helper(hours_left, 'hour')
        return msg
    if hours_left > 0:
        msg = helper(hours_left, 'hour')
        if minutes_left > 0:
            msg += ' and ' + helper(minutes_left, 'minute')
        return msg
    if minutes_left > 0:
        return helper(minutes_left, 'minute')
    return 'less than a minute'


class RunningAvg(object):
    def __init__(self, gamma, init_value=None):
        """Keep a running estimate of a quantity. This is a bit like mean
        but more sensitive to recent changes.

        Parameters
        ----------
        gamma: float
            Must be between 0 and 1, where 0 is the most sensitive to recent
            changes.
        init_value: float or None
            Initial value of the estimate. If None, it will be set on the first update.
        """
        self._value = init_value
        self._gamma = gamma

    def update(self, new_val):
        """Update the estimate.

        Parameters
        ----------
        new_val: float
            new observated value of estimated quantity.
        """
        if self._value is None:
            self._value = new_val
        else:
            self._value = self._gamma * self._value + (1.0 - self._gamma) * new_val

    def __float__(self):
        """Get the current estimate"""
        return self._value


class SimpleMonitor(gym.Wrapper):
    def __init__(self, env):
        """Adds two qunatities to info returned by every step:

            num_steps: int
                Number of steps takes so far
            rewards: [float]
                All the cumulative rewards for the episodes completed so far.
        """
        super().__init__(env)
        # current episode state
        self._current_reward = None
        self._num_steps = None
        # temporary monitor state that we do not save
        self._time_offset = None
        self._total_steps = None
        # monitor state
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_end_times = []

    def _reset(self):
        obs = self.env.reset()
        # recompute temporary state if needed
        if self._time_offset is None:
            self._time_offset = time.time()
            if len(self._episode_end_times) > 0:
                self._time_offset -= self._episode_end_times[-1]
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)
        # update monitor state
        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._episode_end_times.append(time.time() - self._time_offset)
        # reset episode state
        self._current_reward = 0
        self._num_steps = 0

        return obs

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        info['steps'] = self._total_steps
        info['rewards'] = self._episode_rewards
        return (obs, rew, done, info)

    def get_state(self):
        return {
            'env_id': self.env.unwrapped.spec.id,
            'episode_data': {
                'episode_rewards': self._episode_rewards,
                'episode_lengths': self._episode_lengths,
                'episode_end_times': self._episode_end_times,
                'initial_reset_time': 0,
            }
        }

    def set_state(self, state):
        assert state['env_id'] == self.env.unwrapped.spec.id
        ed = state['episode_data']
        self._episode_rewards = ed['episode_rewards']
        self._episode_lengths = ed['episode_lengths']
        self._episode_end_times = ed['episode_end_times']


def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def get_wrapper_by_name(env, classname):
    """Given an a gym environment possibly wrapped multiple times, returns a wrapper
    of class named classname or raises ValueError if no such wrapper was applied

    Parameters
    ----------
    env: gym.Env of gym.Wrapper
        gym environment
    classname: str
        name of the wrapper

    Returns
    -------
    wrapper: gym.Wrapper
        wrapper named classname
    """
    currentenv = env
    while True:
        if classname == currentenv.class_name():
            return currentenv
        elif isinstance(currentenv, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)


def relatively_safe_pickle_dump(obj, path, compression=False):
    """This is just like regular pickle dump, except from the fact that failure cases are
    different:

        - It's never possible that we end up with a pickle in corrupted state.
        - If a there was a different file at the path, that file will remain unchanged in the
          even of failure (provided that filesystem rename is atomic).
        - it is sometimes possible that we end up with useless temp file which needs to be
          deleted manually (it will be removed automatically on the next function call)

    The indended use case is periodic checkpoints of experiment state, such that we never
    corrupt previous checkpoints if the current one fails.

    Parameters
    ----------
    obj: object
        object to pickle
    path: str
        path to the output file
    compression: bool
        if true pickle will be compressed
    """
    temp_storage = path + ".relatively_safe"
    if compression:
        # Using gzip here would be simpler, but the size is limited to 2GB
        with tempfile.NamedTemporaryFile() as uncompressed_file:
            pickle.dump(obj, uncompressed_file)
            with zipfile.ZipFile(temp_storage, "w", compression=zipfile.ZIP_DEFLATED) as myzip:
                myzip.write(uncompressed_file.name, "data")
    else:
        with open(temp_storage, "wb") as f:
            pickle.dump(obj, f)
    os.rename(temp_storage, path)


def pickle_load(path, compression=False):
    """Unpickle a possible compressed pickle.

    Parameters
    ----------
    path: str
        path to the output file
    compression: bool
        if true assumes that pickle was compressed when created and attempts decompression.

    Returns
    -------
    obj: object
        the unpickled object
    """

    if compression:
        with zipfile.ZipFile(path, "r", compression=zipfile.ZIP_DEFLATED) as myzip:
            with myzip.open("data") as f:
                return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)
