"""Derived Rabbit projection for the two Upsilon trigger channels."""

import tensorflow as tf

from rabbit.mappings.mapping import Mapping
from rabbit.mappings.project import Project


class SumUpsilonChannels(Mapping):
    """Project and sum barrel and high, grouping the two processes per state."""

    def __init__(self, indata, key, output_channel, axis):
        super().__init__(indata, key)
        channels = ("barrel", "high")
        self.projections = [Project(indata, key, channel, axis) for channel in channels]
        axes = [p.channel_info[channel]["axes"] for p, channel in zip(self.projections, channels)]
        if axes[0] != axes[1]:
            raise ValueError(f"Cannot combine trigger channels with different {axis} axes")
        states = ("Y1S", "Y2S", "Y3S", "background")
        process_names = [p.decode() if isinstance(p, bytes) else str(p) for p in indata.procs]
        self.process_pairs = [
            [process_names.index(f"{state}_{channel}") for channel in channels]
            for state in states
        ]
        self.channel_info = {
            output_channel: {"axes": axes[0], "flow": False, "processes": states},
        }

    def compute_flat(self, params, observables=None):
        return tf.add_n([p.compute_flat(params, observables) for p in self.projections])

    def compute_flat_per_process(self, params, observables=None):
        values = tf.add_n([p.compute_flat_per_process(params, observables) for p in self.projections])
        return tf.reduce_sum(tf.gather(values, self.process_pairs, axis=1), axis=2)

    @tf.function
    def get_data(self, data, data_var=None, data_cov_inv=None):
        if data_cov_inv is not None:
            return self._get_data(data, data_var, data_cov_inv)
        values = self.compute_flat(None, data)
        variances = values if data_var is None else self.compute_flat(None, data_var)
        return values, variances, tf.linalg.diag(variances)
