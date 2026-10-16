"""Application-owned shared-memory transport for policy observations."""

from dataclasses import dataclass
import mmap
import os
import tempfile

import numpy as np


SHM_RING_TRANSPORT = "shm_ring_v1"
_ALIGNMENT = 64
_CAMERA_FIELDS = (
    "camera_wrist_right",
    "camera_wrist_left",
    "camera_head_left",
    "camera_head_right",
    "camera_ceiling",
)


class ShmRingCapacityError(RuntimeError):
    """Raised when an observation no longer fits the initialized ring."""


@dataclass(frozen=True)
class ShmWriteResult:
    """Descriptor for one observation copied into a ring slot."""

    descriptor: dict
    resource: tuple[str, str, int]
    bytes_written: int


def _align(value, alignment=_ALIGNMENT):
    return (int(value) + alignment - 1) // alignment * alignment


def _reshape_list_field(field, element_shape, name):
    offsets = field.offsets.to_numpy(zero_copy_only=True)
    expected_size = int(np.prod(element_shape))
    if len(field) == 0 or np.any(np.diff(offsets) != expected_size):
        raise ValueError(
            f"Observation field {name!r} must contain fixed-size "
            f"elements with shape {element_shape}"
        )

    start = int(offsets[0])
    stop = int(offsets[-1])
    values = field.values.slice(start, stop - start).to_numpy(zero_copy_only=True)
    return values.reshape(len(field), *element_shape)


def _observation_arrays(observation, metadata):
    field_names = {field.name for field in observation.type.fields}
    position = observation.field("position")
    if len(position) == 0:
        raise ValueError("Observation position history cannot be empty")
    position_offsets = position.offsets.to_numpy(zero_copy_only=True)
    position_size = int(position_offsets[1] - position_offsets[0])

    arrays = {
        "position": _reshape_list_field(
            position,
            (position_size,),
            "position",
        )
    }
    for name in _CAMERA_FIELDS:
        if name not in field_names:
            continue
        height = int(metadata[f"{name}.height"])
        width = int(metadata[f"{name}.width"])
        arrays[name] = _reshape_list_field(
            observation.field(name),
            (height, width, 3),
            name,
        )

    prompt = None
    if "task_prompt" in field_names:
        prompt = observation[-1]["task_prompt"].as_py()
    return arrays, prompt


class SharedMemoryRingWriter:
    """Copy Dora-backed Arrow fields into a fixed application-owned mmap ring."""

    def __init__(self, shared_dir, slot_count=3):
        """Create an uninitialized ring that allocates on its first write."""
        if slot_count < 3:
            raise ValueError("Shared-memory policy ring requires at least 3 slots")
        self.shared_dir = shared_dir
        self.slot_count = int(slot_count)
        self.path = None
        self.slot_size = 0
        self.ring_size = 0
        self.mapping = None
        self.next_slot = 0
        self.sequence = 0

    @property
    def enabled(self):
        """Return whether the backing mmap has been allocated."""
        return self.mapping is not None

    def _initialize(self, payload_size):
        slot_size = _align(max(1, payload_size), mmap.PAGESIZE)
        ring_size = slot_size * self.slot_count
        fp = tempfile.NamedTemporaryFile(
            prefix="policy-observation-",
            suffix=".ring",
            dir=self.shared_dir,
            delete=False,
        )
        try:
            fp.truncate(ring_size)
            fp.flush()
            mapping = mmap.mmap(fp.fileno(), ring_size, access=mmap.ACCESS_WRITE)
        finally:
            fp.close()

        self.path = fp.name
        self.slot_size = slot_size
        self.ring_size = ring_size
        self.mapping = mapping

    def _reserve_slot(self, protected_resources):
        protected_slots = {
            int(resource[2])
            for resource in protected_resources
            if resource
            and len(resource) == 3
            and resource[0] == SHM_RING_TRANSPORT
            and resource[1] == self.path
        }
        for _ in range(self.slot_count):
            slot = self.next_slot
            self.next_slot = (self.next_slot + 1) % self.slot_count
            if slot not in protected_slots:
                return slot
        raise RuntimeError("No free shared-memory policy ring slot")

    def write(self, observation, metadata, protected_resources=()):
        """Copy one observation into a slot and return its wire descriptor."""
        arrays, prompt = _observation_arrays(observation, metadata)
        layout = {}
        payload_size = 0
        for name, array in arrays.items():
            array = np.asarray(array)
            offset = _align(payload_size)
            layout[name] = {
                "offset": offset,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "nbytes": int(array.nbytes),
            }
            payload_size = offset + array.nbytes

        if self.mapping is None:
            self._initialize(payload_size)
        if payload_size > self.slot_size:
            raise ShmRingCapacityError(
                f"Observation payload grew from slot size {self.slot_size} "
                f"to {payload_size} bytes"
            )

        slot = self._reserve_slot(protected_resources)
        slot_base = slot * self.slot_size
        for name, array in arrays.items():
            field = layout[name]
            source = np.asarray(array)
            target = np.ndarray(
                source.shape,
                dtype=source.dtype,
                buffer=self.mapping,
                offset=slot_base + field["offset"],
            )
            np.copyto(target, source, casting="no")

        self.sequence += 1
        descriptor = {
            "path": self.path,
            "ring_size": self.ring_size,
            "slot_count": self.slot_count,
            "slot_size": self.slot_size,
            "slot": slot,
            "sequence": self.sequence,
            "payload_size": payload_size,
            "fields": layout,
            "task_prompt": prompt,
        }
        return ShmWriteResult(
            descriptor=descriptor,
            resource=(SHM_RING_TRANSPORT, self.path, slot),
            bytes_written=payload_size,
        )

    def close(self):
        """Close and unlink the application-owned ring."""
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.path is not None:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            self.path = None
