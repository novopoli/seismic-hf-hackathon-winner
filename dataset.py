from torch.utils.data import Dataset
import h5py
import numpy as np
from scipy.fft import fft, ifft, fftfreq

class MicroseismDataset(Dataset):
    def __init__(self, input_batch, output_batch, selected_sensors_names, number_sources_points,
                 sensors_names, channels, sensors_path, points_path):
        self.input_batch = h5py.File(input_batch,'r')
        self.output_batch = h5py.File(output_batch,'r')
        self.selected_sensors_names = selected_sensors_names
        self.number_sources_points = number_sources_points 
        self.sensors_names = sensors_names
        self.channels = channels
        self.number_tensor_componenets = 6
        self.sensors_coords = self._load_coords(sensors_path)
        self.sources_coords, self.source_names = self._load_coords(points_path, return_names=True)

        self.coord_min = np.array([1683627.62, 6645407.23, 0])
        self.coord_max = np.array([1691827.62, 6654807.23, 0])  # Z фиксирован

        self.data = self.create_dataset()

    def _load_coords(self, path, return_names=False):
        coords = {}
        names = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                name = parts[0]
                x, y, z = map(float, parts[1:4])
                coords[name] = [x, y, z]
                names.append(name)
        return (coords, names) if return_names else coords

    def lowpass_filter(self, signal, fd=500, cutoff=65):
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1 / fd)
        yf[np.abs(xf) > cutoff] = 0
        return ifft(yf).real

    def __len__(self):
        return len(self.data)

    def create_dataset(self):
        self.sens = []
        for ch in self.channels:
            self.sens.extend([s + f'_{ch}' for s in self.selected_sensors_names])

        self.data = []
        for sen in self.sens:
            for pp in self.number_sources_points:
                for comp in range(self.number_tensor_componenets):
                    self.data.append([sen, pp, comp])
        return self.data

    def __getitem__(self, idx):
        sen, pp, comp = self.data[idx]

        input_raw = self.input_batch['Channels'][sen]['data'][pp, comp, :]
        input_filtered = self.lowpass_filter(input_raw)
        input = np.expand_dims(input_filtered, axis=0)

        output = np.expand_dims(self.output_batch['Channels'][sen]['data'][pp, comp, :], axis=0)

        sensor_name = sen.split('_')[0]
        source_name = self.source_names[pp]

        sensor_coords = np.array(self.sensors_coords[sensor_name])
        source_coords = np.array(self.sources_coords[source_name])

        norm_sensor = (sensor_coords - self.coord_min) / (self.coord_max - self.coord_min + 1e-8)
        norm_source = (source_coords - self.coord_min) / (self.coord_max - self.coord_min + 1e-8)

        features = norm_sensor - norm_source

        return input.astype(np.float32), output.astype(np.float32), features.astype(np.float32)
