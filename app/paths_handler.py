import os
cache_folder = "cache"

"""
5 channel files in stationary
registered
unmixed
clustered

"""

class PathsManager():
    def __init__(self, file_object):

        self._path_parts = file_object.object_name.split("/")
        _file_name_parts = self._path_parts[-1].split('_')
        assert len(_file_name_parts) >= 2, f"{file_object.object_name} doesn't satisfy naming convention"
        dot_split = _file_name_parts[-1].split('.')
        _file_name_parts[-1] = '.'.join(dot_split[0:-1])
        self._file_name_parts = _file_name_parts
        self._file_format = dot_split[-1]

        channels = range(1, 6)
        number_part = self._file_name_parts[1]

        self.file_paths_stationary = [self._get_assumed_path_name_minio(channel) for channel in channels]
        self.file_paths_stationary = [self._get_assumed_path_name_minio(x) for x in channels]
        self.file_path_registered = "/".join(self._path_parts[0:-1]) + "/" + f"{self._file_name_parts[0]}_{self._file_name_parts[1]}.{self._file_format}"
        self.file_name_unmixed = ""

        self.cache_folder = os.path.join(cache_folder, *self._path_parts[0:-1], number_part)
        self.file_paths_cache = [os.path.join(self.cache_folder, self._get_assumed_file_name(channel))
                           for channel in channels]

    def _get_assumed_file_name(self, channel_nr):
        assumed_file_name = f"{self._file_name_parts[0]}_{self._file_name_parts[1]}_{channel_nr}.{self._file_format}"
        return assumed_file_name

    def _get_assumed_path_name_minio(self, channel_nr):
        assumed_file_name = self._get_assumed_file_name(channel_nr)
        assumed_file_path = os.path.join(*self._path_parts[0:-1], assumed_file_name)
        return assumed_file_path
