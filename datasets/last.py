# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import json
import os.path as osp

from .bases import BaseImageDataset


class LaST(BaseImageDataset):
    """
    LaST dataset with optional train-set gender and age-group attributes.
    """
    dataset_dir = 'last'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(LaST, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
        self.descriptions_path = osp.join(self.dataset_dir, 'extract_descriptions_train.jsonl')

        self._check_before_run()
        self.pid_begin = pid_begin
        self.pid_description = self._load_descriptions()

        train = self._process_dir(self.train_dir, descriptions=True)
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)

        if verbose:
            print("=> LaST loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper."""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.descriptions_path):
            raise RuntimeError("'{}' is not available".format(self.descriptions_path))

    def _load_descriptions(self):
        pid_description = {}
        with open(self.descriptions_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                pid = int(data['pid'])
                pid_description[pid] = {
                    'gender': data['labels']['gender'],
                    'age': data['labels']['age'],
                }
        return pid_description

    def _get_descriptions(self, pid):
        gender_map = {
            'male': 0,
            'female': 1,
        }
        age_map = {
            'child': 0,
            'youth': 1,
            'adult': 2,
            'elderly': 3,
        }
        description = self.pid_description.get(pid, {})
        gender_id = gender_map.get(description.get('gender'), -1)
        age_id = age_map.get(description.get('age'), -1)
        return gender_id, age_id

    def _process_dir(self, dir_path, descriptions=False):
        pattern = osp.join(dir_path, '**', '*.jpg')
        img_paths = glob.glob(pattern, recursive=True)

        pid_container = set()
        for img_path in sorted(img_paths):
            img_name = osp.basename(img_path)
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in sorted(img_paths):
            img_name = osp.basename(img_path)
            pid = int(img_name.split('_')[0])
            camid = 0
            trackid = 0

            if descriptions:
                gender, age = self._get_descriptions(pid)
                pid = pid2label[pid]
            else:
                gender, age = -1, -1
            dataset.append((img_path, self.pid_begin + pid, camid, trackid, gender, age))
        return dataset
