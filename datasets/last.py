# encoding: utf-8
"""
LaST person re-identification dataset with train-time gender and age labels.
"""

import glob
import json
import os.path as osp

from .bases import BaseImageDataset


class LaST(BaseImageDataset):
    """
    LaST dataset.

    Expected layout:
        last/
          train/**/*.jpg
          val/query/**/*.jpg
          val/gallery/**/*.jpg
          extract_descriptions_train.jsonl

    Training annotations are loaded from extract_descriptions_train.jsonl and
    mapped to integer auxiliary labels for gender and age group.
    """
    dataset_dir = 'last'
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

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(LaST, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
        self.descriptions_path = osp.join(self.dataset_dir, 'extract_descriptions_train.jsonl')

        self._check_before_run()
        self.pid_description = self._load_descriptions()
        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, relabel=True, descriptions=True)
        query = self._process_dir(self.query_dir, relabel=False, descriptions=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False, descriptions=False)

        if verbose:
            print("=> LaST loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        self.num_train_genders, self.num_train_ages = self.get_attribute_info(self.train)
        self.num_genders = len(self.gender_map)
        self.num_ages = len(self.age_map)

    def _check_before_run(self):
        """Check if all files are available before going deeper."""
        for path in [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir, self.descriptions_path]:
            if not osp.exists(path):
                raise RuntimeError("'{}' is not available".format(path))

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
        description = self.pid_description.get(pid)
        if description is None:
            raise KeyError('Missing gender/age description for train pid {}'.format(pid))

        gender_id = self.gender_map.get(description['gender'], -1)
        age_id = self.age_map.get(description['age'], -1)
        return gender_id, age_id

    def _process_dir(self, dir_path, relabel=False, descriptions=False):
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
            gender = -1
            age = -1

            if descriptions:
                gender, age = self._get_descriptions(pid)
            if relabel:
                pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, trackid, gender, age))
        return dataset
