import gzip
import json
import os
import numpy as np
import re

path = "./data/datasets/audionav/replica/v1/train_multiple/content"
sd = dict()
total = 0
for file in os.listdir(path):
    # if file == "content":
    #     continue
    filepath = os.path.join(path, file)
    print(file)
    with gzip.open(filepath, "rb") as f:
        data = json.loads(f.read(), encoding="utf-8")

    for e in data['episodes']:
        if e['info']['sound'] not in sd.keys():
            sd[e['info']['sound']] = 1
        else:
            sd[e['info']['sound']] += 1
        total += 1

sound_ids = {}
real_sound_name_ids = {}
for root, dirs, files in os.walk("data/sounds/1s_all"):
    idx = 0
    for file in files:
        sound_name = file.split('.')[0]
        real_sound_name = sound_name #re.sub("^c_|[_0-9]*$", "", sound_name)
        if real_sound_name not in sd.keys():
            continue
        if real_sound_name in real_sound_name_ids.keys():
            sound_ids[sound_name] = real_sound_name_ids[real_sound_name]
        else:
            sound_ids[sound_name] = idx
            real_sound_name_ids[real_sound_name] = idx
            idx += 1

for key, value in sound_ids.items():
    print(key, value)

np.save("data/sounds/sound_ids_1s_all_not_merge_train.npy", sound_ids)
