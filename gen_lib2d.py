import os
import os.path as osp
import json
from models.tags_model import TagsModel
from models.anime_face_model import AnimeFaceModel

def write_json(json_data, out_json):
    with open(out_json, 'w') as f:
        json.dump(json_data, f)

def run_model(model, images_dir, **kwargs):
    lib2d_res = dict()
    for filename in os.listdir(images_dir):
        filepath = osp.join(images_dir, filename, **kwargs)
        feat = model.run(filepath, thr=0.85, mode='feat')
        if feat is not None:
            lib2d_res[filename] = feat

    return lib2d_res

if __name__ == '__main__':
    lib2d_images_dir = 'lib2d/lib2d_1000_images'
    out_feats_json = 'lib2d_feats.json'
    out_tags_json = 'lib2d_tags.json'

    model = AnimeFaceModel()
    feats_res = run_model(model, lib2d_images_dir)
    write_json(feats_res, out_feats_json)
    print(f'{out_feats_json}, num of samples: {len(feats_res)}')

    model = TagsModel(os.getcwd())
    tags_res = run_model(model, lib2d_images_dir)
    write_json(tags_res, out_tags_json)
    print(f'{out_tags_json}, num of samples: {len(tags_res)}')


