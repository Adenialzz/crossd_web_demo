import sys
import os
import shutil
import os.path as osp
sys.path.append(os.getcwd())
from models.base_model import BaseCVServiceModel
# from utils.video_jumper import VideoJumper
from utils.utils import read_json
from utils.tags_retriever import TopKTagsRetriever
from models.anime_face_model import AnimeFaceModel
from models.tags_model import TagsModel

import numpy as np

class CrossDModel(BaseCVServiceModel):
    def __init__(self, base_out_dir, lib2d_tags_file, lib2d_feats_file, lib2d_images_dir, tsv_file, tags_thr, tags_k, device):
        self.device = device
        self.face_feat_model = AnimeFaceModel(path_prefix=os.getcwd(), device=self.device)
        self.tags_model = TagsModel(path_prefix=os.getcwd(), device=self.device)

        self.base_out_dir = base_out_dir
        self.lib2d_tags_file = lib2d_tags_file
        self.lib2d_feats_file = lib2d_feats_file
        self.lib2d_images_dir = lib2d_images_dir
        self.tsv_file = tsv_file
        self.tags_thr = tags_thr
        self.tags_k = tags_k

        self.load_feats_file()
        self.init_tags_retriever()
        # self.video_jumper = VideoJumper(self.tsv_file)

    def load_feats_file(self):
        # init self.images_list and self.feats_2d
        json_data_2d = read_json(self.lib2d_feats_file)
        self.images_list = []
        feats = []
        for filename, feat in json_data_2d.items():
            self.images_list.append(filename)
            feats.append(feat[0])
        self.feats_2d = np.array(feats)

    def init_tags_retriever(self):
        # init single_tags_retriever
        self.single_tags_retriever = TopKTagsRetriever(self.lib2d_tags_file, self.tags_thr)

    def run(self, image_path):
        # process query_3d image
        query_3d = self.face_feat_model.run(image_path, mode='feat', thr=0.85)
        if query_3d is None:
            print('No Face Detected!')
            return 
        query_3d = np.array(query_3d[0]).reshape(1, 512)
        tags_3d = self.tags_model.run(image_path)

        simis = np.dot(self.feats_2d, query_3d.T).reshape(self.feats_2d.shape[0])

        topk_matched_2d_image_indices = []
        topktags_images_list = []

        # single_tags_retriever = TopKTagsRetriever(self.lib2d_tags_file, tags_thr=0.3)
        srt_topk = self.single_tags_retriever.extract_topk_matched(tags_3d, topk=self.tags_k)

        # prefix = "../crossd_data/all_2d_images/"  # 由保存的json文件决定
        print(srt_topk)
        prefix = ""
        for pair in srt_topk:
            for image_2d in pair[1]:
                try:
                    index2d = self.images_list.index(osp.join(prefix, image_2d))
                except ValueError:      # 可能这张图没有检测到人脸，因此没有feat
                    continue
                topk_matched_2d_image_indices.append(index2d)
                topktags_images_list.append(self.images_list[index2d])
        topktags_simis = simis[topk_matched_2d_image_indices]
        print(topktags_simis.shape)

        top10 = np.argsort(topktags_simis)[:: -1][: 50]   # TODO rename

        out_dir = osp.join(self.base_out_dir, f"{osp.basename(image_path)}")
        if not osp.isdir(out_dir):
            os.mkdir(out_dir)
        shutil.copyfile(image_path, osp.join(out_dir, f"query_{osp.basename(image_path)}"))

        # top10_video_url_dict = dict()
        print(top10)

        top10_image_name = [topktags_images_list[idx] for idx in top10]
        # for k, idx in enumerate(top10):
        #     filename = topktags_images_list[idx]
        #     path = osp.join(self.lib2d_images_dir, filename)
        #     # info = f"top{k+1}_conf{topktags_simis[idx]:.3f}"
        #     # shutil.copyfile(path, osp.join(out_dir, info+osp.basename(path)))
        #     shutil.copyfile(path, osp.join(out_dir, f"top{k+1}.jpg"))
        #     top10_video_url_dict[f'top{k+1}'] = self.video_jumper.imagename2videourl(filename)
        # return top10_video_url_dict
        return top10_image_name

    def post_run(self):
        pass

if __name__ == '__main__':
    image_path = '../crossd_data/all_annoted_2d3d_images/all_original_annoted_3d_images/Dakki_2_3D_1.png'
    # image_path = sys.argv[1]
    model = CrossDModel( base_out_dir='res_query/', lib2d_tags_file='lib2d/lib2d_tags.json', lib2d_feats_file='lib2d/lib2d_feats.json', lib2d_images_dir='lib2d/lib2d_1000_images', tsv_file='ogv_cover_info.tsv', device='cuda:0', tags_thr=0.3, tags_k=3 )
    model.run(image_path)


