from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import os.path as osp
import cv2
import time
from datetime import timedelta
from models.crossd_model import CrossDModel
from utils.video_jumper import VideoJumper

ALLOWED_EXTENSIONS = set([ "png", "jpg", "jpeg", "JPG", "PNG", "bmp" ])
def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)

# 静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

cv_model = CrossDModel( base_out_dir='static/res_query/', lib2d_tags_file='lib2d/lib2d_tags.json', lib2d_feats_file='lib2d/lib2d_feats.json', lib2d_images_dir='lib2d/lib2d_1000_images', tsv_file='ogv_cover_info.tsv', tags_thr=0.3, tags_k=3, device='cuda:0')
video_jumper = VideoJumper('ogv_cover_info.tsv')

# @app.route("/upload", methods=['POST', 'GET'])
@app.route("/", methods=['POST', 'GET'])
def upload():
    basepath = os.path.dirname(__file__)
    if request.method == "POST":
        f = request.files['file']
        if not ( f and is_allowed_file(f.filename) ):
            return jsonify({
                "error": 1, 
                "msg": f"Only {list(ALLOWED_EXTENSIONS)} are supported currently. Please check your file format.,"
            })
        user_input = request.form.get("name")

        upload_image_path = osp.join("static/images", secure_filename(f.filename))

        f.save(upload_image_path)
        
        # detected_image_path = osp.join("static/images", "output_" + secure_filename(f.filename))
        top50_image_name = cv_model.run(upload_image_path)
        avid_set = set()
        top10_image_name = []
        for imagename in top50_image_name:
            avid = video_jumper.imagename2avid(imagename)
            if avid in avid_set:
                continue
            avid_set.add(avid)
            top10_image_name.append(imagename)
            if len(top10_image_name) == 10:
                break

        # path = "/images/" + "output_" + secure_filename(f.filename)
        input_image_path = osp.join('images/', secure_filename(f.filename))
        # base_out = osp.join('res_query/', secure_filename(f.filename))
        top10_paths_dict = {f"top{i+1}": osp.join('lib2d_images/', top10_image_name[i]) for i in range(len(top10_image_name))}
        top10_video_url_dict = {f"top{i+1}": video_jumper.imagename2videourl(top10_image_name[i]) for i in range(len(top10_image_name))}
        print(top10_paths_dict['top1'])
        return render_template("upload_ok.html", input_image_path=input_image_path, top10_paths_dict=top10_paths_dict, top10_video_url_dict=top10_video_url_dict, val1 = time.time())
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=23340, debug=True)
