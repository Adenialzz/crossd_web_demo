import os.path as osp
import pandas as pd

class VideoJumper:
    def __init__(self, tsv_file):
        self._imagename2videourl= dict()
        self.load_info(tsv_file)

    def imagename2videourl(self, imageurl):
        if imageurl[-1] == '1':   # handle error like xxxx.jpg.1
             print(imageurl[:-2])
             return self._imagename2videourl[imageurl[:-2]]
        return self._imagename2videourl[imageurl]

    def load_info(self, tsv_file):
        # cover_url, season_id, avid, cid, fram_seq, fps, material_id
        df = pd.read_csv(tsv_file, sep='\t')
        
        for idx in range(len(df)):
            cover_url = df['cover_url'][idx]
            avid = df['avid'][idx]
            fram_seq = df['fram_seq'][idx]
            fps = df['fps'][idx]

            bvid = self.avid2bvid(avid)
            pos = self.get_pos(fram_seq, fps)
            video_url = self.get_video_url(bvid, pos)
            self._imagename2videourl[osp.basename(cover_url)] = video_url
            # print(cover_url, video_url)

    def get_pos(self, fram_seq, fps):
        # get video pos (seconds) according to fram_seq and fps
        pos = fram_seq // fps
        return pos

    def avid2bvid(self, av):
        Str = 'fZodR9XQDSUm21yCkr6zBqiveYah8bt4xsWpHnJE7jL5VG3guMTKNPAwcF'  # 准备的一串指定字符串
        s = [11, 10, 3, 8, 4, 6, 2, 9, 5, 7]  # 必要的解密列表
        xor = 177451812
        add = 100618342136696320  # 这串数字最后要被减去或加上
        ret = av
        av = int(av)
        av = (av ^ xor) + add
        # 将BV号的格式（BV + 10个字符） 转化成列表方便后面的操作
        r = list('BV          ')
        for i in range(10):
            r[s[i]] = Str[av // 58 ** i % 58]
        return ''.join(r)

    def get_video_url(self, bvid, time_seconds):
        return f'https://www.bilibili.com/video/{bvid}?t={time_seconds}'

if __name__ == '__main__':
    print('loading')
    video_jumper = VideoJumper('./ogv_cover_info.tsv')
    print('look up')
    videourl = video_jumper.imagename2videourl('bb3df189b4b6ae55630d4b963bb6eff3c775956b.jpg')
    print(videourl)

