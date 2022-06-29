import pandas as pd

def load_info(tsv_file):
    # cover_url, season_id, avid, cid, fram_seq, fps, material_id
    df = pd.read_csv(tsv_file, sep='\t')
    
    for idx in range(len(df)):
        cover_url = df['cover_url'][idx]
        avid = df['avid'][idx]
        fram_seq = df['fram_seq'][idx]
        fps = df['fps'][idx]

        bvid = algorithm_enc(avid)
        pos = get_pos(fram_seq, fps)
        video_url = get_video_url(bvid, pos)
        print(cover_url, video_url)

def get_pos(fram_seq, fps):
    # get video pos (seconds) according to fram_seq and fps
    pos = fram_seq // fps
    return pos


def algorithm_enc(av):
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

def get_video_url(bvid, time_seconds):
    return f'https://www.bilibili.com/video/{bvid}?t={time_seconds}'

if __name__ == '__main__':
    avid = '111188974'
    bvid = algorithm_enc(avid)
    url = get_video_url(bvid, 1)
    print(url)

    load_info('./ogv_cover_info.tsv')
