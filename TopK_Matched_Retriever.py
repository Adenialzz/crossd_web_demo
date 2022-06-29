import json

def read_json(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return json_data


class TopK_Matched_Retriever:
    def __init__(self, ret_res_file):
        self.json_data = read_json(ret_res_file)

    def extract_topk_matched(self, filename3d, topk):
        '''
        return value -> list: [['num_matched_tags', [filename2d_0, filename2d_1]], ...]
        '''
        matched = self.json_data[filename3d]
        srt = sorted(matched.items(), key=lambda x: int(x[0]), reverse=True)
        srt_topk = srt[: topk]
        return srt_topk

class TopKTagsRetriever:
    def __init__(self, lib2d_json, tags_thr=0.3):
        self.lib2d = read_json(lib2d_json)
        self.tags_thr = tags_thr

    def extract_tag(self, tags_result):
        tags_set = set()
        for tag in tags_result:
            if tag['score'] < self.tags_thr:
                continue
            tags_set.add(tag['tag'])
        return tags_set

    def match_tags(self, tags2d, tags3d):
        tags3d_set = self.extract_tag(tags3d)
        tags2d_set = self.extract_tag(tags2d)
        return tags3d_set & tags2d_set


    def extract_topk_matched(self, tags3d, topk):
        all_matched = {}
        for name2d, tags2d in self.lib2d.items():
            matched_tags_set = self.match_tags(tags2d, tags3d)
            num_matched = len(matched_tags_set)
            try:
                all_matched[num_matched].append(name2d)
            except KeyError:
                all_matched[num_matched] = [name2d]
        srt = sorted(all_matched.items(), key=lambda x: int(x[0]), reverse=True)
        srt_topk = srt[: topk]
        return srt_topk

if __name__ == '__main__':
    tags3d = [{"tag": "1girl", "score": 0.9788736701011658}, {"tag": "arm_up", "score": 0.1554606556892395}, {"tag": "armband", "score": 0.11676490306854248}, {"tag": "bare_shoulders", "score": 0.1724521517753601}, {"tag": "bdsm", "score": 0.17134001851081848}, {"tag": "black_gloves", "score": 0.23563823103904724}, {"tag": "black_legwear", "score": 0.2048187553882599}, {"tag": "black_panties", "score": 0.36353379487991333}, {"tag": "blonde_hair", "score": 0.8054797649383545}, {"tag": "blood", "score": 0.11417385935783386}, {"tag": "blue_eyes", "score": 0.779490053653717}, {"tag": "bondage_outfit", "score": 0.19620904326438904}, {"tag": "bra", "score": 0.2585117518901825}, {"tag": "breasts", "score": 0.4947473406791687}, {"tag": "brown_eyes", "score": 0.15610283613204956}, {"tag": "brown_hair", "score": 0.20834016799926758}, {"tag": "bustier", "score": 0.12830406427383423}, {"tag": "choker", "score": 0.651354193687439}, {"tag": "cleavage", "score": 0.22118204832077026}, {"tag": "collar", "score": 0.10266110301017761}, {"tag": "corset", "score": 0.8630435466766357}, {"tag": "dominatrix", "score": 0.2784532308578491}, {"tag": "elbow_gloves", "score": 0.8116765022277832}, {"tag": "fishnets", "score": 0.15342050790786743}, {"tag": "garter_belt", "score": 0.9299484491348267}, {"tag": "garter_straps", "score": 0.8310046792030334}, {"tag": "gloves", "score": 0.9335534572601318}, {"tag": "holding", "score": 0.11437144875526428}, {"tag": "lace", "score": 0.11476749181747437}, {"tag": "large_breasts", "score": 0.13661038875579834}, {"tag": "leather", "score": 0.294417142868042}, {"tag": "lingerie", "score": 0.8340012431144714}, {"tag": "lips", "score": 0.2702874541282654}, {"tag": "looking_at_viewer", "score": 0.1786695122718811}, {"tag": "medium_breasts", "score": 0.44474560022354126}, {"tag": "panties", "score": 0.7590448260307312}, {"tag": "realistic", "score": 0.13206565380096436}, {"tag": "short_hair", "score": 0.6529055833816528}, {"tag": "solo", "score": 0.9600651264190674}, {"tag": "standing", "score": 0.23669514060020447}, {"tag": "sword", "score": 0.22202426195144653}, {"tag": "thighhighs", "score": 0.8346762657165527}, {"tag": "underwear", "score": 0.912589967250824}, {"tag": "weapon", "score": 0.39420104026794434}, {"tag": "rating:safe", "score": 0.5210904479026794}, {"tag": "rating:questionable", "score": 0.4812251925468445}]

    K = 3
    tags_retriever = TopKTagsRetriever('all_2d_tags.json')
    srt_topk = tags_retriever.extract_topk_matched(tags3d, K)
    for item in srt_topk:
        # print(item)
        print(item[0], len(item[1]))

    print('*'*42)
    name3d = 'Naomi_Armitage_2_3D_0.jpg'
    r = TopK_Matched_Retriever('./ret_original3d_all_tags.json')
    t = r.extract_topk_matched(name3d, K)
    for item in t:
        # print(item)
        print(item[0], len(item[1]))
