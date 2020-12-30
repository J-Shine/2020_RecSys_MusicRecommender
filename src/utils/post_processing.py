from tqdm import tqdm
import numpy as np

MAX_PLAYLISTS = 153430

def eurm_to_recommendation_list(eurm, cat='all', remove_seed=True, datareader=None):
    """
    Convert the eurm = (10.000, 2M) into a recommendation list if cat is set to 'all', otherwhise
    Convert the eurm = (10.000, 2M) into a recommendation list if a category is specified. #TODO @seba 1k o 10k?
    :param eurm: the estimated user rating matrix
    :param remove_seed: remove seed tracks from playlists
    :param datareader: a Datareader object for seeds removing
    :param cat: 'all' or a value between 1 and 10
    :return: recommendation_list: a list of list of recommendations of shape (10k,500)
    """

    # Convert eurm
    eurm = eurm.tocsr()

    # Remove seeds
    if datareader is None and remove_seed is True:
        print('[ WARNING! Datareader is None. It was not possible to remove seeds while converting the eurm ]')
    elif datareader is not None and remove_seed is True:
        eurm = eurm_remove_seed(eurm, datareader)
        print('Seeds removed!')


    # Initialize rec_list
    recommendation_list = [[] for x in range(23015)]
    # 처음부터 끝까지 열을 순회한다
    for row in tqdm((range(eurm.shape[0])), desc='Converting eurm'):
        # 이번에 해당하는 행을 뽑아서 val에 넣는다
        val = eurm.data[eurm.indptr[row]:eurm.indptr[row+1]]
        # 오름차순 정렬된 걸 뒤에서 100개 뽑은 후
        # 전체 범위를 가져와서 뒤에서부터 거꾸로 ind에 넣는다
        ind = val.argsort()[-100:][::-1]
        # 아까 뽑은 index를 해당 행에 적용해서 ind에 넣는다(score가 높은 순서로 트랙이 추천된다)
        ind = list(eurm[row].indices[ind])
        # 추천리스트에 추가한다
        recommendation_list[row] = ind

    return recommendation_list

   # return recommendation_list
"""
    def _generate_answers(self, train, questions):
        _, song_mp = most_popular(train, "songs", 200)
        _, tag_mp = most_popular(train, "tags", 100)

        answers = []

        for q in tqdm(questions):
            answers.append({
                "id": q["id"],
                "songs": remove_seen(q["songs"], song_mp)[:100],
                "tags": remove_seen(q["tags"], tag_mp)[:10],
            })

        return answers
        """

def eurm_remove_seed(eurm, datareader, eliminate_negative=True):
    """
    Remove seed tracks from the eurm (10K, 2M)
    :param eurm: original eurm
    :param datareader: a Datareader object, the same used to build the original eurm
    :return: eurm: eurm with no seed tracks
    """
    # Convert eurm
    eurm = eurm.tocsr()

    # Get urm with shape of eurm
    urm = datareader.get_urm(only_load=True)
    # pids = datareader.get_val_pids()
    urm_test = urm[MAX_PLAYLISTS - 1]
    max_value = eurm.max()

    new_data = np.ones(len(urm_test.data)) * max_value
    urm_test.data = new_data

    # Remove seen
    eurm = eurm - urm_test

    if eliminate_negative:
        eurm.data[eurm.data <= 0] = 0
        eurm.eliminate_zeros()

    return eurm